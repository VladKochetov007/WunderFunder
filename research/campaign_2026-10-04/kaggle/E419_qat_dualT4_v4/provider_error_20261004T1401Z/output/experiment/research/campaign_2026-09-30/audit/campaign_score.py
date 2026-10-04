"""Frozen clean-panel scorer for explicitly preregistered prediction mixtures.

Weights, members, epochs, comparisons and folds come from a prior design JSON.
Artifact identities come from a separate post-fit specification. Neither input
is changed, and output creation is exclusive. Stored NPZ target arrays and NPY
predictions are memory mapped; only one sequence of floating arrays is copied.
"""

from __future__ import annotations

import argparse
import json
import resource
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from wnn.evaluation import paired_component_bootstrap  # noqa: E402
from wnn.gmetric import _merge_moments, _wp_from_moments  # noqa: E402
from wnn.metric import block_scores  # noqa: E402
from wnn.splits import file_sha256, load_campaign_split  # noqa: E402

METRIC_ID = "global_wp_clip2_v1"
SOFTENED = (0.02, 0.05)


def artifact(identity: dict[str, str]) -> Path:
    """Validate immutable bytes before using an artifact."""
    path = (ROOT / identity["path"]).resolve()
    if not path.is_file() or file_sha256(path) != identity["sha256"]:
        raise ValueError(f"artifact missing or hash mismatch: {path}")
    return path


def mmap_stored_npz(path: Path) -> dict[str, np.memmap]:
    """Map NPY members inside an uncompressed NPZ without extracting arrays."""
    arrays = {}
    with zipfile.ZipFile(path) as archive, path.open("rb") as raw:
        for info in archive.infolist():
            if info.compress_type != zipfile.ZIP_STORED or not info.filename.endswith(".npy"):
                raise ValueError("target cache must contain only stored NPY members")
            with archive.open(info) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError("unsupported target-cache NPY version")
                header_bytes = member.tell()
            if dtype.hasobject:
                raise ValueError("object target arrays are not permitted")
            raw.seek(info.header_offset)
            header = raw.read(30)
            if header[:4] != b"PK\x03\x04":
                raise ValueError("invalid NPZ local header")
            filename_bytes, extra_bytes = struct.unpack("<HH", header[26:30])
            offset = info.header_offset + 30 + filename_bytes + extra_bytes + header_bytes
            arrays[info.filename.removesuffix(".npy")] = np.memmap(
                path,
                dtype=dtype,
                shape=shape,
                offset=offset,
                mode="r",
                order="F" if fortran else "C",
            )
    required = {"y", "mask", "row_group", "market_class"}
    if not required.issubset(arrays):
        raise ValueError("target cache lacks required arrays")
    n = len(arrays["row_group"])
    if arrays["y"].shape != (n, 20_000, 2) or arrays["mask"].shape != (n, 20_000):
        raise ValueError("incomplete target sequence cache")
    if arrays["market_class"].shape != (n,) or len(np.unique(arrays["row_group"])) != n:
        raise ValueError("invalid target row/class index")
    return arrays


def moments(y: np.ndarray, p: np.ndarray, selection: np.ndarray) -> np.ndarray:
    """Compute clipped, centered moments with exact or softened selection."""
    yy, pp = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    inclusion = np.asarray(selection, dtype=np.float64)
    if yy.shape != pp.shape or inclusion.shape != yy.shape:
        raise ValueError("unaligned moment inputs")
    if not np.isfinite(inclusion).all() or np.any(inclusion < 0):
        raise ValueError("invalid inclusion weights")
    selected = inclusion > 0
    if not np.isfinite(yy[selected]).all() or not np.isfinite(pp[selected]).all():
        raise ValueError("nonfinite raw selected target or prediction")
    yy, pp = np.clip(yy[selected], -2, 2), np.clip(pp[selected], -2, 2)
    weight = inclusion[selected] * np.abs(yy)
    active = weight > 0
    if not active.any():
        return np.zeros(6, dtype=np.float64)
    yy, pp, weight = yy[active], pp[active], weight[active]
    total = weight.sum()
    dy, dp = yy - yy[0], pp - pp[0]
    my, mp = np.dot(weight, dy) / total, np.dot(weight, dp) / total
    dy, dp = dy - my, dp - mp
    return np.array(
        [
            total,
            yy[0] + my,
            pp[0] + mp,
            np.dot(weight, dy * dy),
            np.dot(weight, dp * dp),
            np.dot(weight, dy * dp),
        ],
        dtype=np.float64,
    )


def pooled(items: np.ndarray) -> dict[str, float]:
    """Merge per-sequence centered moments instead of summing raw squares."""
    merged = np.zeros((2, 6), dtype=np.float64)
    for item in items:
        for target in range(2):
            merged[target] = _merge_moments(merged[target], item[target])
    scores = [_wp_from_moments(value) for value in merged]
    return {"t0": scores[0], "t1": scores[1], "wp": float(np.mean(scores))}


def weights(values: list[float]) -> None:
    """Require a finite, fixed convex mixture."""
    if (
        not values
        or not np.isfinite(values).all()
        or min(values) < 0
        or abs(sum(values) - 1) > 1e-12
    ):
        raise ValueError("mixture weights must be finite, nonnegative and sum to one")


def validate_member(
    declared: dict,
    identities: dict,
    split: object,
    manifest: dict,
    row_groups: list[int],
) -> tuple[np.memmap, dict]:
    """Bind prediction rows to exact checkpoints, clean fit provenance and jobs."""
    paths = {
        key: artifact(identities[key])
        for key in ("prediction", "checkpoint", "sidecar", "provenance", "job")
    }
    if str(paths["checkpoint"].relative_to(ROOT)) != declared["checkpoint"]:
        raise ValueError("checkpoint path differs from preregistration")
    provenance = json.loads(paths["provenance"].read_text())
    sidecar = json.loads(paths["sidecar"].read_text())
    job = json.loads(paths["job"].read_text())
    args = provenance["args"]
    run, epoch = declared["run"], declared["epoch"]
    if (
        args.get("name") != run
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != epoch
        or not args.get("no_eval_during_fit")
        or provenance.get("selected_epoch") != epoch
        or provenance.get("split_manifest_sha256") != split.sha256
    ):
        raise ValueError(f"{run}: fixed epoch/clean split provenance mismatch")
    if args.get("init"):
        initial_path = artifact(identities["initial_checkpoint"])
        if (
            provenance.get("initial_checkpoint_sha256")
            != identities["initial_checkpoint"]["sha256"]
        ):
            raise ValueError(f"{run}: initializer checkpoint binding mismatch")
        initial_run, separator, initial_epoch = args["init"].partition("@ep")
        initial_name = f"model_ep{int(initial_epoch)}.pt" if separator else "model.pt"
        if initial_path != ROOT / "artifacts/runs" / initial_run / initial_name:
            raise ValueError(f"{run}: initializer path differs from recorded init")
    fit = provenance["fit_row_groups"]
    split.assert_fit_groups(fit.get("train", []), fit.get("valid", []))
    if declared["fit_source"] == "train+valid":
        if set(fit["train"]) != set(manifest["fit"]["train"]):
            raise ValueError(f"{run}: incomplete or changed TRAIN fit membership")
    elif declared["fit_source"] == "valid":
        if fit.get("train"):
            raise ValueError(f"{run}: unexpected TRAIN fitting rows")
    else:
        raise ValueError("preregistered fit_source must be train+valid or valid")
    if set(fit["valid"]) != set(manifest["fit"]["valid"]) or set(
        provenance["evaluation_row_groups"]
    ) != set(split.evaluation_groups):
        raise ValueError(f"{run}: changed fit/evaluation rows")
    if (
        provenance["checkpoint_sha256"].get(paths["checkpoint"].name)
        != identities["checkpoint"]["sha256"]
    ):
        raise ValueError(f"{run}: checkpoint bytes differ from fit provenance")
    for key, expected in declared.get("expected_args", {}).items():
        if args.get(key) != expected:
            raise ValueError(f"{run}: training argument changed: {key}")
    if (
        sidecar.get("run") != run
        or sidecar.get("split") != "valid"
        or sidecar.get("epoch") != epoch
        or sidecar.get("gate_mode") != "soft"
        or sidecar.get("row_groups") != row_groups
        or sidecar.get("split_manifest_sha256") != split.sha256
        or sidecar.get("checkpoint_sha256") != identities["checkpoint"]["sha256"]
        or sidecar.get("prediction_sha256") != identities["prediction"]["sha256"]
        or sidecar.get("quote_cache") != provenance.get("quote_cache")
    ):
        raise ValueError(f"{run}: prediction sidecar linkage mismatch")
    if (
        job.get("name") != declared.get("job_name", run)
        or job.get("status") != "finished"
        or job.get("exit_code") != 0
        or job.get("kind") != "gpu"
        or job.get("peak_observed_vram_bytes", 0) > 5_000_000_000
    ):
        raise ValueError(f"{run}: successful bounded GPU fit record missing")
    for key, expected in declared.get("source_sha256", {}).items():
        if job.get("source_sha256", {}).get(key) != expected:
            raise ValueError(f"{run}: training source identity mismatch: {key}")
    prediction = np.load(paths["prediction"], mmap_mode="r", allow_pickle=False)
    if prediction.shape != (len(row_groups), 20_000, 2) or prediction.dtype != np.float32:
        raise ValueError(f"{run}: malformed/incomplete prediction array")
    return prediction, {
        "run": run,
        "epoch": epoch,
        "identities": identities,
        "fit_job_elapsed_seconds": job.get("elapsed_seconds"),
    }


def score_sequence(y: np.ndarray, mask: np.ndarray, predictions: dict[str, np.ndarray]) -> dict:
    """Score one dense timeline with explicit required-row semantics."""
    y = np.asarray(y, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if y.ndim != 2 or y.shape[1] != 2 or mask.shape != y.shape[:1]:
        raise ValueError("malformed sequence target/mask")
    need = np.arange(len(y)) >= 99
    if not np.isfinite(y[need]).all():
        raise ValueError("nonfinite required target")
    exact_mask = need & mask
    exact, soft, blocks = {}, {str(lam): {} for lam in SOFTENED}, {}
    for name, prediction in predictions.items():
        p = np.asarray(prediction, dtype=np.float32)
        if p.shape != y.shape or not np.isfinite(p).all():
            raise ValueError(f"{name}: malformed or nonfinite raw prediction")
        exact[name] = np.stack([moments(y[:, k], p[:, k], exact_mask) for k in range(2)])
        for lam in SOFTENED:
            inclusion = np.where(mask, 1.0, lam) * need
            soft[str(lam)][name] = np.stack(
                [moments(y[:, k], p[:, k], inclusion) for k in range(2)]
            )
        values = block_scores(y[None], p[None], exact_mask[None])[0]
        blocks[name] = float(np.mean(values)) if np.isfinite(values).all() else None
    return {
        "moments": exact,
        "soft": soft,
        "blocks": blocks,
        "selected_rows": int(exact_mask.sum()),
    }


def summary(
    m: dict[str, np.ndarray],
    soft: dict,
    blocks: dict,
    classes: np.ndarray,
    components: np.ndarray,
    design: dict,
) -> dict:
    """Report exact effects and paired uncertainty from small moment vectors."""
    exact = {name: pooled(value) for name, value in m.items()}
    comparisons = {}
    for candidate, control in design["comparisons"]:
        value = paired_component_bootstrap(
            m[candidate],
            m[control],
            components,
            n_boot=design["n_boot"],
            seed=design["bootstrap_seed"],
            centered=True,
        )
        if abs(value["mean_diff"] - exact[candidate]["wp"] + exact[control]["wp"]) > 1e-10:
            raise ValueError("bootstrap and pooled score mismatch")
        comparisons[f"{candidate}_minus_{control}"] = value
    return {
        "exact_global_wp": exact,
        "paired_component_bootstrap": comparisons,
        "by_class": {
            str(cls): {name: pooled(value[classes == cls]) for name, value in m.items()}
            for cls in np.unique(classes)
        },
        "softened_mask_sensitivity_not_official": {
            lam: {name: pooled(value) for name, value in variants.items()}
            for lam, variants in soft.items()
        },
        "legacy_sequence_mean_not_official": {
            name: {
                "eligible_sequences": sum(v is not None for v in values),
                "wp": float(np.mean([v for v in values if v is not None]))
                if any(v is not None for v in values)
                else None,
            }
            for name, values in blocks.items()
        },
    }


def score_spec(spec_path: Path) -> dict:
    """Validate frozen design and score explicit immutable fold artifacts."""
    spec = json.loads(spec_path.read_text())
    freeze_path = artifact(spec["scorer_freeze"])
    freeze = json.loads(freeze_path.read_text())
    for source, expected in freeze["files"].items():
        artifact({"path": source, "sha256": expected})
    prior_path = artifact(spec["preregistration"])
    design = json.loads(prior_path.read_text())["design"]
    if design != spec["design"] or design["metric_id"] != METRIC_ID:
        raise ValueError("design changed after preregistration")
    if design["n_boot"] < 2000 or design["bootstrap_seed"] != 271828:
        raise ValueError("use at least 2000 resamples and frozen bootstrap seed271828")
    cache_path = artifact(spec["targets"])
    cache = mmap_stored_npz(cache_path)
    groups = cache["row_group"].astype(int).tolist()
    positions = {group: i for i, group in enumerate(groups)}
    collected, collected_soft, collected_blocks = {}, {}, {}
    classes_all, components_all, folds_all = [], [], []
    fold_reports, bindings = {}, {}
    seen_groups = set()
    for fold in design["folds"]:
        manifest_path = artifact(fold["manifest"])
        split = load_campaign_split(manifest_path, ROOT)
        if split.fold != fold["fold"]:
            raise ValueError("declared fold differs from clean manifest")
        manifest = json.loads(manifest_path.read_text())
        if seen_groups & set(split.evaluation_groups):
            raise ValueError("evaluation rows duplicated across folds")
        seen_groups.update(split.evaluation_groups)
        loaded, bindings[str(split.fold)] = {}, {}
        for role, members in fold["roles"].items():
            weights([member["weight"] for member in members])
            loaded[role], bindings[str(split.fold)][role] = [], []
            for member in members:
                values, binding = validate_member(
                    member,
                    spec["artifacts"][str(split.fold)][member["run"]],
                    split,
                    manifest,
                    groups,
                )
                loaded[role].append((member["weight"], values))
                bindings[str(split.fold)][role].append(binding)
        for output, mixture in design.get("derived_predictions", {}).items():
            weights(list(mixture.values()))
            if output in loaded or not set(mixture).issubset(loaded):
                raise ValueError("derived prediction must combine existing base roles")
        moment_rows, soft_rows, block_rows = {}, {str(lam): {} for lam in SOFTENED}, {}
        class_rows, component_rows = [], []
        component_map = {g: c for source, g, c in split.components if source == "valid"}
        selected_rows = 0
        for group in split.evaluation_groups:
            position = positions[group]
            prediction = {}
            for role, members in loaded.items():
                mix = np.zeros((20_000, 2), dtype=np.float64)
                for weight, values in members:
                    raw = values[position]
                    if not np.isfinite(raw).all():
                        raise ValueError(f"{role}: nonfinite member before mixing/clipping")
                    mix += weight * raw
                prediction[role] = mix.astype(np.float32)
            for output, mixture in design.get("derived_predictions", {}).items():
                prediction[output] = sum(
                    weight * prediction[role].astype(np.float64) for role, weight in mixture.items()
                ).astype(np.float32)
            row = score_sequence(cache["y"][position], cache["mask"][position], prediction)
            for name, value in row["moments"].items():
                moment_rows.setdefault(name, []).append(value)
                block_rows.setdefault(name, []).append(row["blocks"][name])
                for lam in soft_rows:
                    soft_rows[lam].setdefault(name, []).append(row["soft"][lam][name])
            selected_rows += row["selected_rows"]
            class_rows.append(int(cache["market_class"][position]))
            component_rows.append(component_map[group])
        if selected_rows == 0 or not np.isin(class_rows, range(4)).all():
            raise ValueError("empty scored fold or invalid retrospective classes")
        m = {name: np.stack(values) for name, values in moment_rows.items()}
        soft = {
            lam: {name: np.stack(values) for name, values in variants.items()}
            for lam, variants in soft_rows.items()
        }
        fold_reports[str(split.fold)] = {
            "manifest_sha256": split.sha256,
            "n_sequences": len(class_rows),
            "selected_rows": selected_rows,
            **summary(m, soft, block_rows, np.array(class_rows), np.array(component_rows), design),
        }
        for name, value in m.items():
            collected.setdefault(name, []).append(value)
            collected_blocks.setdefault(name, []).extend(block_rows[name])
        for lam, variants in soft.items():
            for name, value in variants.items():
                collected_soft.setdefault(lam, {}).setdefault(name, []).append(value)
        classes_all.extend(class_rows)
        components_all.extend(component_rows)
        folds_all.extend([split.fold] * len(class_rows))
    m = {name: np.concatenate(values) for name, values in collected.items()}
    soft = {
        lam: {name: np.concatenate(values) for name, values in variants.items()}
        for lam, variants in collected_soft.items()
    }
    if not m:
        raise ValueError("no frozen evaluation folds")
    return {
        "metric_id": METRIC_ID,
        "preregistration_sha256": spec["preregistration"]["sha256"],
        "spec_sha256": file_sha256(spec_path),
        "targets_sha256": spec["targets"]["sha256"],
        "script_sha256": file_sha256(Path(__file__)),
        "prediction_bindings": bindings,
        "selection": "row_index>=99 AND is_scored; complete dense sequences",
        "n_sequences": len(classes_all),
        "n_components": len(set(components_all)),
        "folds": fold_reports,
        "pooled": summary(
            m, soft, collected_blocks, np.array(classes_all), np.array(components_all), design
        ),
        "boundaries": [
            "Previously inspected clean folds; none called untouched.",
            "Softened masks and sequence means are diagnostics, not hidden-mask estimates.",
            "Local effects do not establish expected public score or promote a candidate.",
            "Shared components across folds are resampled together in pooled uncertainty.",
        ],
    }


def main() -> None:
    """Write one immutable report from the prior frozen design and fit identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.monotonic()
    result = score_spec(args.spec.resolve())
    result["resources"] = {
        "wall_seconds": time.monotonic() - start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "exit_code": 0,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({"output": str(args.output), "n_sequences": result["n_sequences"]}), flush=True
    )


if __name__ == "__main__":
    main()
