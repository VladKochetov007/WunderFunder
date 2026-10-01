"""Frozen, provenance-checked Global WP screen for the fold-0 clock auxiliary.

Reads predictions only after candidate/control fits and fixed-epoch exports
exist. This script never trains, selects epochs, or changes blend weights.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from replay import HERE, ROOT, digest

from wnn.evaluation import paired_component_bootstrap
from wnn.gmetric import _centered_moments, _merge_moments, _wp_from_moments
from wnn.metric import block_scores
from wnn.splits import file_sha256

PAIR_NAMES = ("candidate_pair", "control_pair", "noaux_pair")
SINGLE_NAMES = ("candidate_single", "control_single")
ALL_NAMES = PAIR_NAMES + SINGLE_NAMES
SOFTENED = (0.02, 0.05)
MODEL_SOURCE_KEYS = (
    "scripts/train_rnn.py", "src/wnn/nn.py", "src/wnn/features.py",
    "src/wnn/outcomes.py", "src/wnn/resources.py", "src/wnn/splits.py",
)


def required_model_provenance(
    name: str,
    variant: str | None,
    manifest_sha: str,
    manifest: dict,
    cache_meta: dict,
    cache_meta_sha: str,
    expected_row_groups: list[int],
) -> tuple[np.memmap, dict, dict, dict]:
    """Validate checkpoint, prediction sidecar, and cache linkage before opening rows."""
    checkpoint_path = ROOT / "artifacts/runs" / name / "model_ep3.pt"
    prediction_path = ROOT / "cache" / f"pred_{name}_valid.npy"
    sidecar_path = prediction_path.with_suffix(".json")
    job_path = ROOT / "research/campaign_2026-09-29/jobs" / f"{name}.json"
    if not all(path.is_file() for path in (checkpoint_path, prediction_path, sidecar_path, job_path)):
        raise FileNotFoundError(f"missing fixed model, prediction, sidecar, or durable job for {name}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sidecar = json.loads(sidecar_path.read_text())
    job = json.loads(job_path.read_text())
    checkpoint_args = checkpoint["args"]
    provenance = checkpoint["provenance"]
    if (checkpoint.get("epoch") != 3 or checkpoint_args.get("checkpoint_policy") != "fixed"
            or checkpoint_args.get("checkpoint_epoch") != 3
            or not checkpoint_args.get("no_eval_during_fit")
            or checkpoint_args.get("split_manifest_sha256") != manifest_sha
            or provenance.get("split_manifest_sha256") != manifest_sha
            or checkpoint_args.get("batch") != 32):
        raise ValueError(f"{name}: checkpoint violates frozen clean-fold recipe")
    if (set(provenance["fit_row_groups"]["train"]) != set(manifest["fit"]["train"])
            or set(provenance["fit_row_groups"]["valid"]) != set(manifest["fit"]["valid"])
            or set(provenance["evaluation_row_groups"]) != set(manifest["evaluation"]["valid"])):
        raise ValueError(f"{name}: checkpoint fit/evaluation row groups differ from manifest")
    if (sidecar.get("run") != name or sidecar.get("split") != "valid"
            or sidecar.get("epoch") != 3 or sidecar.get("gate_mode") != "soft"
            or sidecar.get("split_manifest_sha256") != manifest_sha
            or sidecar.get("row_groups") != expected_row_groups
            or sidecar.get("checkpoint_sha256") != file_sha256(checkpoint_path)
            or sidecar.get("prediction_sha256") != file_sha256(prediction_path)):
        raise ValueError(f"{name}: prediction sidecar/checkpoint/file hash mismatch")
    if job.get("status") != "finished" or job.get("exit_code") != 0 or job.get("kind") != "gpu":
        raise ValueError(f"{name}: no successful recorded GPU fit")
    if job.get("peak_observed_vram_bytes", 0) > 5_000_000_000:
        raise ValueError(f"{name}: observed VRAM exceeded the campaign cap")
    quote_cache = provenance.get("quote_cache")
    if sidecar.get("quote_cache") != quote_cache:
        raise ValueError(f"{name}: prediction cache provenance differs from checkpoint")
    if variant is None:
        if quote_cache is not None or checkpoint_args.get("aux_quote", 0) != 0:
            raise ValueError(f"{name}: no-aux reference unexpectedly has quote supervision")
    else:
        expected = {
            split: cache_meta["files"][f"{variant}/{split}"]["sha256"]
            for split in ("train", "valid")
        }
        if (quote_cache is None or quote_cache.get("variant") != variant
                or quote_cache.get("metadata_sha256") != cache_meta_sha
                or quote_cache.get("array_sha256_expected") != expected
                or checkpoint_args.get("aux_quote") != 0.1):
            raise ValueError(f"{name}: auxiliary cache provenance differs from frozen variant")
    prediction = np.load(prediction_path, mmap_mode="r")
    if prediction.shape != (len(expected_row_groups), 20_000, 2) or prediction.dtype != np.float32:
        raise ValueError(f"{name}: prediction array shape or dtype differs")
    return prediction, checkpoint_args, sidecar, job


def centered_weighted_moments(
    target: np.ndarray,
    prediction: np.ndarray,
    selection_weight: np.ndarray,
) -> np.ndarray:
    """Weighted Welford-compatible moments with a softened selection weight."""
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    selection = np.asarray(selection_weight, dtype=np.float64)
    positive = (selection > 0) & (np.abs(y) > 0)
    if not positive.any():
        return np.zeros(6, dtype=np.float64)
    if not np.isfinite(y[positive]).all() or not np.isfinite(p[positive]).all():
        raise ValueError("selected raw target or prediction is nonfinite")
    y = np.clip(y[positive], -2, 2)
    p = np.clip(p[positive], -2, 2)
    weight = selection[positive] * np.abs(y)
    active = weight > 0
    y, p, weight = y[active], p[active], weight[active]
    if not len(y):
        return np.zeros(6, dtype=np.float64)
    total = weight.sum()
    ys, ps = y - y[0], p - p[0]
    my, mp = np.dot(weight, ys) / total, np.dot(weight, ps) / total
    yc, pc = ys - my, ps - mp
    return np.array([
        total, y[0] + my, p[0] + mp,
        np.dot(weight, yc * yc), np.dot(weight, pc * pc), np.dot(weight, yc * pc),
    ], dtype=np.float64)


def score_moments(moments: np.ndarray) -> dict[str, float]:
    pooled = np.zeros((2, 6), dtype=np.float64)
    for item in moments:
        for target in range(2):
            pooled[target] = _merge_moments(pooled[target], item[target])
    values = [_wp_from_moments(pooled[target]) for target in range(2)]
    return {"t0": values[0], "t1": values[1], "wp": 0.5 * sum(values)}


def canonical_training_args(arguments: dict) -> dict:
    """Compare frozen training settings while excluding arm identity and cache path."""
    return {key: value for key, value in arguments.items()
            if key not in ("name", "quote_cache_root")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="c20260929_auxclock_quote8_f0")
    parser.add_argument("--control", default="c20260929_auxclock_code8_f0")
    parser.add_argument("--anchor", default="c20260929_since_s4_f0")
    parser.add_argument("--no-aux", default="c20260929_since_s3_f0")
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.boot < 2000 or args.seed != 271828:
        parser.error("the preregistered screen uses at least 2000 bootstraps and seed 271828")
    if args.output.exists():
        raise FileExistsError(args.output)

    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    manifest = json.loads(split_path.read_text())
    manifest_sha = digest(split_path)
    screen_contract_path = HERE / "gpu_screen_contract.json"
    screen_contract = json.loads(screen_contract_path.read_text())
    if screen_contract["screen_manifest"] != str(split_path.relative_to(ROOT)):
        raise ValueError("screen contract/manifest mismatch")
    cache_meta_path = HERE / "cache_f0_v3/metadata.json"
    cache_meta = json.loads(cache_meta_path.read_text())
    cache_meta_sha = digest(cache_meta_path)
    cache_verified_path = HERE / "cache_verification_f0_v3.json"
    cache_verified = json.loads(cache_verified_path.read_text())
    if (cache_meta["split_manifest_sha256"] != manifest_sha
            or cache_verified["cache_metadata_sha256"] != cache_meta_sha
            or cache_verified["resources"]["exit_code"] != 0):
        raise ValueError("paired cache lacks valid independently checked provenance")

    target_cache_path = ROOT / "cache/valid_targets.npz"
    with np.load(target_cache_path) as cache:
        targets = cache["y"]
        masks = cache["mask"].astype(bool)
        classes = cache["market_class"]
        row_groups = cache["row_group"].astype(int).tolist()
    if (targets.shape != (len(row_groups), 20_000, 2)
            or masks.shape != targets.shape[:2]
            or classes.shape != (len(row_groups),)
            or len(set(row_groups)) != len(row_groups)):
        raise ValueError("validation target cache is malformed")
    positions = {group: i for i, group in enumerate(row_groups)}
    evaluation = [int(group) for group in manifest["evaluation"]["valid"]]
    if len(evaluation) != len(set(evaluation)) or not set(evaluation).issubset(positions):
        raise ValueError("evaluation row groups absent or duplicated")
    components = {int(item["row_group"]): int(item["component"])
                  for item in manifest["sequences"]["valid"]}
    if set(evaluation) & set(manifest["fit"]["valid"]):
        raise ValueError("evaluation group entered fit pool")
    train_components = {int(item["component"]) for item in manifest["sequences"]["train"]
                        if int(item["row_group"]) in set(manifest["fit"]["train"])}
    if {components[group] for group in evaluation} & train_components:
        raise ValueError("evaluation component linked to fit TRAIN")

    requested = {
        "candidate": (args.candidate, "quote8"),
        "control": (args.control, "code"),
        "anchor": (args.anchor, None),
        "noaux": (args.no_aux, None),
    }
    if (args.candidate != screen_contract["candidate"]["name"]
            or args.control != screen_contract["control"]["name"]):
        raise ValueError("candidate/control names changed after preregistration")
    loaded = {
        key: required_model_provenance(name, variant, manifest_sha, manifest,
                                       cache_meta, cache_meta_sha, row_groups)
        for key, (name, variant) in requested.items()
    }
    fits = {key: value[1] for key, value in loaded.items()}
    if canonical_training_args(fits["candidate"]) != canonical_training_args(fits["control"]):
        raise ValueError("candidate/control training arguments differ beyond name and cache root")
    if fits["candidate"].get("seed") != 3 or fits["anchor"].get("seed") != 4:
        raise ValueError("screen seed combination differs from frozen controls")
    code_job, quote_job = loaded["control"][3], loaded["candidate"][3]
    for key in MODEL_SOURCE_KEYS:
        if code_job["source_sha256"].get(key) != quote_job["source_sha256"].get(key):
            raise ValueError(f"candidate/control training source differs: {key}")

    selected_order = [positions[group] for group in evaluation]
    n = len(selected_order)
    moments = {name: np.zeros((n, 2, 6), dtype=np.float64) for name in ALL_NAMES}
    soft = {lam: {name: np.zeros((n, 2, 6), dtype=np.float64) for name in PAIR_NAMES}
            for lam in SOFTENED}
    legacy = {name: [] for name in ALL_NAMES}
    class_values = np.empty(n, dtype=np.int8)
    component_values = np.empty(n, dtype=np.int64)
    scored_rows = 0
    need = np.arange(20_000) >= 99
    for index, (group, position) in enumerate(zip(evaluation, selected_order, strict=True)):
        y = targets[position]
        scored = masks[position] & need
        if not scored.any() or not np.isfinite(y[scored]).all():
            raise ValueError(f"evaluation group {group} has no valid scored targets")
        clipped_y = np.clip(y, -2, 2).astype(np.float64)
        raw = {key: loaded[key][0][position] for key in requested}
        for key, prediction in raw.items():
            if not np.isfinite(prediction[need]).all():
                raise ValueError(f"{key}: nonfinite required prediction in group {group}")
        anchor = raw["anchor"].astype(np.float64)
        predictions = {
            "candidate_pair": (0.5 * (anchor + raw["candidate"])).astype(np.float32),
            "control_pair": (0.5 * (anchor + raw["control"])).astype(np.float32),
            "noaux_pair": (0.5 * (anchor + raw["noaux"])).astype(np.float32),
            "candidate_single": raw["candidate"],
            "control_single": raw["control"],
        }
        for name, prediction in predictions.items():
            clipped_p = np.clip(prediction, -2, 2).astype(np.float64)
            for target in range(2):
                moments[name][index, target] = _centered_moments(
                    clipped_y[scored, target], clipped_p[scored, target]
                )
            legacy[name].append(block_scores(y[None], prediction[None], scored[None])[0])
            if name in PAIR_NAMES:
                for lam in SOFTENED:
                    selection = np.where(masks[position], 1.0, lam) * need
                    for target in range(2):
                        soft[lam][name][index, target] = centered_weighted_moments(
                            y[:, target], prediction[:, target], selection
                        )
        scored_rows += int(scored.sum())
        class_values[index] = int(classes[position])
        component_values[index] = components[group]
    if not np.isin(class_values, range(4)).all():
        raise ValueError("invalid retrospective market class")

    exact = {name: score_moments(value) for name, value in moments.items()}
    bootstrap = {
        "candidate_minus_control": paired_component_bootstrap(
            moments["candidate_pair"], moments["control_pair"], component_values,
            n_boot=args.boot, seed=args.seed, centered=True,
        ),
        "candidate_minus_noaux": paired_component_bootstrap(
            moments["candidate_pair"], moments["noaux_pair"], component_values,
            n_boot=args.boot, seed=args.seed, centered=True,
        ),
    }
    for comparison, reference in (("candidate_minus_control", "control_pair"),
                                  ("candidate_minus_noaux", "noaux_pair")):
        observed = exact["candidate_pair"]["wp"] - exact[reference]["wp"]
        if abs(bootstrap[comparison]["mean_diff"] - observed) > 1e-10:
            raise ValueError(f"{comparison}: bootstrap scorer disagrees with pooled moments")
    class_diagnostics = {
        str(cls): {
            "n_sequences": int(np.sum(class_values == cls)),
            **{name: score_moments(value[class_values == cls]) for name, value in moments.items()},
        }
        for cls in range(4) if np.any(class_values == cls)
    }
    for value in class_diagnostics.values():
        value["candidate_minus_control"] = value["candidate_pair"]["wp"] - value["control_pair"]["wp"]
        value["candidate_minus_noaux"] = value["candidate_pair"]["wp"] - value["noaux_pair"]["wp"]
    soft_diagnostics = {
        str(lam): {name: score_moments(value) for name, value in variants.items()}
        for lam, variants in soft.items()
    }
    for value in soft_diagnostics.values():
        value["candidate_minus_control"] = value["candidate_pair"]["wp"] - value["control_pair"]["wp"]
        value["candidate_minus_noaux"] = value["candidate_pair"]["wp"] - value["noaux_pair"]["wp"]
    legacy_diagnostics = {}
    for name, values in legacy.items():
        arr = np.asarray(values)
        eligible = np.isfinite(arr).all(axis=1)
        legacy_diagnostics[name] = {"eligible_sequences": int(eligible.sum()),
                                    "wp": float(np.mean(arr[eligible])) if eligible.any() else None}
    passes = (bootstrap["candidate_minus_control"]["mean_diff"] >= 0.0005
              and bootstrap["candidate_minus_noaux"]["mean_diff"] > 0)
    output = {
        "version": "clock-screen-scorer-v1",
        "command": sys.argv,
        "script_sha256": digest(Path(__file__)),
        "screen_contract_sha256": digest(screen_contract_path),
        "split_manifest_sha256": manifest_sha,
        "cache_metadata_sha256": cache_meta_sha,
        "target_cache_sha256": file_sha256(target_cache_path),
        "model_prediction_provenance": {
            key: {"run": requested[key][0], "prediction_sidecar": loaded[key][2],
                  "fit_job": {field: loaded[key][3].get(field) for field in
                              ("status", "exit_code", "elapsed_seconds", "peak_observed_vram_bytes")}}
            for key in requested
        },
        "rows": {"evaluation_sequences": n, "evaluation_components": int(len(np.unique(component_values))),
                 "need_and_scored_rows": scored_rows},
        "exact_global_wp": exact,
        "bootstrap": bootstrap,
        "by_class_exact_global_wp": class_diagnostics,
        "softened_mask_sensitivity_not_official": soft_diagnostics,
        "legacy_block_diagnostic_not_official": legacy_diagnostics,
        "passes_frozen_screen_gate": passes,
        "boundaries": [
            "Official score uses Global Weighted Pearson pooled across selected rows, clipped at +/-2 before weights.",
            "Softened masks and per-sequence block means are diagnostics only.",
            "Fold 0 and the fold-2 label-alignment panel were historically inspected; local gains are not public transfer estimates.",
            "A screen pass requires later independent clean-fold confirmation and expectation audit.",
        ],
        "resources": {"wall_seconds": time.monotonic() - started,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "exit_code": 0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"exact": exact, "bootstrap": bootstrap,
                      "passes_frozen_screen_gate": passes}, sort_keys=True))


if __name__ == "__main__":
    main()
