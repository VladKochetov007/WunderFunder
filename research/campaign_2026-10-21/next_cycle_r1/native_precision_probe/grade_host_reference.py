"""Grade four fixed TL3 endpoints after feature-only host-replay parity gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

BINDING = Path(
    "research/campaign_2026-10-21/next_cycle_r1/endpoint_diagnosis/SOURCE_BINDING_R1.json"
)
BINDING_SHA = "1988dba4f0672e1d3cd275e6e801e7ed02ba694e280b6f581b2aae21eb173698"
ADMISSION = Path(
    "research/campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_REFERENCE_AMENDMENT_R1.json"
)
ADMISSION_SHA = "9b6d09bb0732a3f6760fe66da06eddf9148e00156694a3daefeccddf874a0568"
SCREEN_PLAN_SHA = "852f64507fb9d96caa6ad7f8e82f7b7cc00a02eb7a046d7627d56eb42709bc94"
NATIVE_GRADE_SHA = "872ae52f0c83611d390e778d9666f910773e8d24e6494c12cb56fd8384788e35"
ARCHIVE_SHA = "d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569"
NATIVE_SHA = "3cb4546229e80cca81cda9cdec1b9c6146b380271077e26f9324d8a4b6ecad9a"
SAMPLE_DIR = Path(
    "/home/vlad/development/wunder/research_tmp/native-h512-precision-probe-output-20261021"
)
SAMPLE_PINS = {
    "selection_plan.json": "721992ac88538c697b08746471685cb698985fd2f370925d1f4acf0fa6ddb58a",
    "outputs_manifest.json": "4a11ba120520b901800dbf67c13242d20882320aa4156338db99b52fd10e39c7",
    "original_endpoint_prelabel_validation.json": "8f505aa0e1e3d3d44f3b10872912f8b838c9a7c3568c90a3db2f38a88eb67cf0",
    "original_endpoint_exact_callback.npy": "16a9a8d61e63655f99e29b8bda6a83d159f2e5a0301a53ec5c32716cfdccaee7",
}
MEMBERSHIP_SHA = "eacc150ce8f30bf79e749678bdd1a79d9f61c77218aaa8175fefb8224a51d184"
BINS = ((99, 500), (500, 2_000), (2_000, 10_000), (10_000, 20_000))
MODELS = (
    "legacy_cached_tl3",
    "current_host_reference",
    "h512_cp5_replacement",
    "h512_swa345_replacement",
)
PAIRS = (
    ("cp5_minus_host", "h512_cp5_replacement", "current_host_reference"),
    ("swa_minus_host", "h512_swa345_replacement", "current_host_reference"),
    ("host_minus_legacy", "current_host_reference", "legacy_cached_tl3"),
    ("swa_minus_cp5", "h512_swa345_replacement", "h512_cp5_replacement"),
)

# Any is needed for versioned JSON evidence and dynamic prediction-loader results.


def need(ok: bool, message: str) -> None:
    """Fail closed when a frozen source or endpoint differs."""
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one bounded evidence object."""
    need(path.stat().st_size <= 16_000_000, f"evidence JSON too large: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(result, dict), f"expected JSON object: {path}")
    return result


def resolve(root: Path, raw: str | Path) -> Path:
    """Resolve a bound absolute or repository-relative path."""
    path = Path(raw)
    return path.resolve(strict=True) if path.is_absolute() else (root / path).resolve(strict=True)


def load_npy(path: Path, digest: str, shape: tuple[int, ...]) -> np.ndarray:
    """Check a complete C-order float32 NPY cache and map it without copying."""
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        header = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        actual_shape, fortran, dtype = header(stream)
        offset = stream.tell()
    dtype = np.dtype(dtype)
    need(
        not fortran and tuple(actual_shape) == shape and dtype == np.float32,
        f"NPY header differs: {path}",
    )
    need(
        path.stat().st_size == offset + math.prod(shape) * dtype.itemsize,
        f"NPY coverage differs: {path}",
    )
    need(sha(path) == digest, f"NPY hash differs: {path}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def frozen_sources(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]]:
    """Validate the source binding, old grade, endpoint plan, and pinned code."""
    binding_path = resolve(root, BINDING)
    need(sha(binding_path) == BINDING_SHA, "source binding changed")
    binding = read_json(binding_path)
    need(str(root) == binding.get("root"), "project root differs from binding")
    source_paths: dict[str, Path] = {"binding": binding_path}
    for section in ("reference_files", "source_pins"):
        for name, item in binding[section].items():
            path = resolve(root, item["path"])
            need(sha(path) == item["sha256"], f"frozen {name} source changed")
            source_paths[name] = path
    plan_record = binding["reference_files"]["endpoint_screen_plan"]
    grade_record = binding["reference_files"]["native_grade"]
    need(sha(source_paths["endpoint_screen_plan"]) == SCREEN_PLAN_SHA, "endpoint plan changed")
    need(sha(source_paths["native_grade"]) == NATIVE_GRADE_SHA, "old grade changed")
    plan, grade = (
        read_json(source_paths["endpoint_screen_plan"]),
        read_json(source_paths["native_grade"]),
    )
    need(plan.get("schema") == "wnn.tl3-h512-endpoint-screen.v1", "endpoint plan schema differs")
    need(grade.get("plan_sha256") == SCREEN_PLAN_SHA, "old grade plan pin differs")
    need(grade.get("official_metric") == "global_wp_clip2_v1", "old grade metric differs")
    need(
        grade.get("input_helper_sha256")
        == binding["source_pins"]["endpoint_score_inputs"]["sha256"],
        "old grade helper differs",
    )
    need(
        plan_record["sha256"] == SCREEN_PLAN_SHA and grade_record["sha256"] == NATIVE_GRADE_SHA,
        "binding receipts differ",
    )
    return binding, plan, grade, source_paths


def host_reference(
    root: Path, admission: dict[str, Any], source: Any, groups: list[int], sequence_ids: list[int]
) -> tuple[np.ndarray, dict[str, Any], dict[str, Path]]:
    """Validate full host receipt/cache coverage and its runtime/source pins."""
    prefix = Path(str(admission["outputs_prefix"])).resolve()
    paths = {
        "prediction": Path(f"{prefix}.npy").resolve(strict=True),
        "sidecar": Path(f"{prefix}.json").resolve(strict=True),
        "receipt": Path(f"{prefix}_replay.json").resolve(strict=True),
    }
    side, receipt = read_json(paths["sidecar"]), read_json(paths["receipt"])
    expected = {
        "schema": "wnn.tl3-native-f0-replay.v1",
        "passed": True,
        "candidate": "team_tl3_original_20261018",
        "archive_sha256": ARCHIVE_SHA,
        "native_library": "v3/wnn_cpp.cpython-311-x86_64-linux-gnu.so",
        "native_library_sha256": NATIVE_SHA,
        "native_level": "v3",
        "cpu_dispatch_level": "v3",
        "quant_backend": "2",
        "member_count": 3,
        "split_manifest_sha256": source.split.sha256,
        "evaluation_groups_total": 468,
        "evaluation_groups_replayed": 468,
        "rows_total": 9_360_000,
        "warmup_rows_processed_and_suppressed": 46_332,
        "required_prediction_rows": 9_313_668,
        "reset_aba_exact_equal": True,
        "labels_or_scoring_mask_read": False,
        "data_identity_unchanged": True,
        "data_identity": source.data_identity,
    }
    for key, value in expected.items():
        need(
            receipt.get(key) == value or str(receipt.get(key)) == value,
            f"host replay {key} differs",
        )
    recipe = receipt.get("recipe", {})
    need(
        recipe.get("reset_check_performed_in_full_replay") is True
        and recipe.get("labels_or_scoring_mask_read") is False,
        "host replay recipe gates differ",
    )
    need(
        receipt.get("sequence_receipts")
        == [
            {"row_group": g, "sequence_id": s, "rows": 20_000}
            for g, s in zip(groups, sequence_ids, strict=True)
        ],
        "host sequence coverage/order differs",
    )
    prediction_sha = str(receipt.get("predictions_sha256", ""))
    side_expected = {
        "schema": "wnn.tl3-f0-prediction-cache.v1",
        "candidate": "team_tl3_original_20261018",
        "zip_sha256": ARCHIVE_SHA,
        "split_manifest_sha256": source.split.sha256,
        "prediction_sha256": prediction_sha,
    }
    for key, value in side_expected.items():
        need(bool(prediction_sha) and side.get(key) == value, f"host sidecar {key} differs")
    need(
        list(map(int, side.get("row_groups", []))) == groups
        and list(map(int, side.get("sequence_ids", []))) == sequence_ids,
        "host cache row coverage differs",
    )
    need(
        receipt.get("predictions_path") == str(paths["prediction"])
        and receipt.get("prediction_sidecar_path") == str(paths["sidecar"]),
        "host replay output paths differ",
    )
    recorded = receipt.get("replay_source_sha256", {})
    for relative, digest in admission["source_sha256"].items():
        need(
            sha(resolve(root, relative)) == digest and recorded.get(Path(relative).name) == digest,
            f"host source pin differs: {relative}",
        )
    values = load_npy(paths["prediction"], prediction_sha, (468, 20_000, 2))
    info = {
        "receipt": receipt,
        "receipt_sha256": sha(paths["receipt"]),
        "sidecar_sha256": sha(paths["sidecar"]),
        "prediction_sha256": prediction_sha,
        "paths": {k: str(v) for k, v in paths.items()},
    }
    return values, info, paths


def sampled_parity(host: np.ndarray, groups: list[int], sequence_ids: list[int]) -> dict[str, Any]:
    """Compare the eight feature-only groups to the exact R2 callback output."""
    evidence = {name: SAMPLE_DIR / name for name in SAMPLE_PINS}
    for name, path in evidence.items():
        need(sha(path) == SAMPLE_PINS[name], f"saved feature-only sample changed: {name}")
    selection, manifest, prelabel = (
        read_json(evidence[name])
        for name in (
            "selection_plan.json",
            "outputs_manifest.json",
            "original_endpoint_prelabel_validation.json",
        )
    )
    need(
        selection.get("selected_membership_sha256") == MEMBERSHIP_SHA
        and selection.get("labels_or_predictions_read") is False,
        "sample membership was not frozen label blind",
    )
    need(
        manifest.get("target_accessed") is False
        and manifest.get("status") == "FEATURE_ONLY_OUTPUTS_SAVED_BEFORE_TARGET_ACCESS",
        "sample manifest is not feature only",
    )
    need(
        prelabel.get("target_accessed") is False and prelabel.get("scoring_performed") is False,
        "sample prelabel gate accessed targets",
    )
    need(
        prelabel.get("selected_membership_sha256") == MEMBERSHIP_SHA
        and prelabel.get("array_sha256") == SAMPLE_PINS["original_endpoint_exact_callback.npy"],
        "sample prelabel pins differ",
    )
    members = selection["members"]
    need(
        len(members) == 8 and len({int(row["component"]) for row in members}) == 8,
        "sample is not eight distinct components",
    )
    positions = {g: i for i, g in enumerate(groups)}
    sequence_by_group = dict(zip(groups, sequence_ids, strict=True))
    errors, squared, cells = [], 0.0, 0
    sample = load_npy(
        evidence["original_endpoint_exact_callback.npy"],
        SAMPLE_PINS["original_endpoint_exact_callback.npy"],
        (8, 20_000, 2),
    )
    for i, member in enumerate(members):
        group, sequence_id = int(member["row_group"]), int(member["seq_ix"])
        need(
            sequence_by_group.get(group) == sequence_id,
            f"sample sequence identity differs at {group}",
        )
        delta = host[positions[group]].astype(np.float64) - sample[i].astype(np.float64)
        maximum = float(np.abs(delta).max(initial=0.0))
        errors.append({"row_group": group, "sequence_id": sequence_id, "max_abs": maximum})
        squared += float(np.square(delta).sum(dtype=np.float64))
        cells += delta.size
    maximum = max(row["max_abs"] for row in errors)
    need(maximum <= 1e-5, f"full host replay differs from exact R2 sample: {maximum}")
    return {
        "passed": True,
        "max_abs": maximum,
        "rms": math.sqrt(squared / cells),
        "tolerance": 1e-5,
        "membership_sha256": MEMBERSHIP_SHA,
        "prediction_sha256": SAMPLE_PINS["original_endpoint_exact_callback.npy"],
        "groups": errors,
    }


def frozen_predictions(
    root: Path,
    binding: dict[str, Any],
    plan: dict[str, Any],
    grade: dict[str, Any],
    groups: list[int],
    sequence_ids: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Path]]:
    """Map the legacy baseline and validate fixed CP5/SWA endpoint receipts."""
    packet = (
        root / "research/campaign_2026-10-20/submission_delivery_r1/tl3_h512_endpoint_screen_r1"
    )
    sys.path[:0] = [str(packet), str(root / "src")]
    from endpoint_score_inputs import _load_candidate, _load_prediction

    values: dict[str, np.ndarray] = {}
    receipts: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    base = binding["baseline"]
    base_path, side_path = resolve(root, base["path"]), resolve(root, base["sidecar"])
    values["legacy_cached_tl3"], side = _load_prediction(
        base_path,
        side_path,
        {"prediction_sha256": base["sha256"], "row_groups": groups, "sequence_ids": sequence_ids},
    )
    need(
        sha(base_path) == base["sha256"] and sha(side_path) == base["sidecar_sha256"],
        "legacy prediction or sidecar hash differs",
    )
    need(
        side.get("candidate") == "team_tl3_original_20261018"
        and side.get("zip_sha256") == ARCHIVE_SHA
        and side.get("split_manifest_sha256") == binding["source_pins"]["fold0_split"]["sha256"],
        "legacy sidecar source differs",
    )
    receipts["legacy_cached_tl3"] = {
        "prediction_sha256": base["sha256"],
        "sidecar_sha256": base["sidecar_sha256"],
    }
    paths.update({"legacy_prediction": base_path, "legacy_sidecar": side_path})
    for endpoint, name in (("cp5", "h512_cp5_replacement"), ("swa345", "h512_swa345_replacement")):
        rec, prior = binding["native_endpoints"][endpoint], grade["endpoint_sources"][endpoint]
        runtime = plan["runtime_paths"][endpoint]
        expected = {
            "predictions": rec["path"],
            "sidecar": rec["sidecar"],
            "replay_receipt": prior["replay_receipt"],
        }
        for field, raw_path in expected.items():
            need(
                Path(runtime[field]).resolve(strict=True) == resolve(root, raw_path),
                f"{endpoint} {field} path differs",
            )
        array, receipt = _load_candidate(root, endpoint, runtime, groups, sequence_ids)
        for field, raw_path, digest in (
            ("predictions", rec["path"], rec["sha256"]),
            ("sidecar", rec["sidecar"], rec["sidecar_sha256"]),
            ("replay_receipt", prior["replay_receipt"], prior["replay_receipt_sha256"]),
        ):
            need(sha(resolve(root, raw_path)) == digest, f"{endpoint} {field} hash differs")
        need(
            tuple(array.shape) == (468, 20_000, 2) and array.dtype == np.float32,
            f"{endpoint} cache shape differs",
        )
        values[name], receipts[name] = array, receipt
        paths.update(
            {f"{endpoint}_{field}": resolve(root, raw_path) for field, raw_path in expected.items()}
        )
    return values, receipts, paths


def single_target(moments: np.ndarray, target: int) -> np.ndarray:
    """Adapt one target's moments to the scorer's two-channel bootstrap interface."""
    return np.repeat(moments[:, target : target + 1, :], 2, axis=1)


def grade_predictions(
    targets: np.ndarray,
    mask: np.ndarray,
    components: np.ndarray,
    groups: list[int],
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Stream official scores, fixed bins, deltas, and component bootstraps."""
    from wnn.evaluation import paired_component_bootstrap, paired_component_moments
    from wnn.gmetric import GlobalAccumulator

    scores = {name: GlobalAccumulator() for name in MODELS}
    bins = {(lo, hi): {name: GlobalAccumulator() for name in MODELS} for lo, hi in BINS}
    moments = {name: np.zeros((len(groups), 2, 6), dtype=np.float64) for name in MODELS}
    positions = np.arange(20_000)
    for i, group in enumerate(groups):
        y, selected = targets[i], mask[i]
        need(
            selected.any() and np.isfinite(y[selected]).all(), f"invalid selected targets: {group}"
        )
        for name in MODELS:
            prediction = predictions[name][i]
            need(np.isfinite(prediction).all(), f"nonfinite {name} output: {group}")
            scores[name].add(y[None], prediction[None], selected[None])
            moments[name][i] = paired_component_moments(
                y[None], prediction[None], prediction[None], selected[None]
            )[0][0]
            for lo, hi in BINS:
                bins[lo, hi][name].add(
                    y[None],
                    prediction[None],
                    (selected & (positions >= lo) & (positions < hi))[None],
                )
    score_values = {name: acc.result() for name, acc in scores.items()}
    deltas = {
        label: {key: score_values[a][key] - score_values[b][key] for key in ("wp", "t0", "t1")}
        for label, a, b in PAIRS
    }
    boot = {}
    for label, a, b in PAIRS:
        boot[label] = {
            "both_targets": paired_component_bootstrap(
                moments[a], moments[b], components, n_boot=2_000, seed=271_828, centered=True
            )
        }
        for target in range(2):
            boot[label][f"t{target}"] = paired_component_bootstrap(
                single_target(moments[a], target),
                single_target(moments[b], target),
                components,
                n_boot=2_000,
                seed=271_828,
                centered=True,
            )
    bin_results = {}
    for (lo, hi), accs in bins.items():
        values = {name: acc.result() for name, acc in accs.items()}
        bin_results[f"{lo}_{hi}"] = {
            "scores": values,
            "paired_deltas": {
                label: {key: values[a][key] - values[b][key] for key in ("wp", "t0", "t1")}
                for label, a, b in PAIRS
            },
        }
    return {
        "scores": score_values,
        "paired_deltas": deltas,
        "paired_component_bootstrap": boot,
        "position_bins": bin_results,
        "component_count": int(len(np.unique(components))),
        "sequence_count": len(groups),
        "selected_rows": score_values[MODELS[0]]["selected_rows"],
    }


def main() -> int:
    """Run one create-only grade after all feature-only gates pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, started = args.root.resolve(strict=True), time.monotonic()
    binding, plan, old_grade, source_paths = frozen_sources(root)
    admission_path = resolve(root, ADMISSION)
    need(sha(admission_path) == ADMISSION_SHA, "current-host replay admission changed")
    admission = read_json(admission_path)
    need(
        admission.get("status") == "ADMITTED_ONE_FULL_CURRENT_HOST_REFERENCE_REPLAY",
        "current-host replay is not admitted",
    )

    endpoint_packet = (
        root / "research/campaign_2026-10-20/submission_delivery_r1/tl3_h512_endpoint_screen_r1"
    )
    sys.path[:0] = [str(endpoint_packet), str(root / "src")]
    from endpoint_score_inputs import load_campaign_split, load_targets

    source = load_campaign_split(root)
    groups = list(map(int, source.groups))
    sequence_ids = [int(source.sequence_by_group[group]) for group in groups]
    need(
        len(groups) == len(set(groups)) == 468 and len(set(sequence_ids)) == 468,
        "F0 source coverage differs",
    )
    cached, cached_receipts, cached_paths = frozen_predictions(
        root, binding, plan, old_grade, groups, sequence_ids
    )
    host, host_info, host_paths = host_reference(root, admission, source, groups, sequence_ids)
    predictions = {**cached, "current_host_reference": host}
    sample_gate = sampled_parity(host, groups, sequence_ids)
    for name, values in predictions.items():
        need(
            tuple(values.shape) == (468, 20_000, 2) and values.dtype == np.float32,
            f"{name} cache header differs",
        )
        need(
            all(np.isfinite(values[i]).all() for i in range(468)),
            f"nonfinite {name} prediction cache",
        )

    input_paths = {**source_paths, **cached_paths, **host_paths, "host_admission": admission_path}
    input_before = {name: sha(path) for name, path in input_paths.items()}
    # This is the first target-array access, after source, cache, coverage and eight-group gates.
    targets, mask, components, target_groups, target_identity = load_targets(root, source)
    need(list(map(int, target_groups)) == groups, "target rows do not align with F0 source")
    result = grade_predictions(targets, mask, components, groups, predictions)

    expected = {
        "legacy_cached_tl3": "original_tl3",
        "h512_cp5_replacement": "h512_cp5_replacement",
        "h512_swa345_replacement": "h512_swa345_replacement",
    }
    reproduction = {}
    for name, key in expected.items():
        wanted, got = old_grade["endpoint_scores"][key], result["scores"][name]
        delta = {field: got[field] - wanted[field] for field in ("wp", "t0", "t1")}
        need(
            got["selected_rows"] == wanted["selected_rows"]
            and max(map(abs, delta.values())) <= 1e-7,
            f"historical score differs for {name}",
        )
        reproduction[name] = {"expected": wanted, "recomputed": got, "difference": delta}
    input_after = {name: sha(path) for name, path in input_paths.items()}
    need(input_before == input_after, "a frozen input changed during grading")

    from endpoint_contract import physical_scratch

    output = physical_scratch(args.output)
    need(
        not output.exists() and output.parent.is_dir(),
        f"output must be a new file in existing physical scratch: {output}",
    )
    report = {
        "schema": "wnn.tl3-current-host-reference-grade.v1",
        "status": "GRADED_F0_EXPOSED_FITTED_DIAGNOSTIC",
        "exposure": "F0 is exposed and TL3 ancestry is unknown; every score and delta is a fitted diagnostic, not clean CV evidence.",
        "official_metric": "global_wp_clip2_v1",
        "target_identity": target_identity,
        "valid_parquet_identity": source.data_identity,
        "source_binding_sha256": BINDING_SHA,
        "host_admission_sha256": ADMISSION_SHA,
        "old_grade_sha256": NATIVE_GRADE_SHA,
        "endpoint_screen_plan_sha256": SCREEN_PLAN_SHA,
        "host_reference": {
            **host_info,
            "python": host_info["receipt"].get("python"),
            "numpy": host_info["receipt"].get("numpy"),
            "pyarrow": host_info["receipt"].get("pyarrow"),
            "platform": host_info["receipt"].get("platform"),
            "runtime_limitations": admission.get("runtime_limitations"),
        },
        "cached_endpoints": cached_receipts,
        "prelabel_sample_parity": sample_gate,
        "endpoint_scores": result["scores"],
        "historical_score_reproduction": reproduction,
        "paired_deltas": result["paired_deltas"],
        "position_bins": result["position_bins"],
        "paired_component_bootstrap": result["paired_component_bootstrap"],
        "bootstrap": {
            "unit": "source-qualified connected component",
            "n_boot": 2_000,
            "seed": 271_828,
        },
        "input_sha256_before": input_before,
        "input_sha256_after": input_after,
        "selection": "Four frozen endpoints and four fixed paired contrasts only; no endpoint or coefficient search.",
        "grade_script_sha256": sha(Path(__file__).resolve()),
        "wall_seconds": time.monotonic() - started,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"status": report["status"], "output": str(output), "scores": result["scores"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
