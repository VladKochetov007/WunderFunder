"""Independently grade the sealed H230 seed-3 export on the frozen fold-0 mask."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / "research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230"
GRADE_FREEZE = HERE / "H230_PRE_SCORE_GRADE_FREEZE_V1_20261008.json"
INCUMBENT_FREEZE = HERE / "H230_PRE_SCORE_PRIMARY_INCUMBENT_FREEZE_V1_20261008.json"
EXPORT_RECEIPT = HERE / "H230_SEED3_EXPORT_TERMINAL_RECEIPT_V1_SOURCEV2_20261008.json"
FIT_RECEIPT = HERE / "H230_SEED3_FIT_TERMINAL_RECEIPT_V1_SOURCEV2_20261008.json"
PROVENANCE = ROOT / "artifacts/runs/bd26_h230_hetero_s3_f0/provenance.json"
TARGETS = ROOT / "cache/valid_targets.npz"
EXPECTED = {
    "grade_freeze": "210cc9ec2e2a7181bd25ff6f5618941630926a2314386cfa8a5475f3256b1b16",
    "incumbent_freeze": "1d62c65412a504b9d8a596eb546ce9a484a22274e64b6574b3e33f6ff79317b1",
    "export_receipt": "bffaac7584fa337dc7cd0a3fb4cb61f7fca900006218fc236cb4a7d6db4616d9",
    "fit_receipt": "ddc8b33c05c84bc518fc716b92efacf150b543d6bff491d99666988b2e61ea22",
    "prediction": "8f0f53f7990739847cb4ecd785b508ef32a59c982a702b5e05cf04bbd1d78e78",
    "prediction_sidecar": "71feb03d21843c396ded01664b6f160198910c86935cf3fb9d77e8bf5717b4f4",
    "targets": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955",
    "split": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
    "metric": "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa",
    "checkpoint": "59718b92befe647dde74335a17dc00d264e668ff8177201a1c636e7e194d72c9",
}
N_ROWS = 1873
SEQ_LEN = 20_000
WARMUP = 99
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20261008


def sha256_file(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a frozen JSON artifact as an object."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_hash(path: Path, expected: str, name: str) -> str:
    """Verify and return a frozen artifact hash."""
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} SHA256 mismatch: {actual}")
    return actual


def execution_memory() -> dict[str, int | None]:
    """Read the enclosing systemd cgroup peak and hard cap."""
    group = Path("/proc/self/cgroup").read_text().split("::")[-1].strip()
    cgroup = Path("/sys/fs/cgroup") / group.lstrip("/")
    peak_path, max_path = cgroup / "memory.peak", cgroup / "memory.max"
    peak = int(peak_path.read_text()) if peak_path.exists() else None
    max_raw = max_path.read_text().strip() if max_path.exists() else "max"
    cap = int(max_raw) if max_raw.isdigit() else None
    return {
        "cgroup_memory_peak_bytes": peak,
        "cgroup_memory_max_bytes": cap,
    }


def open_prediction(
    relative: str,
    expected_sha: str,
    expected_run: str,
    split_sha: str,
) -> tuple[np.memmap, dict[str, Any], str, str]:
    """Validate a full-valid prediction and its row-order sidecar."""
    prediction_path = ROOT / relative
    sidecar_path = prediction_path.with_suffix(".json")
    prediction_sha = require_hash(prediction_path, expected_sha, f"{expected_run} prediction")
    sidecar_sha = sha256_file(sidecar_path)
    sidecar = read_json(sidecar_path)
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if prediction.shape != (N_ROWS, SEQ_LEN, 2) or prediction.dtype != np.float32:
        raise ValueError(
            f"unexpected prediction ABI for {expected_run}: {prediction.shape}/{prediction.dtype}"
        )
    if sidecar.get("prediction_sha256") != prediction_sha:
        raise ValueError(f"prediction/sidecar hash mismatch for {expected_run}")
    if sidecar.get("run") != expected_run or sidecar.get("split") != "valid":
        raise ValueError(f"unexpected run/split in {sidecar_path}")
    if sidecar.get("split_manifest_sha256") not in (None, split_sha):
        raise ValueError(f"split manifest mismatch in {sidecar_path}")
    if sidecar.get("row_groups") != list(range(N_ROWS)):
        raise ValueError(f"prediction row order differs from canonical valid order: {expected_run}")
    for start in range(0, N_ROWS, 32):
        if not np.isfinite(prediction[start : start + 32]).all():
            raise ValueError(f"nonfinite full-valid prediction values: {expected_run}")
    return prediction, sidecar, prediction_sha, sidecar_sha


def metric_score(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> dict[str, float | int | str]:
    """Stream the frozen official metric one sequence at a time."""
    from wnn.gmetric import GlobalAccumulator

    accumulator = GlobalAccumulator()
    for i in range(len(target)):
        accumulator.add(target[i], prediction[i], mask[i])
    return accumulator.result()


def compare(
    label: str,
    target: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    component_ids: np.ndarray,
) -> dict[str, Any]:
    """Compute official point scores and the frozen paired component bootstrap."""
    from wnn.evaluation import paired_component_bootstrap, paired_component_moments

    candidate_score = metric_score(target, candidate, mask)
    reference_score = metric_score(target, reference, mask)
    candidate_moments, reference_moments = paired_component_moments(
        target, candidate, reference, mask
    )
    paired = paired_component_bootstrap(
        candidate_moments,
        reference_moments,
        component_ids,
        n_boot=BOOTSTRAPS,
        seed=BOOTSTRAP_SEED,
        centered=True,
    )
    if abs(float(paired["candidate_wp"]) - float(candidate_score["wp"])) > 1e-9:
        raise ValueError(f"component moments disagree with official candidate score for {label}")
    if abs(float(paired["reference_wp"]) - float(reference_score["wp"])) > 1e-9:
        raise ValueError(f"component moments disagree with official reference score for {label}")
    return {
        "reference": label,
        "candidate": candidate_score,
        "reference_score": reference_score,
        "delta_candidate_minus_reference": paired,
    }


def main() -> int:
    """Verify frozen inputs, reconstruct the literal mask, and grade all named endpoints."""
    started = time.monotonic()
    hashes = {
        "grade_freeze": require_hash(GRADE_FREEZE, EXPECTED["grade_freeze"], "grade freeze"),
        "incumbent_freeze": require_hash(
            INCUMBENT_FREEZE, EXPECTED["incumbent_freeze"], "incumbent freeze"
        ),
        "export_receipt": require_hash(
            EXPORT_RECEIPT, EXPECTED["export_receipt"], "export receipt"
        ),
        "fit_receipt": require_hash(FIT_RECEIPT, EXPECTED["fit_receipt"], "fit receipt"),
        "targets": require_hash(TARGETS, EXPECTED["targets"], "valid targets"),
        "split": require_hash(
            ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json",
            EXPECTED["split"],
            "split manifest",
        ),
        "metric": require_hash(
            ROOT / "src/wnn/gmetric.py", EXPECTED["metric"], "official metric source"
        ),
    }
    grade = read_json(GRADE_FREEZE)
    incumbent = read_json(INCUMBENT_FREEZE)
    export = read_json(EXPORT_RECEIPT)
    fit = read_json(FIT_RECEIPT)
    for name, binding in grade["source_bindings"].items():
        if "sha256" in binding:
            hashes[f"frozen_{name}"] = require_hash(
                ROOT / binding["path"], binding["sha256"], f"frozen {name}"
            )
    export_binding = grade["source_bindings"]["export_command_freeze"]
    if (
        export["frozen_command"]["freeze_path"] != export_binding["path"]
        or export["frozen_command"]["freeze_sha256"] != export_binding["sha256"]
    ):
        raise ValueError("terminal export receipt does not bind the frozen command")
    fit_resource_path = ROOT / fit["recipe_binding"]["frozen_resource_binding"]
    if fit["recipe_binding"]["frozen_resource_binding_sha256"] != sha256_file(fit_resource_path):
        raise ValueError("fit terminal receipt resource binding hash mismatch")
    hashes["fit_resource_binding"] = fit["recipe_binding"]["frozen_resource_binding_sha256"]
    export_resource_path = ROOT / export["attempt"]["contract_path"]
    if export["attempt"]["contract_sha256"] != sha256_file(export_resource_path):
        raise ValueError("export terminal receipt resource binding hash mismatch")
    hashes["export_resource_binding"] = export["attempt"]["contract_sha256"]
    source_manifest = fit["source_freeze"]["portable_manifest_path"]
    hashes["candidate_source_manifest"] = require_hash(
        ROOT / source_manifest,
        fit["source_freeze"]["portable_manifest_sha256"],
        "candidate portable source manifest",
    )
    commands_path = ROOT / grade["source_bindings"]["candidate_control_commands"]["path"]
    commands = read_json(commands_path)
    if (
        commands["candidate"]["run_name"] != "bd26_h230_hetero_s3_f0"
        or commands["candidate"]["family_loss_weight"] != 0.1
        or commands["control"]["family_loss_weight"] != 0.0
    ):
        raise ValueError("frozen candidate/control command pair differs from the grade contract")
    prediction_rel = export["prediction"]["path"]
    if prediction_rel != incumbent["candidate"]["prediction"]:
        raise ValueError("terminal prediction path differs from the incumbent freeze")
    prediction, candidate_sidecar, pred_sha, pred_sidecar_sha = open_prediction(
        prediction_rel,
        EXPECTED["prediction"],
        "bd26_h230_hetero_s3_f0",
        EXPECTED["split"],
    )
    if pred_sidecar_sha != EXPECTED["prediction_sidecar"]:
        raise ValueError("H230 prediction sidecar hash differs from terminal export receipt")
    if (ROOT / prediction_rel).stat().st_size != export["prediction"]["bytes"]:
        raise ValueError("H230 prediction file size differs from terminal export receipt")
    if not (
        candidate_sidecar.get("checkpoint_sha256") == EXPECTED["checkpoint"]
        and candidate_sidecar.get("epoch") == 1
        and candidate_sidecar.get("inference_precision") == "fp16"
        and hashlib.sha256(
            json.dumps(candidate_sidecar["row_groups"], separators=(",", ":")).encode()
        ).hexdigest()
        == export["prediction"]["row_group_order_sha256"]
    ):
        raise ValueError("H230 sidecar fields differ from the sealed export receipt")
    if export["attempt"]["exit_status"] != 0 or not export["attempt"]["worker_exit_success"]:
        raise ValueError("H230 export terminal receipt is not successful")
    if (
        export["claims"]["metric_value"] is not None
        or export["evaluation_freezes"]["score_computed_by_actor"]
    ):
        raise ValueError("actor receipt indicates scoring was already performed")
    if export["predecessor_fit"]["checkpoint_sha256"] != EXPECTED["checkpoint"]:
        raise ValueError("export checkpoint does not match frozen candidate checkpoint")

    provenance = read_json(PROVENANCE)
    if (
        sha256_file(ROOT / "artifacts/runs/bd26_h230_hetero_s3_f0/model_ep1.pt")
        != EXPECTED["checkpoint"]
    ):
        raise ValueError("candidate checkpoint bytes differ from terminal fit receipt")
    if provenance.get("checkpoint_sha256", {}).get("model_ep1.pt") != EXPECTED["checkpoint"]:
        raise ValueError("embedded candidate provenance checkpoint hash mismatch")
    provenance_artifact = fit["artifacts"]["provenance.json"]
    hashes["checkpoint_provenance"] = require_hash(
        PROVENANCE, provenance_artifact["sha256"], "candidate provenance"
    )
    if provenance.get("split_manifest_sha256") != EXPECTED["split"]:
        raise ValueError("candidate provenance split hash mismatch")
    args = provenance["args"]
    if not (
        provenance.get("seed") == 3
        and args.get("seed") == 3
        and args.get("checkpoint_epoch") == 1
        and args.get("valid_fold") == 0
        and args.get("train_source") == "train+valid"
        and args.get("distill_family_weight") == 0.1
        and args.get("distill_family_head") is True
        and args.get("distill_family_target_mode") == "raw"
        and args.get("init") == "c20261003_combo_s3_f0@ep3"
    ):
        raise ValueError("candidate provenance does not match the frozen H230 treatment")
    if (
        fit["attempt"]["exit_status"] != 0
        or fit["recipe_binding"]["selected_epoch_zero_based"] != 1
    ):
        raise ValueError("candidate fit is not a successful fixed epoch-1 fit")
    if provenance["initial_checkpoint_sha256"] != fit["ancestry"]["initializer_checkpoint_sha256"]:
        raise ValueError("candidate initializer provenance differs from the fit receipt")
    hashes["candidate_initializer"] = require_hash(
        ROOT / "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt",
        fit["ancestry"]["initializer_checkpoint_sha256"],
        "candidate initializer checkpoint",
    )
    cache_metadata = ROOT / args["distill_cache_root"] / "metadata.json"
    hashes["distillation_cache_metadata"] = require_hash(
        cache_metadata,
        fit["ancestry"]["distillation_cache_metadata_sha256"],
        "candidate distillation cache metadata",
    )

    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    from wnn.splits import load_campaign_split

    split = load_campaign_split(split_path, ROOT)
    eval_groups = np.asarray(sorted(split.evaluation_groups), dtype=np.int64)
    if len(eval_groups) != grade["exposure_and_membership"]["official_evaluation_groups"]:
        raise ValueError("reconstructed evaluation-group count differs from frozen grade")
    fit_groups = provenance["fit_row_groups"]
    split.assert_fit_groups(fit_groups["train"], fit_groups["valid"])
    if set(provenance["evaluation_row_groups"]) != set(split.evaluation_groups):
        raise ValueError("candidate provenance evaluation groups differ from frozen split")
    if set(fit_groups["train"]) != set(split.train_groups) or set(fit_groups["valid"]) != set(
        split.valid_groups
    ):
        raise ValueError("candidate fit row groups differ from frozen split membership")

    with np.load(TARGETS, allow_pickle=False) as cache:
        targets = cache["y"]
        is_scored = cache["mask"]
        row_groups = cache["row_group"]
    if targets.shape != (N_ROWS, SEQ_LEN, 2) or targets.dtype != np.float32:
        raise ValueError("valid target array has an unexpected shape or dtype")
    if is_scored.shape != (N_ROWS, SEQ_LEN) or row_groups.tolist() != list(range(N_ROWS)):
        raise ValueError("target mask/index does not match canonical valid row order")
    literal = incumbent["evaluation"]["literal_evaluation_mask"]
    eval_flag = np.zeros(N_ROWS, dtype=bool)
    eval_flag[eval_groups] = True
    if literal["row_group_mask_1_is_eval_0_is_not_eval"] != eval_flag.astype(np.uint8).tolist():
        raise ValueError("reconstructed row-group mask differs from literal frozen mask")
    mask_sha = hashlib.sha256(eval_flag.astype(np.uint8).tobytes()).hexdigest()
    if mask_sha != literal["mask_sha256_lsb0_bytes"]:
        raise ValueError("literal evaluation mask SHA mismatch")
    if sorted(eval_groups.tolist()) != literal["row_group_ids"]:
        raise ValueError("literal frozen evaluation group list differs from split")
    metric_mask = is_scored[eval_groups] & (np.arange(SEQ_LEN)[None, :] >= WARMUP)
    target_eval = targets[eval_groups]
    candidate_eval = prediction[eval_groups]
    if not np.isfinite(target_eval[metric_mask]).all():
        raise ValueError("nonfinite selected official labels")
    selected_rows = int(metric_mask.sum())
    if selected_rows == 0:
        raise ValueError("empty official evaluation mask")
    component_lookup = {
        (split_name, group): component for split_name, group, component in split.components
    }
    components = np.asarray([component_lookup[("valid", int(group))] for group in eval_groups])
    if len(np.unique(components)) != literal["component_count"]:
        raise ValueError("connected-component count differs from frozen grade")

    from wnn.gmetric import global_wp

    sanity_n = min(2, len(eval_groups))
    sanity = global_wp(target_eval[:sanity_n], candidate_eval[:sanity_n], metric_mask[:sanity_n])
    from wnn.gmetric import GlobalAccumulator

    sanity_acc = GlobalAccumulator()
    for i in range(sanity_n):
        sanity_acc.add(target_eval[i], candidate_eval[i], metric_mask[i])
    sanity_difference = abs(float(sanity["wp"]) - float(sanity_acc.result()["wp"]))
    if sanity_difference > 1e-12:
        raise ValueError("streaming metric disagrees with official global_wp on sanity subset")

    references: list[tuple[str, str, str]] = []
    warm_path = ROOT / "cache/pred_c20261003_combo_s3_f0_valid.npy"
    warm_side = read_json(warm_path.with_suffix(".json"))
    references.append(
        (
            "same-seed warm parent c20261003_combo_s3_f0@ep3",
            str(warm_path.relative_to(ROOT)),
            warm_side["prediction_sha256"],
        )
    )
    c1_s3_path = ROOT / "cache/pred_bd26_c1v2_s3_f0_valid.npy"
    c1_s3_side = read_json(c1_s3_path.with_suffix(".json"))
    references.append(
        (
            "C1 seed-3 candidate bd26_c1v2_s3_f0@ep1",
            str(c1_s3_path.relative_to(ROOT)),
            c1_s3_side["prediction_sha256"],
        )
    )
    c1_s4_path = ROOT / "cache/pred_bd26_c1v2_s4_f0_valid.npy"
    c1_s4_side = read_json(c1_s4_path.with_suffix(".json"))
    references.append(
        ("C1 seed-pair mean (equal weights; supplemental local endpoint)", "__C1_PAIR__", "")
    )

    # Validate and prepare the exact primary-incumbent FP32 source reconstruction.
    incumbent_source = incumbent["primary_incumbent"]["cached_fp32_source_reference"]
    incumbent_files: dict[str, np.memmap] = {}
    for metadata in incumbent_source["components"].values():
        path = metadata["prediction_path"]
        arr, sidecar, digest, _ = open_prediction(
            path,
            metadata["prediction_sha256"],
            metadata["run"],
            EXPECTED["split"],
        )
        if (ROOT / path).stat().st_size != metadata["prediction_bytes"] or sidecar.get(
            "epoch"
        ) != metadata["epoch"]:
            raise ValueError(f"incumbent component metadata mismatch: {path}")
        incumbent_files[path] = arr
    zip_path = ROOT / incumbent["primary_incumbent"]["zip"]
    weights_path = ROOT / incumbent["primary_incumbent"]["weights_npz"]
    members_path = ROOT / incumbent["primary_incumbent"]["members_json"]
    primary = incumbent["primary_incumbent"]
    hashes["incumbent_zip"] = require_hash(zip_path, primary["zip_sha256"], "incumbent ZIP")
    hashes["incumbent_weights"] = require_hash(
        weights_path, primary["weights_sha256"], "incumbent weights"
    )
    hashes["incumbent_members"] = require_hash(
        members_path, primary["members_sha256"], "incumbent members"
    )

    candidate_score = metric_score(target_eval, candidate_eval, metric_mask)
    comparisons: list[dict[str, Any]] = []
    reference_hashes: dict[str, dict[str, str]] = {}
    for label, relative, expected_sha in references:
        if relative == "__C1_PAIR__":
            s3, _, s3_sha, s3_side_sha = open_prediction(
                str(c1_s3_path.relative_to(ROOT)),
                c1_s3_side["prediction_sha256"],
                "bd26_c1v2_s3_f0",
                EXPECTED["split"],
            )
            s4, _, s4_sha, s4_side_sha = open_prediction(
                str(c1_s4_path.relative_to(ROOT)),
                c1_s4_side["prediction_sha256"],
                "bd26_c1v2_s4_f0",
                EXPECTED["split"],
            )
            ref = s3[eval_groups].copy()
            ref *= 0.5
            ref += s4[eval_groups] * np.float32(0.5)
            reference_hashes[label] = {
                "seed3_prediction_sha256": s3_sha,
                "seed3_sidecar_sha256": s3_side_sha,
                "seed4_prediction_sha256": s4_sha,
                "seed4_sidecar_sha256": s4_side_sha,
            }
            del s3, s4
        elif label.startswith("actual incumbent"):
            raise AssertionError("incumbent handled below")
        else:
            ref_array, sidecar, digest, sidecar_digest = open_prediction(
                relative, expected_sha, sidecar_expected_run(label), EXPECTED["split"]
            )
            ref = ref_array[eval_groups]
            reference_hashes[label] = {
                "prediction_sha256": digest,
                "sidecar_sha256": sidecar_digest,
            }
        comparisons.append(
            compare(label, target_eval, candidate_eval, ref, metric_mask, components)
        )
        del ref

    parent_meta = next(
        item
        for item in incumbent_source["components"].values()
        if item["run"] == "c20261004_combo_s4_final"
    )
    head_meta = next(
        item
        for item in incumbent_source["components"].values()
        if item["run"] == "c20261004_combo_head_s4_final"
    )
    parent = incumbent_files[parent_meta["prediction_path"]]
    head = incumbent_files[head_meta["prediction_path"]]
    incumbent_ref = parent[eval_groups].copy()
    incumbent_ref *= np.float32(0.5)
    incumbent_ref += head[eval_groups] * np.float32(0.5)
    incumbent_label = "primary incumbent FP32 source reconstruction: 0.5 parent@ep3 + 0.5 head@ep0"
    comparisons.append(
        compare(
            incumbent_label, target_eval, candidate_eval, incumbent_ref, metric_mask, components
        )
    )
    reference_hashes[incumbent_label] = {
        "parent_prediction_sha256": parent_meta["prediction_sha256"],
        "head_prediction_sha256": head_meta["prediction_sha256"],
        "incumbent_zip_sha256": hashes["incumbent_zip"],
    }

    memory = execution_memory()
    if (
        memory["cgroup_memory_peak_bytes"] is not None
        and memory["cgroup_memory_max_bytes"] is not None
        and memory["cgroup_memory_peak_bytes"] > memory["cgroup_memory_max_bytes"]
    ):
        raise ValueError("grading run exceeded its memory cgroup cap")
    report = {
        "schema": "h230-seed3-independent-masked-grade-v1",
        "status": "PASS_DESCRIPTIVE_SEED3_GRADE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Frozen exposed fold-0 development data; descriptive seed-3 candidate grade, no family-loss causal estimate, no public score forecast.",
        "review_script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "freeze_sha256": hashes,
        "candidate": {
            "run": "bd26_h230_hetero_s3_f0",
            "epoch_zero_based": 1,
            "inference_precision": candidate_sidecar["inference_precision"],
            "storage_dtype": str(prediction.dtype),
            "prediction_sha256": pred_sha,
            "prediction_sidecar_sha256": pred_sidecar_sha,
            "checkpoint_sha256": EXPECTED["checkpoint"],
            "checkpoint_provenance_sha256": sha256_file(PROVENANCE),
        },
        "mask": {
            "metric_id": "global_wp_clip2_v1",
            "valid_row_groups": N_ROWS,
            "evaluation_row_groups": len(eval_groups),
            "components": int(len(np.unique(components))),
            "scored_rows": selected_rows,
            "warmup_rows_excluded_per_sequence": WARMUP,
            "row_group_mask_sha256_lsb0_bytes": mask_sha,
            "fit_membership": "reconstructed with frozen split; provenance train/valid memberships match and are component-disjoint from the 468 evaluated valid groups",
        },
        "candidate_standalone_official_metric": candidate_score,
        "comparisons": comparisons,
        "reference_artifacts": reference_hashes,
        "uncertainty": {
            "method": "paired bootstrap of connected components with replacement, recomputing pooled global_wp_clip2_v1 delta per draw",
            "replicates": BOOTSTRAPS,
            "seed": BOOTSTRAP_SEED,
            "components": int(len(np.unique(components))),
            "interpretation": "Conditional on these fixed checkpoints and previously exposed fold-0 groups; excludes training variability and public-transfer uncertainty.",
        },
        "serving_precision_limit": {
            "incumbent_zip_path": str(zip_path.relative_to(ROOT)),
            "incumbent_zip_sha256": hashes["incumbent_zip"],
            "scored_reference": "frozen float32 source-level reconstruction, 0.5*parent@ep3 + 0.5*head@ep0",
            "limit": incumbent_source["construction"],
            "zip_inference_replayed_on_this_mask": False,
        },
        "interpretation": {
            "family_head_effect": "not estimable yet: matched family-head-off seed-3 prediction is not in this grade; the candidate-only result is screening evidence",
            "public_transfer": "none; do not add any local delta to the user-reported public best 0.6660",
            "C1_and_parent_comparisons": "supplemental exposed-data endpoint comparisons; not the frozen primary treatment contrast",
        },
        "sanity": {
            "official_global_wp_vs_streaming_accumulator_abs_difference_first_two_groups": sanity_difference,
            "elapsed_seconds": time.monotonic() - started,
            **memory,
            "resource_limits": {
                "cpu_quota": "100% (one CPU)",
                "worker_threads": 1,
                "memory_max_bytes_requested": 1_500_000_000,
                "gpu_use": False,
            },
        },
    }
    output = HERE / "H230_SEED3_INDEPENDENT_GRADE_V1.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "candidate_wp": candidate_score["wp"],
                "comparisons": [
                    {
                        "reference": item["reference"],
                        "reference_wp": item["reference_score"]["wp"],
                        "delta": item["delta_candidate_minus_reference"]["mean_diff"],
                        "ci95": item["delta_candidate_minus_reference"]["ci95"],
                    }
                    for item in comparisons
                ],
                "elapsed_seconds": report["sanity"]["elapsed_seconds"],
                "memory": {
                    "cgroup_memory_peak_bytes": report["sanity"]["cgroup_memory_peak_bytes"],
                    "cgroup_memory_max_bytes": report["sanity"]["cgroup_memory_max_bytes"],
                },
            },
            indent=2,
        )
    )
    return 0


def sidecar_expected_run(label: str) -> str:
    """Map supplemental endpoint labels to their frozen run IDs."""
    if label.startswith("same-seed warm parent"):
        return "c20261003_combo_s3_f0"
    if label.startswith("C1 seed-3 candidate"):
        return "bd26_c1v2_s3_f0"
    raise ValueError(f"unknown supplemental endpoint label: {label}")


if __name__ == "__main__":
    sys.exit(main())
