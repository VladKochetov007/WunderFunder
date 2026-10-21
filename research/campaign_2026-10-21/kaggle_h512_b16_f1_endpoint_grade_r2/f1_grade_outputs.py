"""Bind accepted H512 checkpoints, the fixed SWA receipt, and predictions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from f1_grade_contract import (
    DERIVED_SHA,
    FIT_NAME,
    ORIGINAL_SHA,
    SWA_NAME,
    accepted_file,
    read_json,
)


def check_worker_outputs(
    plan: dict[str, Any], derived: dict, acceptance: dict, worker: dict, base: Path
) -> tuple[dict[str, np.memmap], dict[str, Path]]:
    """Bind CP5, SWA345, provenance and exports to root-accepted bytes."""
    fit_provenance_sha = acceptance.get("fit_provenance_sha256")
    provenance_path = accepted_file(
        base, f"artifacts/runs/{FIT_NAME}/provenance.json", fit_provenance_sha
    )
    fit_provenance = read_json(provenance_path)
    audit = worker.get("fit_audit", {})
    checkpoint_hashes = acceptance.get("checkpoint_sha256", {})
    cp5_hash = checkpoint_hashes.get("model_ep5.pt")
    accepted_file(base, f"artifacts/runs/{FIT_NAME}/model_ep5.pt", cp5_hash)
    fit_args = fit_provenance.get("args", {})
    if (
        audit.get("checkpoint") != f"artifacts/runs/{FIT_NAME}/model_ep5.pt"
        or audit.get("checkpoint_sha256") != cp5_hash
        or audit.get("input_width") != 211
        or fit_provenance.get("split_manifest_sha256") != DERIVED_SHA
        or fit_provenance.get("seed") != 1004
        or fit_args.get("seed") != 1004
        or fit_args.get("hidden") != 512
        or fit_args.get("layers") != 1
        or fit_args.get("batch") != 16
        or fit_args.get("precision") != "fp32"
        or fit_args.get("loss") != "wmse"
        or fit_args.get("loss_norm") != "chunk"
        or fit_args.get("epochs") != 6
        or fit_args.get("checkpoint_policy") != "fixed"
        or fit_args.get("checkpoint_epoch") != 5
        or fit_args.get("split_manifest_sha256") != DERIVED_SHA
        or "resume_state" not in fit_args
        or fit_args["resume_state"] is not None
        or fit_args.get("init")
        or fit_args.get("raw_only") is not False
        or fit_args.get("research_features") != "none"
        or fit_provenance.get("initial_checkpoint_sha256") is not None
        or fit_provenance.get("cumulative_updates", {}).get("optimizer_completed_updates") != 44760
        or fit_provenance.get("normalization_identity") != audit.get("normalization_identity")
        or audit.get("normalization_invariant_across_epochs") is not True
        or not audit.get("initial_parameter_sha256")
    ):
        raise ValueError("Accepted fit provenance does not match the cold H512 F1 recipe")
    normalization = fit_provenance.get("normalization_identity", {})
    if set(normalization) != {"mean", "std"} or any(
        item.get("shape") != [211] or item.get("dtype") != "torch.float32"
        for item in normalization.values()
    ):
        raise ValueError("Fit normalization identity differs from the 211-input FP32 model")
    for epoch in (3, 4, 5):
        expected = checkpoint_hashes.get(f"model_ep{epoch}.pt")
        if not expected:
            raise ValueError(f"Root acceptance omits CP{epoch} source identity")
        accepted_file(base, f"artifacts/runs/{FIT_NAME}/model_ep{epoch}.pt", expected)
        if fit_provenance.get("checkpoint_sha256", {}).get(f"model_ep{epoch}.pt") != expected:
            raise ValueError(f"Fit provenance checkpoint map differs at CP{epoch}")
    fit_membership = fit_provenance.get("fit_row_groups", {})
    expected_fit = derived["fit"]
    expected_eval = derived["evaluation"]["valid"]
    if (
        set(fit_membership.get("train", [])) != set(expected_fit["train"])
        or set(fit_membership.get("valid", [])) != set(expected_fit["valid"])
        or set(fit_provenance.get("evaluation_row_groups", [])) != set(expected_eval)
        or len(fit_membership.get("train", [])) != len(set(fit_membership.get("train", [])))
        or len(fit_membership.get("valid", [])) != len(set(fit_membership.get("valid", [])))
        or len(fit_provenance.get("evaluation_row_groups", []))
        != len(set(fit_provenance.get("evaluation_row_groups", [])))
    ):
        raise ValueError("Accepted fit membership is incomplete or duplicated")
    component_ids = {
        (split, int(node["row_group"])): int(node["component"])
        for split in ("train", "valid")
        for node in derived["sequences"][split]
    }
    fit_components = {component_ids[("train", group)] for group in fit_membership["train"]} | {
        component_ids[("valid", group)] for group in fit_membership["valid"]
    }
    evaluation_components = {
        component_ids[("valid", group)] for group in fit_provenance["evaluation_row_groups"]
    }
    if fit_components & evaluation_components:
        raise ValueError("Fit membership overlaps an evaluation connected component")

    derivation_rel = f"artifacts/runs/{SWA_NAME}/weight_derivation_receipt.json"
    if acceptance.get("swa_derivation_receipt_path") != derivation_rel:
        raise ValueError("SWA receipt path differs from the fixed derived run")
    derivation_path = accepted_file(
        base,
        acceptance.get("swa_derivation_receipt_path"),
        acceptance.get("swa_derivation_receipt_sha256"),
    )
    derivation = read_json(derivation_path)
    worker_swa = worker.get("swa345", {})
    source_paths = [f"artifacts/runs/{FIT_NAME}/model_ep{epoch}.pt" for epoch in (3, 4, 5)]
    source_hashes = [checkpoint_hashes[f"model_ep{epoch}.pt"] for epoch in (3, 4, 5)]
    inner_derivation = worker_swa.get("derivation", {})
    source_checkpoints = [
        {"epoch": epoch, "path": path, "sha256": digest}
        for epoch, path, digest in zip((3, 4, 5), source_paths, source_hashes, strict=True)
    ]
    output_paths = {
        "model_ep5.pt": f"artifacts/runs/{SWA_NAME}/model_ep5.pt",
        "provenance.json": f"artifacts/runs/{SWA_NAME}/provenance.json",
        "model.pt": f"artifacts/runs/{SWA_NAME}/model.pt",
    }
    output_records = derivation.get("outputs", {})
    if set(output_records) != set(output_paths):
        raise ValueError("Standalone SWA receipt has an incomplete output identity map")
    output_files = {
        name: accepted_file(base, output_paths[name], output_records[name].get("sha256", ""))
        for name in output_paths
    }
    swa_provenance_path = output_files["provenance.json"]
    swa_provenance = read_json(swa_provenance_path)
    expected_input_manifest = {
        "path": "research/campaign_2026-10-21/kaggle_h512_b16_conditional_f1_r2/fold1_profile_view.json",
        "sha256": DERIVED_SHA,
    }
    expected_original_manifest = {
        "path": "research/campaign_2026-09-29/artifacts/splits/fold1.json",
        "sha256": ORIGINAL_SHA,
    }
    if (
        derivation.get("schema") != "wnn.kaggle-h512-swa345-weight-derivation.v1"
        or derivation.get("status") != "DERIVED_UNGRADED"
        or derivation.get("method") != "equal_parameter_mean_epochs_3_4_5"
        or derivation.get("source_run") != FIT_NAME
        or derivation.get("destination_run") != SWA_NAME
        or derivation.get("source_checkpoints") != source_checkpoints
        or derivation.get("input_manifest") != expected_input_manifest
        or derivation.get("original_canonical_manifest") != expected_original_manifest
        or derivation.get("initial_parameter_sha256") != audit.get("initial_parameter_sha256")
        or derivation.get("normalization_identity") != audit.get("normalization_identity")
        or derivation.get("training_run") is not False
        or derivation.get("quality_evaluated") is not False
        or output_records["model_ep5.pt"].get("path") != output_paths["model_ep5.pt"]
        or output_records["provenance.json"].get("path") != output_paths["provenance.json"]
        or output_records["model.pt"].get("path") != output_paths["model.pt"]
        or worker_swa.get("run") != SWA_NAME
        or worker_swa.get("model_ep5_path") != output_paths["model_ep5.pt"]
        or worker_swa.get("model_ep5_sha256") != output_records["model_ep5.pt"].get("sha256")
        or worker_swa.get("provenance_path") != output_paths["provenance.json"]
        or worker_swa.get("provenance_sha256") != output_records["provenance.json"].get("sha256")
        or worker_swa.get("weight_derivation_receipt_path") != derivation_rel
        or worker_swa.get("weight_derivation_receipt_sha256")
        != acceptance.get("swa_derivation_receipt_sha256")
        or {key: derivation.get(key) for key in inner_derivation} != inner_derivation
        or swa_provenance.get("weight_derivation") != inner_derivation
        or swa_provenance.get("checkpoint_sha256", {}).get("model_ep5.pt")
        != output_records["model_ep5.pt"].get("sha256")
        or swa_provenance.get("checkpoint_sha256", {}).get("model.pt")
        != output_records["model.pt"].get("sha256")
        or swa_provenance.get("args") != fit_args
        or swa_provenance.get("split_manifest_sha256") != DERIVED_SHA
    ):
        raise ValueError("Standalone SWA derivation receipt differs from the frozen mean")

    exports = acceptance.get("exports", {})
    worker_exports = {item.get("run_name"): item for item in worker.get("exports", [])}
    predictions: dict[str, np.memmap] = {}
    tracked: dict[str, Path] = {
        "fit_provenance": provenance_path,
        "swa_derivation": derivation_path,
        "swa_provenance": swa_provenance_path,
    }
    for key, run in (("cp5", FIT_NAME), ("swa345", SWA_NAME)):
        binding = exports.get(key)
        record = worker_exports.get(run)
        if not isinstance(binding, dict) or not isinstance(record, dict):
            raise ValueError(f"Root/worker export identity missing: {key}")
        for field in (
            "checkpoint_path",
            "checkpoint_sha256",
            "prediction_path",
            "prediction_sha256",
            "sidecar_path",
            "sidecar_sha256",
        ):
            if binding.get(field) != record.get(field):
                raise ValueError(f"Root and worker export identity differ: {key}/{field}")
        cp_path = accepted_file(base, binding["checkpoint_path"], binding["checkpoint_sha256"])
        pred_path = accepted_file(base, binding["prediction_path"], binding["prediction_sha256"])
        sidecar_path = accepted_file(base, binding["sidecar_path"], binding["sidecar_sha256"])
        metadata = read_json(sidecar_path)
        expected_cp5 = cp5_hash if key == "cp5" else output_records["model_ep5.pt"]["sha256"]
        ordered_rows_sha = hashlib.sha256(
            np.asarray(derived["evaluation"]["valid"], dtype="<i8").tobytes()
        ).hexdigest()
        if (
            binding["checkpoint_sha256"] != expected_cp5
            or binding["checkpoint_path"] != f"artifacts/runs/{run}/model_ep5.pt"
            or binding["prediction_path"] != f"cache/pred_{run}_valid.npy"
            or binding["sidecar_path"] != f"cache/pred_{run}_valid.json"
            or metadata.get("run") != run
            or metadata.get("epoch") != 5
            or metadata.get("checkpoint_sha256") != binding["checkpoint_sha256"]
            or metadata.get("campaign_split_manifest_sha256") != DERIVED_SHA
            or metadata.get("campaign_export_scope") != "evaluation"
            or metadata.get("feature_only") is not True
            or metadata.get("inference_precision") != "fp32"
            or metadata.get("prediction_sha256") != binding["prediction_sha256"]
            or not isinstance(metadata.get("row_groups"), list)
            or metadata["row_groups"] != derived["evaluation"]["valid"]
            or len(set(metadata["row_groups"])) != 457
        ):
            raise ValueError(f"Prediction sidecar differs from accepted F1 feature export: {key}")
        values = np.load(pred_path, mmap_mode="r", allow_pickle=False)
        if values.shape != (457, 20_000, 2) or values.dtype != np.float32:
            raise ValueError(f"Malformed evaluation-only prediction array: {key}")
        if (
            record.get("campaign_split_manifest_sha256") != DERIVED_SHA
            or record.get("campaign_export_scope") != "evaluation"
            or record.get("feature_only") is not True
            or record.get("row_group_count") != 457
            or record.get("ordered_row_groups_sha256") != ordered_rows_sha
            or record.get("prediction_size_bytes") != pred_path.stat().st_size
        ):
            raise ValueError(f"Worker export receipt is not bound to the derived view: {key}")
        predictions[key] = values
        tracked[f"{key}_checkpoint"] = cp_path
        tracked[f"{key}_prediction"] = pred_path
        tracked[f"{key}_sidecar"] = sidecar_path
    return predictions, tracked
