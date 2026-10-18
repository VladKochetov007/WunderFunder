"""Cross-check frozen F0 receipts before the diagnostic reads predictions."""

from __future__ import annotations

from typing import cast

from wnn.gmetric import METRIC_ID


def _require_equal(actual: object, expected: object, description: str) -> None:
    """Raise when a receipt does not match a sealed source binding."""
    if actual != expected:
        raise ValueError(f"receipt binding mismatch: {description}")


def _validate_receipts(
    manifest: dict[str, object],
    loaded: dict[str, dict[str, object]],
    bundle_sha: str,
    row_count: int,
) -> dict[str, dict[str, object]]:
    """Check the exact F0 extraction, blend, and Q8 alignment ancestry."""
    receipts = cast(dict[str, dict[str, object]], manifest["receipts"])
    extraction = loaded["EXTRACTION"]
    selection = loaded["BLEND_SELECTION"]
    grade = loaded["INDEPENDENT_NATIVE_GRADE"]
    replay = loaded["Q8_FULL_F0"]
    source = cast(dict[str, object], manifest["source"])
    bundle_spec = cast(dict[str, object], manifest["bundle"])
    models = cast(dict[str, dict[str, object]], manifest["models"])

    _require_equal(selection.get("metric_id"), METRIC_ID, "blend-selection metric")
    _require_equal(
        replay.get("official_score", {}).get("metric_id"), METRIC_ID, "full replay metric"
    )
    _require_equal(extraction.get("rows"), row_count, "extraction row count")
    _require_equal(extraction.get("groups"), source["row_groups"], "extraction row groups")
    _require_equal(extraction.get("components"), source["components"], "extraction components")
    extraction_files = cast(dict[str, dict[str, object]], extraction["files"])
    _require_equal(
        extraction_files["scored.npz"].get("sha256"), bundle_sha, "extracted compact bundle"
    )
    _require_equal(
        extraction_files["scored.npz"].get("bytes"), bundle_spec["bytes"], "compact bundle size"
    )
    _require_equal(
        extraction_files["fold0.json"].get("sha256"),
        source["split_manifest_sha256"],
        "extracted fold file",
    )
    source_pins = cast(dict[str, str], extraction["source_pins"])
    _require_equal(source_pins.get("fold"), source["split_manifest_sha256"], "fold split")
    for model_name in ("wide", "deep"):
        _require_equal(
            source_pins.get(model_name),
            models[model_name]["source_prediction_sha256"],
            f"{model_name} source prediction",
        )

    selection_input = cast(dict[str, object], selection["input"])
    _require_equal(selection_input.get("sha256"), bundle_sha, "blend-selection input bundle")
    _require_equal(
        selection_input.get("extraction_receipt_sha256"),
        receipts["EXTRACTION"]["sha256"],
        "blend-selection extraction receipt",
    )
    _require_equal(
        selection_input.get("grade_sha256"),
        bundle_spec["scalar_bindings"]["grade_sha"],
        "blend-selection endpoint grade",
    )
    _require_equal(selection.get("selected_rows"), row_count, "blend-selection rows")
    _require_equal(
        selection_input.get("split_manifest_sha256"),
        source["split_manifest_sha256"],
        "blend-selection fold",
    )
    _require_equal(
        selection_input.get("target_array_sha256"),
        bundle_spec["array_sha256"]["target"],
        "blend-selection targets",
    )
    _require_equal(
        selection_input.get("wide_array_sha256"),
        bundle_spec["array_sha256"]["wide"],
        "blend-selection wide rows",
    )
    _require_equal(
        selection_input.get("deep_array_sha256"),
        bundle_spec["array_sha256"]["deep"],
        "blend-selection deep rows",
    )
    _require_equal(
        selection_input.get("row_group_array_sha256"),
        bundle_spec["array_sha256"]["row_group"],
        "blend-selection row groups",
    )
    _require_equal(
        selection_input.get("component_array_sha256"),
        bundle_spec["array_sha256"]["component"],
        "blend-selection components",
    )

    _require_equal(grade.get("scored_source_npz_sha256"), bundle_sha, "Q8 aligned bundle")
    q8 = models["q8"]
    _require_equal(grade.get("scored_prediction_sha256"), q8["sha256"], "Q8 selected prediction")
    _require_equal(grade.get("selected_rows"), row_count, "Q8 aligned row count")
    _require_equal(grade.get("all_finite"), True, "Q8 selected predictions finite")
    _require_equal(grade.get("dtype"), "float32", "Q8 selected prediction dtype")
    _require_equal(grade.get("shape"), [row_count, 2], "Q8 selected prediction shape")
    _require_equal(grade.get("groups"), source["row_groups"], "Q8 aligned row groups")
    _require_equal(grade.get("components"), source["components"], "Q8 aligned components")
    _require_equal(
        grade.get("full_replay_sha256"), receipts["Q8_FULL_F0"]["sha256"], "Q8 replay receipt"
    )
    _require_equal(replay.get("fold_sha256"), source["split_manifest_sha256"], "full replay fold")
    _require_equal(
        replay.get("panel_sha256"), source["valid_parquet_sha256"], "full replay validation parquet"
    )
    _require_equal(replay.get("reported_model_precision"), "q8vnni", "full replay precision")
    _require_equal(q8.get("paired_bundle_rows"), row_count, "Q8 paired row count")
    _require_equal(q8.get("paired_bundle_sha256"), bundle_sha, "Q8 paired bundle hash")
    return loaded
