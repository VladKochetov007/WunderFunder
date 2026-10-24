"""Validate Kaggle transport, source membership and immutable local attestations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wnn.splits import file_sha256

from .age_readout_portable_builder import decode_header
from .age_readout_portable_identity import payload_pin, require_refs
from .age_readout_portable_paths import regular_bundle_file

JsonObject = dict[str, Any]


def attest_bundle(
    repo_root: Path,
    recipe: JsonObject,
    recipe_sha: str,
    fold_id: int,
    bundle_root: Path,
    transport_sha: str,
) -> tuple[JsonObject, JsonObject, str, JsonObject, JsonObject, JsonObject]:
    """Verify transport plus root/local receipts before any cache reconstruction."""
    transport_file = regular_bundle_file(bundle_root / "transport.json", bundle_root)
    if file_sha256(transport_file) != transport_sha:
        raise ValueError("portable transport manifest SHA differs from root admission")
    transport = _json(transport_file)
    source_binding = recipe["source_cache_recipe"]
    if (
        transport.get("schema") != "wnn.age-readout-portable-transport.v1"
        or transport.get("machine_recipe_sha256") != recipe_sha
        or transport.get("source_cache_recipe_sha256") != source_binding["sha256"]
    ):
        raise ValueError("portable transport does not bind both frozen recipes")
    attestations = _attestations(bundle_root, transport)
    cache_receipt = attestations["cache_verification"]
    source_path = Path(source_binding["path"])
    source_path = source_path if source_path.is_absolute() else repo_root / source_path
    if file_sha256(source_path) != source_binding["sha256"]:
        raise ValueError("frozen source-cache recipe SHA changed")
    source_recipe = _json(source_path)
    source_matches = [item for item in source_recipe["folds"] if int(item["fold"]) == fold_id]
    if len(source_matches) != 1:
        raise ValueError("frozen source recipe has no unique requested fold")
    source_fold = source_matches[0]
    _validate_receipts(transport, attestations, recipe_sha, source_binding["sha256"])
    _validate_all_fold_pins(
        bundle_root, transport, cache_receipt, attestations["parent_cache_milestone"]
    )
    return (
        transport,
        source_recipe,
        str(source_binding["sha256"]),
        source_fold,
        cache_receipt,
        attestations["parent_cache_milestone"],
    )


def fold_metadata(
    repo_root: Path,
    source_recipe_sha: str,
    source_fold: JsonObject,
    fold_id: int,
    bundle_fold: JsonObject,
    cache_receipt: JsonObject,
    milestone: JsonObject,
    bundle_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], JsonObject, JsonObject]:
    """Load the pinned sidecars and compare their exact row order to the recipes."""
    metadata = {
        name: _json(transport_path(bundle_root, item))
        for name, item in bundle_fold["sidecars"].items()
    }
    subset_path = Path(source_fold["subset"]["path"])
    subset_path = subset_path if subset_path.is_absolute() else repo_root / subset_path
    if file_sha256(subset_path) != source_fold["subset"]["sha256"]:
        raise ValueError("frozen fit subset metadata SHA changed")
    subset = _json(subset_path)
    selected = subset["selected_groups"]
    subset_members_sha256 = subset.get("selected_members_sha256")
    if (
        not isinstance(subset_members_sha256, str)
        or len(subset_members_sha256) != 64
        or any(character not in "0123456789abcdef" for character in subset_members_sha256)
    ):
        raise ValueError("frozen fit subset has no selected-membership SHA256")
    fit_refs = [
        {"split": split, "row_group": int(group)}
        for split, groups in (("train", selected["train"]), ("valid", selected["valid"]))
        for group in groups
    ]
    eval_refs = [
        {"split": "valid", "row_group": int(group)} for group in source_fold["evaluation_groups"]
    ]
    fit_meta, eval_meta, parent_meta = (
        metadata["fit_sequences"],
        metadata["evaluation_sequences"],
        metadata["evaluation_predictions"],
    )
    for name, value in (("fit", fit_meta), ("evaluation", eval_meta)):
        require_refs(value, fit_refs if name == "fit" else eval_refs, name)
        if value.get("recipe_sha256") != source_recipe_sha or int(value["fold"]) != fold_id:
            raise ValueError(f"portable {name} sidecar recipe or fold identity changed")
        if value.get("split_manifest_sha256") != source_fold["split_manifest_sha256"]:
            raise ValueError(f"portable {name} cache split identity changed")
        if value.get("subset_manifest_sha256") != source_fold["subset"]["sha256"]:
            raise ValueError(f"portable {name} cache subset identity changed")
        if value.get("subset_members_sha256") != subset_members_sha256:
            raise ValueError(f"portable {name} cache subset membership identity changed")
    source_datasets = fit_meta.get("source_datasets", {})
    if source_datasets.get("campaign_split_sha256") != source_fold["split_manifest_sha256"]:
        raise ValueError("portable raw-source metadata differs from the frozen campaign split")
    require_refs(parent_meta, eval_refs, "parent prediction")
    parent = milestone["parent_caches"][str(fold_id)]
    if parent_meta.get("recipe_sha256") != source_recipe_sha or int(parent_meta["fold"]) != fold_id:
        raise ValueError("parent prediction sidecar recipe or fold identity changed")
    if (
        parent_meta.get("scope") != "feature_only_eval"
        or parent_meta.get("sampled_profile") is not False
        or parent_meta.get("prediction_file") != "evaluation_predictions.npy"
        or parent_meta.get("shape") != [len(eval_refs), 20_000, 2]
        or parent_meta.get("dtype") != "<f4"
        or parent_meta.get("endpoint") != {"precision": "strict_fp32", "backend": "cpu"}
    ):
        raise ValueError(
            "parent predictions do not match the full strict-FP32 CPU endpoint contract"
        )
    eval_sidecar = bundle_fold["sidecars"]["evaluation_sequences"]
    parent_source = milestone["parent_caches"][str(fold_id)]["sidecars"]
    if parent_meta.get("evaluation_cache_sidecar_sha256") != eval_sidecar["sha256"]:
        raise ValueError("parent prediction sidecar does not bind this evaluation cache")
    if (
        parent_meta.get("evaluation_cache_sidecar")
        != parent_source["evaluation_sequences.json"]["path"]
    ):
        raise ValueError(
            "parent prediction sidecar references a different logical evaluation cache"
        )
    parent_pin = payload_pin(cache_receipt, fold_id, "evaluation_predictions")
    if parent_meta.get("prediction_sha256") != parent_pin["sha256"]:
        raise ValueError("parent prediction sidecar differs from the local payload pin")
    expected_parent = source_fold["parent"]
    parent_identity = parent_meta.get("parent", {})
    parent_fields = (
        ("run", "run"),
        ("epoch", "epoch"),
        ("checkpoint_sha256", "checkpoint_sha256"),
        ("provenance_sha256", "provenance_sha256"),
        ("mean_identity", "mean_identity"),
        ("std_identity", "std_identity"),
    )
    if any(
        parent_identity.get(left) != expected_parent.get(right) for left, right in parent_fields
    ):
        raise ValueError("parent prediction sidecar ancestry differs from the source recipe")
    manifest_sha = file_sha256(
        transport_path(bundle_root, bundle_fold["sidecars"]["evaluation_predictions"])
    )
    if parent["sidecars"]["evaluation_predictions.json"]["sha256"] != manifest_sha:
        raise ValueError("parent milestone manifest digest differs from transport")
    return fit_refs, eval_refs, parent_meta, fit_meta["source_datasets"]["sources"]


def transport_path(root: Path, item: JsonObject) -> Path:
    """Resolve a manifest-relative input and reject traversal, symlinks or non-files."""
    relative = Path(item["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("portable transport paths must be relative and traversal-free")
    return regular_bundle_file(root / relative, root)


def _attestations(root: Path, transport: JsonObject) -> dict[str, JsonObject]:
    expected_names = {
        "cache_verification",
        "parent_cache_milestone",
        "actual_parent_metadata_check",
    }
    if set(transport.get("attestations", {})) != expected_names:
        raise ValueError("portable transport has an incomplete root-attestation map")
    result: dict[str, JsonObject] = {}
    for name, relative in transport["attestations"].items():
        entry = {"path": relative}
        path = transport_path(root, entry)
        expected_sha = transport.get(f"{name}_sha256")
        if not expected_sha or file_sha256(path) != expected_sha:
            raise ValueError(f"portable {name} attestation SHA changed")
        result[name] = _json(path)
    return result


def _validate_receipts(
    transport: JsonObject,
    attestations: dict[str, JsonObject],
    recipe_sha: str,
    source_recipe_sha: str,
) -> None:
    cache = attestations["cache_verification"]
    milestone = attestations["parent_cache_milestone"]
    metadata = attestations["actual_parent_metadata_check"]
    if (
        cache.get("schema") != "wnn.age-readout-local-cache-verification.v1"
        or cache.get("status") != "PASS_PAYLOAD_HASH_AND_HEADER"
        or cache.get("recipe_sha256") != recipe_sha
        or cache.get("source_cache_recipe", {}).get("sha256") != source_recipe_sha
    ):
        raise ValueError("local cache-verification receipt does not bind both frozen recipes")
    if milestone.get("execution", {}).get("recipe_sha256") != source_recipe_sha:
        raise ValueError("parent-cache milestone does not bind the frozen cache recipe")
    if (
        metadata.get("status") != "PASS_METADATA_AND_PARENT_PREDICTION_CHECKS"
        or metadata.get("recipe_sha256") != recipe_sha
        or metadata.get("cache_verification_sha256") != transport["cache_verification_sha256"]
        or metadata.get("parent_cache_milestone_sha256")
        != transport["parent_cache_milestone_sha256"]
    ):
        raise ValueError("root parent-metadata attestation does not bind this transport")
    checks = metadata.get("checks", {})
    required_true = (
        "recipe_and_source_qualified_fold_membership",
        "six_attested_cache_sidecars",
        "two_parent_prediction_headers_and_finite_values",
        "parent_provenance_and_fit_eval_exclusion",
    )
    required_false = (
        "fit_and_eval_sequence_payloads_mapped_or_read",
        "target_cache_mapped_or_opened",
        "candidate_predictions_mapped_or_opened",
        "score_computed",
    )
    if any(checks.get(key) is not True for key in required_true):
        raise ValueError(
            "root metadata attestation omitted a required ancestry or membership check"
        )
    if any(checks.get(key) is not False for key in required_false):
        raise ValueError("root metadata attestation is not a pre-fit metadata-only check")


def _validate_all_fold_pins(
    root: Path,
    transport: JsonObject,
    receipt: JsonObject,
    milestone: JsonObject,
) -> None:
    folds = transport.get("folds", {})
    if set(folds) != {"0", "1"}:
        raise ValueError("portable transport must bind both frozen folds")
    receipt_files = {(int(item["fold"]), item["kind"]): item for item in receipt["files"]}
    if len(receipt_files) != 6:
        raise ValueError("local cache receipt does not contain exactly six immutable payloads")
    for fold_id in (0, 1):
        fold = folds[str(fold_id)]
        local = milestone["parent_caches"][str(fold_id)]
        cache_kinds = {
            "fit_sequences": ("fit_sequences", "fit_sequences.json"),
            "evaluation_sequences": ("evaluation_sequences", "evaluation_sequences.json"),
            "evaluation_predictions": ("evaluation_predictions", "evaluation_predictions.json"),
        }
        for name, (kind, sidecar_name) in cache_kinds.items():
            pin = receipt_files[(fold_id, kind)]
            sidecar = fold["sidecars"][name]
            manifest_entry = local["sidecars"][sidecar_name]
            sidecar_path = transport_path(root, sidecar)
            if file_sha256(sidecar_path) != pin["sidecar_sha256"]:
                raise ValueError(
                    f"transported {name} sidecar differs from local cache verification"
                )
            if manifest_entry["sha256"] != pin["sidecar_sha256"]:
                raise ValueError(f"milestone does not bind the transported {name} sidecar")
            if name == "evaluation_predictions":
                parent = fold["evaluation_predictions"]
                parent_path = transport_path(root, parent)
                if parent["sha256"] != pin["sha256"] or file_sha256(parent_path) != pin["sha256"]:
                    raise ValueError(
                        "transported parent predictions differ from the local payload pin"
                    )
        for cache_name in ("fit_sequences", "evaluation_sequences"):
            pin = receipt_files[(fold_id, cache_name)]
            header = fold["original_cache_headers"][cache_name]
            if (
                int(header["bytes"]) != int(pin["header"]["header_bytes"])
                or pin["header"]["descr"] != "<f4"
                or pin["header"]["fortran_order"]
            ):
                raise ValueError("transported original cache header differs from local attestation")
            decode_header(header, tuple(pin["header"]["shape"]))
        if not transport.get("actual_parent_metadata_check_sha256"):
            raise ValueError("transport omitted the root parent-metadata-check identity")


def _json(path: Path) -> JsonObject:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value
