"""Cross-bind trainer provenance to the frozen recipe and verified caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any  # JSON metadata is validated at runtime.

from online_relationship_grade_contract import (
    ParentCacheArtifacts,
    _canonical_json_sha256,
    _expect_path_hash,
    _expect_path_value,
    _require_sha,
    _resolve,
    _verify_source_hashes,
    row_refs_sha256,
)
from online_relationship_grade_core import sha256_file
from online_relationship_grade_fit_contract import (
    verify_fit_records,
    verify_parent_normalization,
    verify_row_weight_contract,
)


def verify_candidate_provenance(
    root: Path,
    recipe: dict[str, Any],
    recipe_path: Path,
    recipe_sha: str,
    fold: dict[str, Any],
    cell: dict[str, Any],
    subset: dict[str, Any],
    parent_cache: ParentCacheArtifacts,
    parent_cache_manifest: Path,
    provenance: dict[str, Any],
    provenance_path: Path,
    provenance_sha: str,
    checkpoint_path: Path,
    checkpoint_sha: str,
    prediction_path: Path,
    sidecar_path: Path,
    prediction_sha: str,
    normalizer_sha: str,
    eval_refs: list[dict[str, Any]],
    split_path: Path,
) -> None:
    """Cross-bind trainer provenance to recipe, parent caches and exported bytes."""
    run = str(cell["run"])
    fold_id = int(cell["fold"])
    fit_refs = [
        {"split": split_name, "row_group": int(group)}
        for split_name in ("train", "valid")
        for group in subset["selected_groups"][split_name]
    ]
    subset_path = _resolve(root, fold["subset"]["path"])
    if provenance.get("schema") != "wnn.online-relationships.readout-provenance.v1":
        raise ValueError(f"{run}: unsupported trainer provenance schema")
    scalar_fields = {
        "run": run,
        "fold": fold_id,
        "seed": int(cell["seed"]),
        "lag": int(cell["lag"]),
        "arm": str(cell["arm"]),
        "epoch": 2,
    }
    if any(provenance.get(key) != value for key, value in scalar_fields.items()):
        raise ValueError(f"{run}: trainer provenance cell identity differs from the recipe")
    _expect_path_hash(provenance.get("recipe", {}), "path", "sha256", recipe_path, recipe_sha)
    _expect_path_hash(
        provenance.get("source_manifest", {}),
        "path",
        "sha256",
        split_path,
        fold["split_manifest_sha256"],
    )
    fit_membership = provenance.get("fit_membership", {})
    if (
        fit_membership.get("groups") != fit_refs
        or fit_membership.get("membership_sha256") != row_refs_sha256(fit_refs)
        or fit_membership.get("selected_component_sha256") != subset["selected_component_sha256"]
        or fit_membership.get("subset_manifest_sha256") != fold["subset"]["sha256"]
    ):
        raise ValueError(f"{run}: trainer fit membership differs from the exact component subset")
    _expect_path_value(root, fit_membership, "subset_manifest", subset_path)

    cache_dir = Path(fold["parent_cache"]).resolve()
    cache_fields = provenance.get("caches", {})
    cache_expected = {
        "fit": (
            cache_dir / "fit_sequences.npy",
            cache_dir / "fit_sequences.json",
            parent_cache.identity["fit_cache_sha256"],
            parent_cache.identity["fit_cache_sidecar_sha256"],
            "in_sample_fit",
        ),
        "evaluation": (
            cache_dir / "evaluation_sequences.npy",
            cache_dir / "evaluation_sequences.json",
            parent_cache.identity["evaluation_cache_sha256"],
            parent_cache.identity["evaluation_cache_sidecar_sha256"],
            "feature_only_eval",
        ),
    }
    for name, (data_path, meta_path, payload_sha, meta_sha, scope) in cache_expected.items():
        entry = cache_fields.get(name, {})
        if (
            entry.get("sidecar_sha256") != meta_sha
            or entry.get("payload_sha256") != payload_sha
            or entry.get("scope") != scope
        ):
            raise ValueError(f"{run}: {name} cache identities differ from verified parent caches")
        _expect_path_value(root, entry, "path", data_path)
        _expect_path_value(root, entry, "sidecar", meta_path)
    parent_eval = cache_fields.get("parent_evaluation", {})
    if (
        parent_eval.get("prediction_sha256")
        != parent_cache.identity["evaluation_prediction_sha256"]
        or parent_eval.get("sidecar_sha256")
        != parent_cache.identity["evaluation_prediction_sidecar_sha256"]
        or parent_eval.get("scope") != "feature_only_eval"
    ):
        raise ValueError(f"{run}: parent evaluation prediction identity is inconsistent")
    _expect_path_value(
        root, parent_eval, "prediction_path", cache_dir / "evaluation_predictions.npy"
    )
    _expect_path_value(root, parent_eval, "sidecar", parent_cache_manifest)

    parent = fold["parent"]
    parent_record = provenance.get("parent", {})
    if (
        parent_record.get("run") != parent["run"]
        or parent_record.get("epoch") != parent["epoch"]
        or parent_record.get("checkpoint_sha256") != parent["checkpoint_sha256"]
        or parent_record.get("provenance_sha256") != parent["provenance_sha256"]
        or parent_record.get("mean_identity") != parent["mean_identity"]
        or parent_record.get("std_identity") != parent["std_identity"]
        or parent_record.get("input_width") != parent["input_width"]
        or parent_record.get("fit_prediction_scope") != "in_sample_fit"
        or parent_record.get("endpoint_precision")
        != parent_cache.identity["parent_endpoint_precision"]
        or parent_record.get("endpoint_backend") != parent_cache.identity["parent_endpoint_backend"]
    ):
        raise ValueError(f"{run}: selected parent or in-sample fit scope differs")
    _expect_path_value(root, parent_record, "checkpoint", _resolve(root, parent["checkpoint"]))
    _expect_path_value(root, parent_record, "provenance", _resolve(root, parent["provenance"]))
    propensity = provenance.get("propensity", {})
    expected_propensity = fold["canonical_propensity"]
    if (
        propensity.get("run") != expected_propensity["run"]
        or propensity.get("precision") != "FP32 CPU probabilities; no cached propensity rows reused"
    ):
        raise ValueError(f"{run}: propensity source or precision differs from the recipe")
    if (
        propensity.get("checkpoint_sha256") != expected_propensity["checkpoint_sha256"]
        or propensity.get("provenance_sha256") != expected_propensity["provenance_sha256"]
    ):
        raise ValueError(f"{run}: propensity artifact hashes differ from the recipe")
    propensity_checkpoint = root / "artifacts" / "runs" / expected_propensity["run"] / "model.pt"
    propensity_provenance = (
        root / "artifacts" / "runs" / expected_propensity["run"] / "provenance.json"
    )
    _expect_path_value(
        root,
        propensity,
        "checkpoint",
        propensity_checkpoint,
    )
    _expect_path_value(
        root,
        propensity,
        "provenance",
        propensity_provenance,
    )
    _require_sha(
        propensity_checkpoint, expected_propensity["checkpoint_sha256"], "propensity checkpoint"
    )
    _require_sha(
        propensity_provenance, expected_propensity["provenance_sha256"], "propensity provenance"
    )
    verify_row_weight_contract(run, provenance.get("row_weight_contract", {}))

    normalization = provenance.get("normalization", {})
    normalizer = normalization.get("relationship_normalizer")
    if not isinstance(normalizer, dict) or normalizer.get("lag") != int(cell["lag"]):
        raise ValueError(f"{run}: relationship normalizer is missing or uses a different lag")
    computed_normalizer_sha = _canonical_json_sha256(normalizer)
    if (
        computed_normalizer_sha != normalizer_sha
        or normalization.get("relationship_normalizer_sha256") != normalizer_sha
    ):
        raise ValueError(f"{run}: relationship normalizer content differs from its digest")
    verify_parent_normalization(fold, normalization)

    checkpoint = provenance.get("checkpoint", {})
    _expect_path_hash(checkpoint, "path", "sha256", checkpoint_path, checkpoint_sha)
    if checkpoint.get("epoch") != 2:
        raise ValueError(f"{run}: provenance checkpoint epoch is not two")
    prediction = provenance.get("prediction", {})
    if prediction.get("sha256") != prediction_sha:
        raise ValueError(f"{run}: provenance prediction SHA differs from the output")
    _expect_path_value(root, prediction, "path", prediction_path)
    _expect_path_value(root, prediction, "sidecar", sidecar_path)
    verify_fit_records(
        run,
        recipe,
        cell,
        fit_refs,
        provenance.get("fit", {}),
        normalizer,
        normalization,
    )
    source_hashes = provenance.get("source_sha256", {})
    _verify_source_hashes(root, source_hashes)
    required_sources = {
        "src/wnn/online_relationship.py",
        "src/wnn/online_relationship_numba.py",
        "src/wnn/splits.py",
        "src/wnn/training/online_relationship_fit.py",
        "src/wnn/training/online_relationship.py",
        "src/wnn/training/online_relationship_data.py",
        "src/wnn/training/online_relationship_data_writer.py",
        "src/wnn/training/online_relationship_cache.py",
        "src/wnn/training/online_relationship_cache_validation.py",
        "src/wnn/training/online_relationship_delivery.py",
        "src/wnn/training/online_relationship_exports.py",
        "src/wnn/training/membership.py",
        "src/wnn/conditional_readout.py",
        "src/wnn/features.py",
        "src/wnn/training/checkpoints.py",
    }
    if not required_sources.issubset(source_hashes):
        raise ValueError(f"{run}: provenance omits a required online feature or fit source hash")
    cache_sources = parent_cache.identity["cache_source_sha256"]
    if any(source_hashes.get(name) != digest for name, digest in cache_sources.items()):
        raise ValueError(f"{run}: readout sources differ from the frozen cache builder sources")
    if provenance_sha != sha256_file(provenance_path):
        raise ValueError(f"{run}: provenance changed during verification")
