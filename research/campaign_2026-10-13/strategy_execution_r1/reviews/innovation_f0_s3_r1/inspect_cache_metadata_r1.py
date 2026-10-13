#!/usr/bin/env python3
"""Read only the small metadata sidecar for the frozen teacher target cache."""

from __future__ import annotations

import hashlib
import json
import sys
from unittest.mock import patch
from pathlib import Path

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
CACHE = MAIN / "cache/clean_t3_f0_h390_t3r5_a9_r2"
SOURCE = Path("/home/vlad/development/wunder/wnn-agent-worktrees/strategy-innovation-fit-20261013")
SPLIT_PATH = MAIN / "research/campaign_2026-09-29/artifacts/splits/fold0.json"


def main() -> None:
    path = CACHE / "metadata.json"
    raw = path.read_bytes()
    metadata = json.loads(raw)
    files = {}
    for name, entry in metadata.get("files", {}).items():
        row_groups = list(map(int, entry.get("row_groups", [])))
        files[name] = {
            "path": entry.get("path"),
            "bytes": entry.get("bytes"),
            "dtype": entry.get("dtype"),
            "shape": entry.get("shape"),
            "declared_sha256": entry.get("sha256"),
            "row_group_count": len(row_groups),
            "row_groups_sha256_i64le": hashlib.sha256(
                b"".join(group.to_bytes(8, "little", signed=True) for group in row_groups)
            ).hexdigest(),
            "has_file_sha256": "sha256" in entry,
        }
    sources = []
    for entry in metadata.get("sources", []):
        sources.append({
            "run": Path(entry.get("checkpoint_path", "")).parent.name,
            "checkpoint_path": entry.get("checkpoint_path"),
            "checkpoint_sha256": entry.get("checkpoint_sha256"),
            "epoch": entry.get("epoch"),
            "seed": entry.get("seed"),
            "family": entry.get("family"),
            "coefficient": entry.get("coefficient"),
            "fit_group_memberships": entry.get("fit_group_memberships"),
            "prediction_sources": {
                split: {key: value.get(key) for key in (
                    "source_split", "scope", "row_group_count", "ordered_row_groups_sha256",
                    "prediction_sha256", "sidecar_sha256", "ready",
                )} for split, value in entry.get("predictions", {}).items()
            },
        })
    fit_groups = {
        name: {
            "count": len(list(map(int, groups))),
            "ordered_group_ids_sha256_i64le": hashlib.sha256(
                b"".join(int(group).to_bytes(8, "little", signed=True) for group in groups)
            ).hexdigest(),
        }
        for name, groups in metadata.get("fit_groups", {}).items()
    }
    # Reconstruct only the frozen membership object consumed by the validator.
    # This intentionally avoids load_campaign_split(), which reopens large
    # target-index inputs before the validator's first metadata rejection.
    split_raw = SPLIT_PATH.read_bytes()
    split_manifest = json.loads(split_raw)
    nodes = tuple(
        (split, int(node["row_group"]), int(node["component"]))
        for split, sequences in split_manifest["sequences"].items()
        for node in sequences
    )
    sys.path.insert(0, str(SOURCE / "src"))
    from wnn.distillation_cache import validate_distillation_cache
    from wnn.splits import CampaignSplit
    import numpy as np

    campaign_split = CampaignSplit(
        SPLIT_PATH,
        hashlib.sha256(split_raw).hexdigest(),
        int(split_manifest["fold"]),
        int(split_manifest["n_folds"]),
        tuple(map(int, split_manifest["fit"]["train"])),
        tuple(map(int, split_manifest["fit"]["valid"])),
        tuple(map(int, split_manifest["evaluation"]["valid"])),
        nodes,
    )
    cache_fit_groups_exact = metadata.get("fit_groups") == {
        "train": list(campaign_split.train_groups),
        "valid": list(campaign_split.valid_groups),
    }
    cache_evaluation_groups_exact = metadata.get("evaluation_valid_groups") == list(
        campaign_split.evaluation_groups
    )
    attempted_array_reads = []

    def forbid_array_read(*args, **kwargs):
        attempted_array_reads.append(str(args[0]) if args else "<no path>")
        raise AssertionError("validator attempted an array read before its first rejection")

    try:
        with patch("wnn.distillation_cache.np.load", side_effect=forbid_array_read):
            validate_distillation_cache(
                CACHE, MAIN, campaign_split, "homogeneous", require_families=False
            )
        first_rejection = None
    except ValueError as error:
        first_rejection = str(error)
    except AssertionError as error:
        first_rejection = f"UNEXPECTED: {error}"
    summary = {
        "metadata_path": str(path),
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "metadata_bytes": len(raw),
        "version": metadata.get("version"),
        "format": metadata.get("format"),
        "cache_profile": metadata.get("cache_profile"),
        "fold": metadata.get("fold"),
        "split_manifest": metadata.get("split_manifest"),
        "split_manifest_sha256": metadata.get("split_manifest_sha256"),
        "fit_groups": fit_groups,
        "evaluation_valid_group_count": len(metadata.get("evaluation_valid_groups", [])),
        "evaluation_valid_groups_sha256_i64le": hashlib.sha256(
            b"".join(int(group).to_bytes(8, "little", signed=True)
                     for group in metadata.get("evaluation_valid_groups", []))
        ).hexdigest(),
        "family_order": metadata.get("family_order"),
        "homogeneous_family": metadata.get("homogeneous_family"),
        "homogeneous_scale": metadata.get("homogeneous_scale"),
        "coefficient_sum": metadata.get("coefficient_sum"),
        "target_contract": metadata.get("target_contract"),
        "student_contract": metadata.get("student_contract"),
        "source_coefficients": metadata.get("source_coefficients"),
        "sources": sources,
        "files": files,
        "source_qualification": metadata.get("source_qualification"),
        "cache_adapter_contract_sha256": metadata.get("adapter_contract_sha256"),
        "consumer_check": {
            "consumer": "frozen R1 src/wnn/distillation_cache.py::validate_distillation_cache",
            "variant": "homogeneous",
            "first_rejection": first_rejection,
            "array_reads_attempted_before_rejection": attempted_array_reads,
            "cache_fit_groups_match_manifest_order": cache_fit_groups_exact,
            "cache_evaluation_groups_match_manifest_order": cache_evaluation_groups_exact,
            "target_index_row_order_crosscheck_reached": False,
            "source_qualification_note": (
                "The validator rejects family_order before opening target indexes or cache arrays."
                if not attempted_array_reads else ""
            ),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
