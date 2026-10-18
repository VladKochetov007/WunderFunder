from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Mapping
from pathlib import Path

from f3_checkpoint_common import (
    ARCHIVE_MANIFEST_SHA256,
    FOLD3_SHA256,
    PLAN_SHA256,
    REGISTRY_SHA256,
    REMOTE_ROOT,
    SOURCE_REVISION,
    arm_spec,
    expected_counts,
    mapping,
    read_json,
    require,
    sha256_file,
)
from f3_checkpoint_semantics import validate_pair_contracts


def verify_lineage(
    args: Namespace,
    plan: Mapping[str, object],
    spec: Mapping[str, object],
    contract: Mapping[str, object],
    history: list[dict[str, object]],
    provenance: Mapping[str, object],
    collection: Mapping[str, object],
    load_training_state: Callable[[Path], object],
    verification_helper_sha256: str,
    verification_dependencies: Mapping[str, str],
) -> str | None:
    """Verify cold-start, same-arm continuation and peer-arm stage1 ancestry."""
    parent_sha: str | None = None
    if args.phase == "terminal":
        require(
            args.stage1_snapshot_dir is not None and args.stage1_verification is not None,
            "same-arm stage1 inputs",
        )
        parent_path = args.stage1_snapshot_dir / "training_state.pt"
        parent_state = mapping(load_training_state(parent_path), "stage1 parent state")
        stage1_receipt = read_json(args.stage1_verification)
        stage1_attempt = mapping(stage1_receipt["attempt"], "stage1 attempt receipt")
        expected_stage1 = arm_spec(plan, args.arm, "stage1")
        require(
            stage1_attempt.get("id") == expected_stage1["attempt"]
            and stage1_attempt.get("command") == expected_stage1["command"]
            and (stage1_attempt.get("status"), stage1_attempt.get("exit_status"))
            == ("finished", 0),
            "verified stage1 attempt identity",
        )
        parent_sha = sha256_file(parent_path)
        stage1_artifacts = mapping(stage1_receipt["artifacts"], "stage1 verification artifacts")
        stage1_state_entry = mapping(stage1_artifacts["training_state.pt"], "stage1 state receipt")
        require(
            (
                stage1_receipt.get("schema"),
                stage1_receipt.get("status"),
                stage1_receipt.get("phase"),
                stage1_receipt.get("arm"),
                stage1_receipt.get("run"),
                stage1_receipt.get("next_epoch"),
                stage1_receipt.get("plan_sha256"),
                stage1_receipt.get("fold3_manifest_sha256"),
                stage1_receipt.get("portable_registry_sha256"),
                stage1_receipt.get("source_archive_manifest_sha256"),
                stage1_receipt.get("source_revision"),
                stage1_receipt.get("verification_helper_sha256"),
                stage1_receipt.get("verification_dependencies"),
                stage1_receipt.get("completed_update_counts"),
                stage1_state_entry.get("sha256"),
            )
            == (
                "priority25.f3_checkpoint_verification.v1",
                "passed",
                "stage1",
                args.arm,
                str(spec["stage1_run"]),
                2,
                PLAN_SHA256,
                FOLD3_SHA256,
                REGISTRY_SHA256,
                ARCHIVE_MANIFEST_SHA256,
                SOURCE_REVISION,
                verification_helper_sha256,
                dict(verification_dependencies),
                expected_counts(2),
                parent_sha,
            ),
            "verified stage1 parent identity",
        )
        require(
            (parent_state["next_epoch"], parent_state["contract"]) == (2, contract),
            "stage1 continuation contract",
        )
        require(
            history[:2] == parent_state["progress"]["history"],
            "stage1 history changed during continuation",
        )
        expected_parent = {
            "path": str(
                REMOTE_ROOT / "data/artifacts/runs" / str(spec["stage1_run"]) / "training_state.pt"
            ),
            "sha256": parent_sha,
            "next_epoch": 2,
        }
        require(provenance.get("resumed_from") == expected_parent, "exact stage1 resume provenance")
        require(collection.get("resumed_from") == expected_parent, "collection stage1 lineage")
    else:
        require(provenance.get("resumed_from") is None, "cold stage1 has no resume parent")

    if args.peer_stage1_snapshot_dir is not None:
        peer_state = mapping(
            load_training_state(args.peer_stage1_snapshot_dir / "training_state.pt"),
            "peer stage1 state",
        )
        peer_arm = "deep" if args.arm == "wide" else "wide"
        peer_receipt_path = args.peer_stage1_verification
        require(peer_receipt_path is not None, "peer stage1 verification receipt")
        peer_receipt = read_json(peer_receipt_path)
        expected_peer = arm_spec(plan, peer_arm, "stage1")
        peer_attempt = mapping(peer_receipt["attempt"], "peer stage1 attempt receipt")
        require(
            peer_attempt.get("id") == expected_peer["attempt"]
            and peer_attempt.get("command") == expected_peer["command"]
            and (peer_attempt.get("status"), peer_attempt.get("exit_status")) == ("finished", 0),
            "peer stage1 attempt identity",
        )
        require(
            (
                peer_receipt.get("schema"),
                peer_receipt.get("status"),
                peer_receipt.get("phase"),
                peer_receipt.get("arm"),
                peer_receipt.get("next_epoch"),
                peer_receipt.get("plan_sha256"),
                peer_receipt.get("fold3_manifest_sha256"),
                peer_receipt.get("portable_registry_sha256"),
                peer_receipt.get("source_archive_manifest_sha256"),
                peer_receipt.get("source_revision"),
                peer_receipt.get("verification_helper_sha256"),
                peer_receipt.get("verification_dependencies"),
            )
            == (
                "priority25.f3_checkpoint_verification.v1",
                "passed",
                "stage1",
                peer_arm,
                2,
                PLAN_SHA256,
                FOLD3_SHA256,
                REGISTRY_SHA256,
                ARCHIVE_MANIFEST_SHA256,
                SOURCE_REVISION,
                verification_helper_sha256,
                dict(verification_dependencies),
            ),
            "peer stage1 receipt identity",
        )
        require(
            peer_receipt.get("run") == str(arm_spec(plan, peer_arm, "stage1")["run"])
            and peer_receipt.get("verification_helper_sha256") == verification_helper_sha256,
            "peer stage1 run/helper identity",
        )
        peer_entry = mapping(
            mapping(peer_receipt["artifacts"], "peer artifacts")["training_state.pt"],
            "peer state artifact",
        )
        require(
            peer_entry.get("sha256")
            == sha256_file(args.peer_stage1_snapshot_dir / "training_state.pt"),
            "peer state receipt hash",
        )
        peer_contract = mapping(peer_state["contract"], "peer contract")
        require(peer_state["next_epoch"] == 2, "peer stage1 next_epoch")
        require(
            peer_state["progress"]["cumulative_updates"] == expected_counts(2),
            "peer stage1 counters",
        )
        if args.arm == "wide":
            validate_pair_contracts(contract, peer_contract)
        else:
            validate_pair_contracts(peer_contract, contract)
        require(
            contract["fit_membership"] == peer_contract["fit_membership"], "pair fit membership"
        )
        require(contract["normalization"] == peer_contract["normalization"], "pair normalization")
    elif args.phase == "terminal":
        raise ValueError(
            "terminal verification requires peer stage1 snapshot and verification receipt"
        )
    return parent_sha
