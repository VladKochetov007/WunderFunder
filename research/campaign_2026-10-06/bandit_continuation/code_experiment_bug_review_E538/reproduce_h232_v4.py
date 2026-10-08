from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any  # Primary JSON receipts have heterogeneous nested schemas.


def file_sha256(path: Path) -> str:
    """Hash a small inspected source or receipt."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_gate_arguments(worker_path: Path) -> dict[str, Any]:
    """Recover unrelated validated arguments from the frozen gate source."""
    tree = ast.parse(worker_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_conditional_fit_gate"
    )
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "required_args" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("frozen gate has no required_args assignment")


def reproduce_digest_scope(worker: ModuleType) -> dict[str, Any]:
    """Execute exact handoff with two tiny raw files and four auxiliary files."""
    with tempfile.TemporaryDirectory(prefix="e538-digest-") as temporary:
        root = Path(temporary)
        raw = tuple(root / name for name in ("train.parquet", "valid.parquet"))
        auxiliary = tuple(root / f"auxiliary-{index}" for index in range(4))
        for index, path in enumerate((*raw, *auxiliary)):
            path.write_bytes(f"source-{index}".encode())
        handoff = worker.VerifiedDigestHandoff(raw)
        for path in raw:
            candidate = handoff.hash_file(path)
            expected = file_sha256(path)
            handoff.authorize(candidate, expected_sha256=expected, registry_sha256=expected)
        original_calls: list[str] = []

        def original_digest(path: str, *_identity: int) -> str:
            original_calls.append(Path(path).name)
            return file_sha256(Path(path))

        portable = ModuleType("tiny_portable_oracle")
        portable._content_digest = original_digest
        handoff.install_portable_verifier(portable)
        for path in (*raw, *auxiliary):
            stat = path.stat()
            observed = portable._content_digest(
                str(path),
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
            assert observed == file_sha256(path)
        assert original_calls == [path.name for path in auxiliary]
        assert handoff.metrics() == {
            "verifier_lookups": 6,
            "digest_reuse_hits": 2,
            "digest_reuse_misses": 4,
            "fallback_digest_calls": 4,
        }
        return {
            "aggregate": handoff.metrics(),
            "raw": {path.name: handoff.metrics_for(path) for path in raw},
            "original_verifier_calls": original_calls,
        }


def main() -> None:
    """Replay V4 admission against actual primary profile receipts without fitting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.main_root / (
        "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
    )
    packet = campaign / "h232_conditional_v4/packet"
    receipt_root = campaign / "downloads/h232_conditional_v4/h232_conditional_v4"
    source_root = campaign / "downloads/h232_conditional_v4/h232_source_v4"
    sys.path.insert(0, str(packet))
    worker = importlib.import_module("profile_worker")
    profile = json.loads((receipt_root / "profile_result.json").read_text())
    recorded_gate = json.loads((receipt_root / "profile_gate_result.json").read_text())
    trace = json.loads((receipt_root / "profile_update_trace.json").read_text())
    identity = json.loads((receipt_root / "input_identity.json").read_text())
    sys.path.insert(0, str(source_root / "src"))
    import numpy as np

    from wnn.data import N_FEATURES
    from wnn.features import N_EVENT_FEATURES, event_features

    tiny_raw = np.zeros((1, 3, N_FEATURES), dtype=np.float32)
    tiny_events = event_features(tiny_raw)
    actual_width = N_FEATURES + tiny_events.shape[-1]
    assert N_EVENT_FEATURES == 99
    assert actual_width == 211
    assert all(
        row["model_input_shape"][-1] == actual_width for row in recorded_gate["export_shape_probe"]
    )
    replay_inputs = {
        "trace": trace,
        "timing": recorded_gate["timing"],
        "loader_budget": recorded_gate["loader_budget"],
        "trainer_args": frozen_gate_arguments(packet / "profile_worker.py"),
        "n_in": actual_width,
        "resources": profile["resources_final"],
        "export_probe": recorded_gate["export_shape_probe"],
        "raw_input_identity": recorded_gate["raw_input_identity"],
        "profile_handoff_delta": recorded_gate["profile_digest_handoff_delta"],
        "elapsed_seconds": recorded_gate["lifetime_projection"][
            "worker_elapsed_before_fit_seconds"
        ],
    }
    replay = worker.evaluate_conditional_fit_gate(**replay_inputs)
    assert replay["reasons"] == recorded_gate["reasons"]
    assert replay["reasons"] == [
        "profile_input_width_is_not_full_feature_160",
        "same_worker_profile_digest_handoff_not_observed",
    ]
    result = {
        "status": "confirmed_exact_replay",
        "actual_raw_width": N_FEATURES,
        "actual_event_width": N_EVENT_FEATURES,
        "actual_model_width": actual_width,
        "replayed_gate": replay,
        "tiny_digest_scope_reproduction": reproduce_digest_scope(worker),
        "actual_preprofile_raw_digest_metrics": identity["portable_verifier_raw_digest_metrics"],
        "primary_fit_started": profile["fit_started"],
        "source_sha256": {
            str(path.relative_to(args.main_root)): file_sha256(path)
            for path in (
                packet / "profile_worker.py",
                packet / "source_archive.tar",
                source_root / "scripts/train_rnn.py",
                source_root / "src/wnn/features.py",
                receipt_root / "profile_result.json",
                receipt_root / "profile_gate_result.json",
            )
        },
        "limitations": [
            "trainer args reconstructed from exact gate source; actual receipt reports no argument mismatches",
            "resources_final replays unchanged passing resource checks; original gate-time resources not separately persisted",
            "raw per-file metrics in primary identity are preprofile; postprofile proof uses unchanged identities, aggregate counts and exact scope reproduction",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "reasons": replay["reasons"]}))


if __name__ == "__main__":
    main()
