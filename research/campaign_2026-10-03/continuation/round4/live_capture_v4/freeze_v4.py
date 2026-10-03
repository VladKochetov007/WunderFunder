"""Freeze only minimal v4 evidence and preservation hashes; no approval."""

from __future__ import annotations

import difflib
import json
from pathlib import Path

from live_capture import HERE, ROOT, identities, sha, verify_generated
from prepare_runtime import main as prepare_lineage


def main() -> None:
    """Bind current evidence and reject changed protected earlier versions."""
    before = json.loads((HERE / "LINEAGE.json").read_text())
    prepare_lineage()
    lineage = json.loads((HERE / "LINEAGE.json").read_text())
    if before["preserved"] != lineage["preserved"]:
        raise ValueError("earlier-version bytes changed during v4 build")
    review = HERE.parent / "capture_review_v3"
    supplement = json.loads((review / "SUPPLEMENT_HASHES.json").read_text())
    for name, expected in supplement["sha256"].items():
        if sha(review / name) != expected:
            raise ValueError("critic supplement binding mismatch: " + name)
    verify_generated()
    previous = HERE.parent / "live_capture_v3"
    identical = (
        "runner.py",
        "baselines.py",
        "fit_panel.py",
        "fit_panel.json",
        "source_binding.json",
    )
    for name in identical:
        if sha(HERE / name) != sha(previous / name):
            raise ValueError("unexpected non-RNG candidate change: " + name)
    (HERE / "v3_to_v4.patch").write_text(
        "".join(
            difflib.unified_diff(
                (previous / "live_capture.py").read_text().splitlines(True),
                (HERE / "live_capture.py").read_text().splitlines(True),
                fromfile="live_capture_v3/live_capture.py",
                tofile="live_capture_v4/live_capture.py",
            )
        )
    )
    result = json.loads((HERE / "cpu_smoke/results.json").read_text())
    deltas = {
        key: result[key]
        for key in (
            "normal_max_delta",
            "repeat_max_delta",
            "skip_max_delta",
            "uninstrumented_default_max_delta",
        )
    }
    if any(deltas.values()):
        raise ValueError("CPU pristine replay differs")
    schema = json.loads((HERE / "export_schema.json").read_text())
    schema.update(
        {
            "version": 4,
            "Torch_RNG_roundtrip": "private installed CPU/CUDA engine exact metadata/bytes get_state immediately after set_state; reject before globals; no draws/normalization/masks",
            "lineage": "v1/v2/v3 byte preserved; only padding-normalization blocker fixed; no historical full-fit recovery",
        }
    )
    (HERE / "export_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    state = {
        "phase": "frozen_v4_pending_independent_review",
        "requested_model": "gpt6.1-sol medium",
        "CPU_targeted_tests": 30,
        "prior_tests_not_claimed_as_v4": True,
        "CPU_deltas": deltas,
        "CPU_normal_skip_floors": [
            result["independent_floor"][key] for key in ("normal_max_floor", "skip_max_floor")
        ],
        "smoke_seconds": result["seconds"],
        "payload_tensor_bytes": result["payload_tensor_bytes"],
        "component_tensor_bytes": result["component_tensor_bytes"],
        "roundtrip_validation_no_draws": True,
        "padding_rejected_before_globals": [20, 28, 36, 44, 5048],
        "critic_final_verdict_read": lineage["review_v3_verdict"],
        "critic_supplement_hashes_verified": True,
        "v1_v2_v3_all_file_bytes_preserved": True,
        "runner_baselines_panel_source_bindings_identical_to_v3": True,
        "actual_trainer_sha256": identities()["scripts/train_rnn.py"],
        "CUDA_RNG_engine_roundtrip_measured": False,
        "GPU_launched": False,
        "GPU_floor_measured": False,
        "receipt_written": False,
        "self_approved": False,
        "controller_patch_applied": False,
        "live_source_edits": False,
        "queue_changes": False,
        "today_full_fit_states_recovered": False,
        "missing_historical_state_bytes": "unknown/irrecoverable",
        "real_prestate_estimate_bytes": 600000000,
        "inspected_actual_fit_rows": [],
        "pending": [
            "independent final-v4 review",
            "controller v4 runtime binding/exclusive admission",
            "separately authorized unrun CUDA floor/default/skip gates",
        ],
        "last_failure": None,
    }
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    names = (
        "runner.py",
        "live_capture.py",
        "fit_panel.py",
        "fit_panel.json",
        "PREREG.txt",
        "source_binding.json",
        "trainer.patch",
        "baselines.py",
        "generated_hashes.json",
    )
    request = {
        "status": "pending_final_v4_independent_review",
        "approval_or_receipt": False,
        "candidate_sha256": {name: sha(HERE / name) for name in names},
        "source": identities(),
        "runtime_manifest_sha256": sha(HERE / "runtime_manifest.json"),
        "controller_patch_sha256": sha(HERE / "controller_payload.patch"),
        "review_scope": "only private Torch engine exact set/get roundtrip; unchanged CPU original/default/skip parity; no CUDA evidence",
    }
    (HERE / "review_request.json").write_text(json.dumps(request, indent=2) + "\n")
    (HERE / "results.json").write_text(
        json.dumps({"state": state, "CPU_smoke": result, "lineage": lineage}, indent=2) + "\n"
    )
    hashes = {
        str(path.relative_to(HERE)): sha(path)
        for path in HERE.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_hashes.json", "freeze.log", "protected_hashes.sha256"}
        and not any(part.startswith("test_tmp") for part in path.relative_to(HERE).parts)
    }
    (HERE / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    protected = {str(HERE / name): value for name, value in hashes.items()}
    protected[str(HERE / "artifact_hashes.json")] = sha(HERE / "artifact_hashes.json")
    protected.update({str(ROOT / name): value for name, value in identities().items()})
    protected.update({str(review / name): sha(review / name) for name in lineage["review_files"]})
    (HERE / "protected_hashes.sha256").write_text(
        "".join(
            value + "  " + str(Path(name).relative_to(ROOT)) + "\n"
            for name, value in sorted(protected.items())
        )
    )
    print(
        json.dumps(
            {
                "phase": state["phase"],
                "CPU_targeted_tests": 30,
                "CPU_deltas": deltas,
                "GPU_launched": False,
            }
        )
    )


if __name__ == "__main__":
    main()
