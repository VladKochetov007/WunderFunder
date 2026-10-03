"""Freeze minimal RNG/moment grammar correction for independent v3 review."""

from __future__ import annotations

import json

from live_capture import HERE, identities, sha, verify_generated
from prepare_runtime import main as prepare_lineage


def main() -> None:
    prepare_lineage()
    verify_generated()
    if sha(HERE / "baselines.py") != sha(HERE.parent / "live_capture_v2/baselines.py"):
        raise ValueError("baseline protocol changed unexpectedly")
    result = json.loads((HERE / "cpu_smoke_corrected/results.json").read_text())
    schema = json.loads((HERE / "export_schema.json").read_text())
    schema.update(
        {
            "version": 3,
            "RNG_grammar": "canonical installed-private-engine schemas; MT19937 native tuple5 uint32shape624/native-int index/cache/native-float; Python canonical tuples; CPU/CUDA byte state metadata and private acceptance validated BEFORE all global setters",
            "NumPy_compare": "exact class/dtype/shape/byteorder/metadata/C-order bytes; signedzero/NaNpayload exact; object arrays unsupported",
            "native_float_scalars": "type plus packed float64 bytes",
            "moment_storage": "finite detached CPU fp32 exact parameter shape; canonical IDs/names; no dtype normalization during load",
            "lineage": "v1/v2 all files preserved; final critic NumPy+moment blockers addressed in v3; no full-fit recovery",
        }
    )
    (HERE / "export_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    lineage = json.loads((HERE / "LINEAGE.json").read_text())
    state = {
        "phase": "frozen_v3_request_final_independent_review",
        "role": "round4-capture",
        "requested_model": "gpt6.1-sol medium",
        "CPU_tests_v3": 66,
        "prior_tests_not_claimed_as_v3_evidence": True,
        "smoke_seconds": result["seconds"],
        "CPU_deltas": {
            key: result[key]
            for key in (
                "normal_max_delta",
                "repeat_max_delta",
                "skip_max_delta",
                "uninstrumented_default_max_delta",
            )
        },
        "normal_floor": result["independent_floor"]["normal_max_floor"],
        "skip_floor": result["independent_floor"]["skip_max_floor"],
        "independent_repeats_per_control": 6,
        "baseline_protocol_identical_to_v2": True,
        "critic_final_verdict_read": lineage["review_v2_verdict"],
        "minimal_additional_critic_moment_fix": True,
        "v1_v2_all_files_preserved": True,
        "actual_trainer_sha256": lineage["actual_trainer_sha256"],
        "panel_bindings_unchanged": lineage["panel_byte_identical_to_v2"],
        "NumPy_dtype_aliases_rejected_before_set": True,
        "moment_dtype_aliases_rejected_before_load": True,
        "legitimate_seed_independent_roundtrip_passed": True,
        "GPU_launched": False,
        "GPU_floor_measured": False,
        "CUDA_RNG_engine_validation_executed": False,
        "receipt_written": False,
        "self_approved": False,
        "controller_patch_applied": False,
        "live_source_edits": False,
        "queue_changes": False,
        "today_full_fit_states_recovered": False,
        "pending": [
            "independent final-v3 review",
            "controller scope/runtime binding and exclusive admission",
            "unrun CUDA floor/default/skip gates",
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
        "status": "pending_final_v3_independent_review",
        "approval_or_receipt": False,
        "candidate_sha256": {name: sha(HERE / name) for name in names},
        "source": identities(),
        "runtime_manifest_sha256": sha(HERE / "runtime_manifest.json"),
        "controller_patch_sha256": sha(HERE / "controller_payload.patch"),
        "review_scope": "exact NumPy/RNG and moment typed prestate; unchanged pristine normal/repeat/skip and prior three fixes",
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
    print(
        json.dumps(
            {
                "phase": state["phase"],
                "CPU_tests": 66,
                "CPU_normal_skip_floors": [state["normal_floor"], state["skip_floor"]],
                "GPU_launched": False,
                "review_verdict_read": state["critic_final_verdict_read"],
            }
        )
    )


if __name__ == "__main__":
    main()
