"""Freeze corrected candidate with v1 integrity and new CPU evidence."""

from __future__ import annotations

import json

from live_capture import HERE, identities, sha, verify_generated


def main() -> None:
    verify_generated()
    smoke = json.loads((HERE / "cpu_smoke_final/results.json").read_text())
    old = json.loads((HERE / "export_schema.json").read_text())
    old.update(
        {
            "version": 2,
            "optimizer_identity": "canonical source-reconstructed serialized IDs/names checked before load; optimizer_id_names required",
            "comparison": "byte-exact floats by default; integer/bool/RNG/types/clocks exact; frozen six-independent-default policy only on CUDA",
            "calibration": "max pairwise default-only floor*2 bounded by fixed abs1e-5/rel1e-4 ceiling; no online difference or automatic relaxation; NaN/inf masks exact",
            "cold_backend": "process warm, first reconstructed descriptor included; global cold process/cache unsupported",
            "measurement_generated_artifacts": "verification-only; no rewrite",
            "lineage": "v1 preserved FAIL_AS_FROZEN; actual full-fit states not recovered",
        }
    )
    old["required_top_level"].append("optimizer_id_names")
    old["post"].extend(["out", "out_ema", "target", "clip_norm"])
    (HERE / "export_schema.json").write_text(json.dumps(old, indent=2) + "\n")
    state = {
        "phase": "frozen_v2_request_independent_review",
        "requested_model": "gpt6.1-sol medium",
        "role": "round4-capture",
        "lineage": "v1 FAIL_AS_FROZEN preserved",
        "CPU_tests_v2": 37,
        "v1_CPU25_evidence_not_reused_as_v2": True,
        "smoke_seconds": smoke["seconds"],
        "CPU_max_deltas": {
            name: smoke[name]
            for name in (
                "normal_max_delta",
                "repeat_max_delta",
                "skip_max_delta",
                "uninstrumented_default_max_delta",
            )
        },
        "independent_baselines_per_control": 6,
        "CPU_floor_normal": smoke["independent_floor"]["normal_max_floor"],
        "CPU_floor_skip": smoke["independent_floor"]["skip_max_floor"],
        "GPU_floor_measured": False,
        "GPU_launched": False,
        "receipt_written": False,
        "candidate_self_approved": False,
        "today_full_fit_states_recovered": False,
        "controller_patch_applied": False,
        "real_panel_unchanged": True,
        "live_trainer_edits": False,
        "queue_changes": False,
        "source": identities(),
        "pending": [
            "independent v2 review and preserved adversary rerun",
            "controller payload patch/runtime hash attachment",
            "independent queue drain and exclusive admission",
            "CUDA six-default empirical floors then exact online/default/skip gates before any scientific contrast",
        ],
        "last_failure": None,
        "supported_compatibility": "fixed actual Muon/OneCycle/FastEMA/fp16 CUDA recipe; CPU fp32 tiny actual-code protocol; see export_schema",
        "global_cold_CUDA": "unsupported",
        "backup": "retain complete ~600MB prestate; controller streams backup beyond old512MiB part bound",
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
        "status": "pending_independent_review",
        "candidate_sha256": {name: sha(HERE / name) for name in names},
        "source": identities(),
        "runtime_manifest_sha256": sha(HERE / "runtime_manifest.json"),
        "controller_patch_sha256": sha(HERE / "controller_payload.patch"),
        "approval_or_receipt": False,
        "request": "Independent reviewer reruns optimizer-ID, clock, empirical-floor and pristine executable adversaries; controller owns future runtime/queue attestation",
    }
    (HERE / "review_request.json").write_text(json.dumps(request, indent=2) + "\n")
    (HERE / "results.json").write_text(
        json.dumps(
            {
                "state": state,
                "CPU_smoke": smoke,
                "lineage": json.loads((HERE / "LINEAGE.json").read_text()),
            },
            indent=2,
        )
        + "\n"
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
                "CPU_tests": 37,
                "CPU_floor_normal": 0,
                "CPU_floor_skip": 0,
                "GPU_launched": False,
            }
        )
    )


if __name__ == "__main__":
    main()
