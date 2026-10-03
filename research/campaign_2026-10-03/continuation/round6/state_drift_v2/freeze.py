"""Freeze only this corrected candidate; verify original actor/critic/native preservation."""

from __future__ import annotations

import json

from protocol import HERE, V4, cap
from run_real import own_hashes
from runtime import LIMITS, ROOT, RUNTIME_FILES


def main() -> None:
    """Serialize terminal-ready manifests without issuing GPU approval."""
    results = json.loads((HERE / "results.json").read_text())
    assert results["passed"]
    preservation = {}
    for name in ("state_drift", "state_drift_review"):
        directory = HERE.parent / name
        manifest = json.loads((directory / "HASHES.json").read_text())
        for relative, expected in manifest.items():
            assert cap.sha(directory / relative) == expected, (name, relative)
        preservation[name] = len(manifest)
    original = HERE.parent / "state_drift"
    assert (HERE / "PREREG.json").read_bytes() == (original / "PREREG.json").read_bytes()
    assert cap.identities() == results["source_hashes"]
    assert all(
        cap.sha(V4 / name) == expected for name, expected in results["adapter_hashes"].items()
    )
    candidate = own_hashes()
    runtime_hashes = {name: cap.sha(ROOT / name) for name in RUNTIME_FILES}
    for name, value in (
        ("candidate_hashes.json", candidate),
        ("runtime_hashes.json", runtime_hashes),
        ("preservation.json", preservation),
    ):
        (HERE / name).write_text(json.dumps(value, indent=2) + "\n")
    state = {
        "state": "FROZEN_V2_CPU_PASS_TERMINAL_READY_GPU_UNAPPROVED",
        "checks": len(results["checks"]),
        "original_prereg_byte_identical": True,
        "warm_protocol": "original four completed/skips0 only; amendment not used",
        "actual_capture_reads": 0,
        "actual_GPU_calls": 0,
        "real_binary_tests": 0,
        "candidate_sha256": candidate,
        "runtime_sha256": runtime_hashes,
        "runtime_limits": LIMITS,
        "cost": results["cost"],
        "preservation": preservation,
        "next": "independent critic conditional re-review, then controller-bound receipt/runtime only",
    }
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    (HERE / "COST.json").write_text(
        json.dumps(
            {
                "numerical_CPU_run": results["cost"],
                "resources": "run_cpu.sh round2-critic,1core2GiB900s",
                "measurement": "main-body wall/CPU excludes Python imports; peak RSS process lifetime",
                "ordinary_exit1": "Ruff B023 finite-adversary loop lambda late-binding warnings; fixed by explicit default capture. Test calls were synchronous and all49 numerical checks already passed. No OOM/timeout/barrier.",
                "controller_220500_notice": "shared journal identifies round3-matched-usage failed Ruff command as round6/amp_linearity_audit_v2, not this actor; that actor's code/results were not inspected",
                "GPU_seconds": 0,
            },
            indent=2,
        )
        + "\n"
    )
    report = f"""# Corrected state-drift v2: terminal-ready CPU candidate

PASS: {len(results["checks"])} bounded CPU checks. Real scientific test remains UNRUN and UNAPPROVED. Frozen original actor and critic manifests are fully preserved ({preservation}); native source/adapter hashes still match. No actual capture, real prediction/model panel or GPU private RNG restoration was read/run. New files exist only under state_drift_v2. Independent critic re-review and controller approval remain mandatory; this report does not approve its own candidate.

The original PREREG.json is BYTE IDENTICAL. This first probe requires four completed warm updates and zero skips, checked against scalar int64 skip count, scheduler/attempt/EMA clocks and every actual Adam integer step. No warm-skip amendment is used. Thresholds remain hidden relative>=.01, main predictionRMS>=1e-4, whole gradient OR actual update relative>=.01, each required effect>5 times its independent RMS floor. The four source1000-row prefix calls, original student/current weights/normalization, initialNone, source train/grad-enabled forward context, detached local hidden, all four original loss branches, immutable target/weights/denominator semantics and whole-vector-only comparison remain unchanged. CPU null is exact; planted drift remains detectable subthreshold.

Blocker1 fixed: stale student AND EMA hidden are finite-checked before restore; reference hidden is checked at every prefix boundary and before local step; baseline/alternative full outputs, EMA output, targets, hidden, loss, clip norm, whole gradients and update vectors are explicitly finite checked. RMS rejects empty/nonfinite input and derived overflow. Independent policies/floors/ceilings and all final effect metrics are recursively finite-checked before classification/serialization; JSON disallows NaN. NaN/+Inf/-Inf adversaries now reject rather than classify unresolved. No tolerance/threshold relaxation occurred.

Blocker2 fixed in candidate: runtime.py is hash-bound in own_hashes. The NEW controller_state_drift_payload.sh and campaign_job/limited/run_gpu source hashes are compared to receipt.runtime_sha256; old R4 runtime remains only the separate default measurement receipt. Exact new entrypoint and resolved pre/post/trajectory/receipt/output CLI paths are also receipt-bound. Scope flags, new wrapper SHA handshake, deadline1791057600 and observed5.5GB environment are mandatory. Main sets the explicit allocator fraction min(.70,4_800_000_000/device_total) BEFORE any capture tensor load/model/private-RNG restore, preserving source configure settings. Observed TOTAL device usage is checked before/after default updates, each prefix chunk and the local scientific step; no GPU checks occur in CPU controls.

Date fixture verifies epoch1791057600 = 2026-10-03 20:00UTC / 23:00Kyiv. Reviewed wrapper source requires >=325seconds at entry, timeout300 plus kill5; this leaves >=20seconds slack. Candidate also arms a min(300,deadline-now-10) Python alarm and checks deadline at phases and final artifacts. Hard interruption of a stalled native/CUDA call depends on the separately controller-reviewed timeout/hard-kill wrapper; Python signals alone are not a guarantee. Receipt specifies outer320seconds, job8CPU/24GiB and root12CPU/36GiB; actual controller/cgroup enforcement is a later independent runtime gate. CPU tests remain1core/2GiB/900s. R4 wrapper is not changed.

Tensor identity now streams SHA256 over identical meta JSON then bounded1MiB uint8 memoryview slices. It performs no full .tobytes() or meta+fullbytes concatenation. Five small CPU fixtures (float32, float16/transposed shape, int64, empty and bfloat16) match the original hash byte-for-byte. Source-contiguous real inputs need no additional contiguous tensor copy; a noncontiguous input may still need one contiguous copy, rather than multiple full bytes copies. Sidecar hash grammar and estimand are unchanged.

Ordinary failure record: this actor's initial Ruff exit1 reported B023 on loop variable captures in new finite adversaries (initial_lint_failure.log). Tests invoke these lambdas synchronously and all49 numeric checks passed; explicit default captures fix lint without changing cases. The controller's22:05:00 round3-matched-usage notice is a DIFFERENT scope: its journal Started command names amp_linearity_audit_v2/audit.py,metadata.py,tests.py. No claim about that actor's exact lint rule/fix is made here, and its source/results were not opened. Neither notice is an OOM or timeout/resource failure. New candidate lint is checked before freeze.

Exact receipt interface is RECEIPT_INTERFACE.json. candidate_hashes.json and runtime_hashes.json supply the exact map values; trajectory still binds initialNone/four starts, actual complete4/skips0, source/panel/input/student normalization and executed source entrypoint. This actor creates an interface, NEVER an approval receipt. Controller must bind the actual new wrapper and outer resource/runtime enforcement, default-v4 PASS and independent state-drift critic PASS before admission. Runtime/source changes invalidate receipt hashes. Actual CUDA/fused/AMP numerical parity, altered-branch noise behavior and measured host/device peaks remain untested here; default floors cannot certify arbitrary altered-branch noise or component attribution.

Reproduce CPU controls into a NEW label: `bash research/campaign_2026-10-03/continuation/round6/state_drift_v2/run_cpu.sh --output-label independent_controls`. Existing floor directories are never overwritten. Current evidence is checks.log/results.json: {results["cost"]["wall_seconds"]:.3f}s main wall, {results["cost"]["cpu_seconds"]:.3f}s CPU, peak{results["cost"]["peak_rss_kib"]}KiB. This is instrument correctness only: no historical full-fit recovery, harmful drift, predictive necessity, local/public WP gain or recipe attribution.
"""
    (HERE / "REPORT.md").write_text(report)
    hashes = {
        str(p.relative_to(HERE)): cap.sha(p)
        for p in sorted(HERE.rglob("*"))
        if p.is_file() and p.name != "HASHES.json"
    }
    (HERE / "HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(
        json.dumps(
            {
                "frozen": True,
                "checks": len(results["checks"]),
                "GPU_approved": False,
                "receipt_interface": str(HERE / "RECEIPT_INTERFACE.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
