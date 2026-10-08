from __future__ import annotations

import hashlib
import json
import sys
from difflib import unified_diff
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKET = ROOT / "packet"
WORKTREE = ROOT.parents[4]
V7A_ROOT = ROOT.parent / "h232_conditional_v7_proactive_20261008"
V7A_PACKET = V7A_ROOT / "packet"
AUTH_PATH = ROOT / "V7B_ROOT_DELIVERY_AUTHORIZATION_20261008_R1.json"
ADMISSION_PATH = ROOT / "basis/H232_V7_ROOT_PUSH_ADMISSION_20261008_R1.json"
PUSH_PATH = ROOT / "basis/H232_V7_PROVIDER_PUSH_20261008_R1.json"
TERMINAL_PATH = WORKTREE / "delivery/h232_v7_proactive/version1/V7A_TERMINAL_RECEIPT.json"
KERNEL_ID = "vladkochetov07/h232-gru384-b8-amp4096-v7b-proactive-cache-recovery"
KERNEL_TITLE = "H232 GRU384 B8 AMP4096 V7B Proactive Cache Recovery"
REVIEWED_FILES = (
    "profile_worker.py",
    "profile_kernel.py",
    "gpu_memory_domains.py",
    "verified_digest_handoff.py",
    "prior_input_identity_v2.json",
    "H232_PROFILE_PLAN.json",
    "H232_AMP4096_SUCCESSOR_RECIPE.json",
    "source_archive_manifest.json",
    "cache_pressure_recovery.py",
    "source_archive.tar",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes())


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reviewed_hashes(admission: dict[str, object]) -> dict[str, str]:
    packet_hashes = admission["packet_files"]
    if not isinstance(packet_hashes, dict):
        raise ValueError("V7A root admission has no packet hash map")
    reviewed = {name: sha256_file(PACKET / name) for name in REVIEWED_FILES}
    for name, observed in reviewed.items():
        if packet_hashes.get(name) != observed:
            raise ValueError(f"V7B core packet differs from root-reviewed V7A file: {name}")
    return reviewed


def _build_review(
    reviewed_hashes: dict[str, str],
    admission: dict[str, object],
    push: dict[str, object],
    authorization: dict[str, object],
    terminal: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": "h232-independent-review-v1",
        "experiment_id": "E529_H232_GRU384_B8_S3_AMP4096_PROACTIVE_CACHE_RECOVERY_20261008_V7A",
        "scope": "conditional-profile-gated-fit",
        "status": "approved",
        "conditional_fit_authorized": True,
        "fit_admission": "same-worker-only-after-profile-gate",
        "provider_push_authorized": False,
        "reviewer": {
            "id": "independent-luna-max-reviewer-qualified-pass-24-tests",
            "role": "independent-controller",
            "model": "gpt-6-luna",
            "reasoning_effort": "max",
        },
        "reviewed_hashes": reviewed_hashes,
        "qualified_independent_review": {
            "result": "qualified_pass",
            "focused_tests_passed": 24,
            "basis": "V7A root push admission and independent reviewer finding",
            "scope": [
                "all V7A packet hashes and unchanged scientific source archive",
                "identity-bound recent rowgroup ranges and bounded file-cache advice",
                "28% proactive trigger, unchanged 25% and 6.5 GB hard gates, and 512 MB target margin",
            ],
        },
        "root_approval": {
            "prior_v7a_admission_path": "basis/H232_V7_ROOT_PUSH_ADMISSION_20261008_R1.json",
            "prior_v7a_admission_sha256": sha256_file(ADMISSION_PATH),
            "prior_v7a_push_receipt_path": "basis/H232_V7_PROVIDER_PUSH_20261008_R1.json",
            "prior_v7a_push_receipt_sha256": sha256_file(PUSH_PATH),
            "prior_v7a_packet_commit": admission.get("packet_commit"),
            "v7b_delivery_authorization_path": AUTH_PATH.name,
            "v7b_delivery_authorization_sha256": sha256_file(AUTH_PATH),
            "v7b_scope": "delivery-only review receipt correction; a separate fresh root admission is required before V7B push",
            "provider_push_authorized": False,
        },
        "v7a_terminal_basis": {
            "path": "delivery/h232_v7_proactive/version1/V7A_TERMINAL_RECEIPT.json",
            "sha256": sha256_file(TERMINAL_PATH),
            "status": terminal.get("provider_status"),
            "failure": terminal.get("failure"),
        },
        "resource_review": {
            "proactive_cgroup_headroom_fraction": 0.28,
            "hard_cgroup_headroom_fraction": 0.25,
            "hard_cgroup_headroom_bytes": 6_500_000_000,
            "post_recovery_margin_bytes": 512_000_000,
            "max_advice_bytes_per_event": 768_000_000,
            "max_relief_events": 320,
            "max_cumulative_advice_bytes": 245_760_000_000,
            "minimum_relief_interval_seconds": 20,
            "range_policy": "recent traced rowgroups bound to the existing raw-file identity; no system-wide cache flush",
        },
        "runtime_accounting_limitation": {
            "status": "nonblocking_but_unresolved",
            "note": "Root's review records a zero-second CUDA-cleanup phase and a pre-cleanup projection. The 16000-second worker hard wall remains enforced; do not promote the prior duration estimate.",
            "provider_quota_and_startup": "must be freshly admitted by root before V7B push",
        },
        "note": "This receipt authorizes only same-worker conditional fit after the unchanged strict profile gate. It does not authorize provider push; the V7A placeholder-review error is preserved as execution evidence.",
    }


def _build_v7a_diff() -> bytes:
    pairs = (
        "packet/external_review.json",
        "packet/packet_meta.json",
        "packet/kernel-metadata.json",
    )
    chunks = [
        "# V7B delivery-only metadata and review correction against immutable V7A.\n",
        "# The generated Kaggle wrapper is omitted; rebuild it with build_h232_bundle.py.\n",
        "# Scientific source, recipe, split, profile, fit, and execution guard files are byte-identical.\n",
    ]
    for name in pairs:
        chunks.extend(
            unified_diff(
                (V7A_ROOT / name).read_text(encoding="utf-8").splitlines(keepends=True),
                (ROOT / name).read_text(encoding="utf-8").splitlines(keepends=True),
                fromfile=f"V7A/{name}",
                tofile=f"V7B/{name}",
                n=0,
            )
        )
    path = ROOT / "V7B_V7A_DELIVERY_DIFF.patch"
    diff_bytes = "".join(chunks).encode("utf-8")
    path.write_bytes(diff_bytes)
    return diff_bytes


def _assert_scientific_packet_unchanged() -> None:
    allowed_packet_changes = {
        "external_review.json",
        "packet_meta.json",
        "kernel-metadata.json",
        "h232_profile_kernel_bundle.py",
    }
    old_files = {path.name for path in V7A_PACKET.iterdir() if path.is_file()}
    new_files = {path.name for path in PACKET.iterdir() if path.is_file()}
    if old_files != new_files:
        raise ValueError("V7B packet member set differs from V7A")
    for name in sorted(old_files - allowed_packet_changes):
        if (V7A_PACKET / name).read_bytes() != (PACKET / name).read_bytes():
            raise ValueError(f"V7B changed protected V7A packet file: {name}")
    if (V7A_ROOT / "source").exists():
        old_sources = {
            path.relative_to(V7A_ROOT / "source").as_posix(): sha256_file(path)
            for path in (V7A_ROOT / "source").rglob("*")
            if path.is_file()
        }
        new_sources = {
            path.relative_to(ROOT / "source").as_posix(): sha256_file(path)
            for path in (ROOT / "source").rglob("*")
            if path.is_file()
        }
        if old_sources != new_sources:
            raise ValueError("V7B extracted training source tree differs from V7A")


def build() -> dict[str, object]:
    authorization = _read_json(AUTH_PATH)
    admission = _read_json(ADMISSION_PATH)
    push = _read_json(PUSH_PATH)
    terminal = _read_json(TERMINAL_PATH)
    if authorization.get("provider_push_authorized") is not False:
        raise ValueError("V7B delivery authorization must not authorize provider push")
    if admission.get("experiment_id") != "E529_H232_GRU384_B8_S3_AMP4096_PROACTIVE_CACHE_RECOVERY_20261008_V7A":
        raise ValueError("V7A admission basis is for a different experiment")
    if push.get("push_count") != 1 or push.get("version") != 1:
        raise ValueError("V7A provider push receipt is not the expected single version-1 push")
    if terminal.get("provider_status") != "KernelWorkerStatus.ERROR":
        raise ValueError("V7A terminal evidence does not record the diagnosed error")

    reviewed_hashes = _reviewed_hashes(admission)
    review = _build_review(reviewed_hashes, admission, push, authorization, terminal)
    review_path = PACKET / "external_review.json"
    write_json(review_path, review)

    meta = _read_json(PACKET / "packet_meta.json")
    meta["external_review_sha256"] = sha256_file(review_path)
    meta["reviewed_hashes"] = reviewed_hashes
    meta["current_packet_hashes"] = reviewed_hashes
    meta["status"] = "v7b_review_bound_pending_fresh_root_push_admission"
    meta["provider_push_authorized"] = False
    write_json(PACKET / "packet_meta.json", meta)

    kernel_meta = _read_json(PACKET / "kernel-metadata.json")
    kernel_meta["id"] = KERNEL_ID
    kernel_meta["title"] = KERNEL_TITLE
    write_json(PACKET / "kernel-metadata.json", kernel_meta)

    _assert_scientific_packet_unchanged()

    sys.path.insert(0, str(ROOT))
    from build_h232_bundle import build as build_bundle
    from build_h232_source_archive import build as build_source_archive

    source_result = build_source_archive()
    if source_result["source_archive_byte_identical_to_v6"] is not True:
        raise ValueError("V7B source archive is not byte-identical to V6")
    bundle_result = build_bundle(PACKET)
    diff_bytes = _build_v7a_diff()

    packet_files = admission["packet_files"]
    if not isinstance(packet_files, dict):
        raise ValueError("V7A admission packet hash map is malformed")
    v7b_hashes = {name: sha256_file(PACKET / name) for name in sorted(packet_files)}
    if len(v7b_hashes) != 14:
        raise ValueError(f"Expected 14 V7B packet hashes, observed {len(v7b_hashes)}")
    changed_packet_files = {
        name for name, old_hash in packet_files.items() if v7b_hashes[name] != old_hash
    }
    expected_delivery_changes = {
        "external_review.json",
        "packet_meta.json",
        "kernel-metadata.json",
        "h232_profile_kernel_bundle.py",
    }
    if changed_packet_files != expected_delivery_changes:
        raise ValueError(f"Unexpected V7B packet differences: {sorted(changed_packet_files)}")
    if v7b_hashes["source_archive.tar"] != source_result["source_archive_sha256"]:
        raise ValueError("V7B packet hash map and source archive result disagree")

    test_hashes = {
        path.name: sha256_file(path) for path in sorted((ROOT / "tests").glob("*.py"))
    }
    receipt = {
        "schema": "h232-v7b-delivery-build-receipt-v1",
        "variant": "V7B",
        "experiment_id": meta["packet_id"],
        "kernel_id": KERNEL_ID,
        "kernel_title": KERNEL_TITLE,
        "status": "built_pending_fresh_root_push_admission",
        "provider_push_authorized": False,
        "provider_status": "not_queried_for_v7b",
        "v7a_terminal_receipt_sha256": sha256_file(TERMINAL_PATH),
        "v7a_root_admission_sha256": sha256_file(ADMISSION_PATH),
        "v7a_push_receipt_sha256": sha256_file(PUSH_PATH),
        "v7b_delivery_authorization_sha256": sha256_file(AUTH_PATH),
        "external_review_sha256": sha256_file(review_path),
        "reviewed_hashes": reviewed_hashes,
        "packet_files_14": v7b_hashes,
        "packet_files_14_match_v7a": {
            name: v7b_hashes[name] == prior_hash
            for name, prior_hash in packet_files.items()
        },
        "source_archive": {
            "sha256": source_result["source_archive_sha256"],
            "manifest_sha256": source_result["source_archive_manifest_sha256"],
            "byte_identical_to_v6": source_result["source_archive_byte_identical_to_v6"],
        },
        "bundle": bundle_result,
        "delivery_report_sha256": sha256_file(
            ROOT / "H232_V7B_DELIVERY_CORRECTION_REPORT_V1.md"
        ),
        "test_results_receipt_sha256": sha256_file(ROOT / "V7B_TEST_RECEIPT.json"),
        "v7a_review_diff": {
            "path": "V7B_V7A_DELIVERY_DIFF.patch",
            "sha256": sha256(diff_bytes),
            "bytes": len(diff_bytes),
        },
        "tests": test_hashes,
        "runtime_accounting": review["runtime_accounting_limitation"],
        "fit_or_score_claim": None,
    }
    write_json(ROOT / "V7B_BUILD_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
