"""Render the locally reviewed parent-fit V3 CPU verifier submission packet."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[2]
PREPARATION = CAMPAIGN / "preparation"
RECORDS = CAMPAIGN / "records/parent_fit_r2_v1"
CACHE_RECORDS = CAMPAIGN / "records/teacher_cache_cpu_build_v2"
RNG_RECORDS = RECORDS / "rng_diagnostic_v2"
OUTPUT = Path(__file__).resolve().parent
TEMPLATE = OUTPUT / "REMOTE_SUBMISSION_TEMPLATE.py"
ROOT_PREPARATION = PREPARATION / "PARENT_FIT_TERMINAL_ROOT_PREPARATION_V1.json"
REMOTE_OUTPUT = (
    "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_parent_fit_r2_verify_v3"
)
RUN_NAME = "fg19_parent_h256x2_f0_s1001_fit_r2_v1"
NATIVE_UNIT = "computer-vision-homework-fg19-parent-fit-r2-v1.service"
VERIFIER_UNIT = "computer-vision-homework-fg19-parent-fit-r2-verify-v3.service"
INVOCATION = "5186b3016ff54b1fa754c10a1f87b3e4"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
MANIFEST_SHA = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
TERMINAL_EVIDENCE_SHA = "780e64538cb34dd9e6a9c7da528a4232d4362daca9bee1dd871bd722373f7615"
TERMINAL_RAW_SHA = "a65af766fafaaa811d0991072035b51ec04dc01353da3059d7c594dffef239cb"
CACHE_METADATA_SHA = "03d9cf1fbd20b8bb3ad7b7d0c4513293fd1d5128e816256e4bf3cdfbfa4b43e3"
RNG_REPORT_SHA = "0bb2a87b9796872d5e26666ea7398bca69b7f0a0ce27bb0895714b5ed654598b"
RNG_ACCEPTANCE_SHA = "655c5f12db5e166cfabf68c343aeb10d709f95db4abffeaf2925b413226837ec"
RNG_STOP_SHA = "8e935356da08ac249e47389bb65bc5597134e6442ab9d914da1e7ec256b73c67"
CUDA_POLICY_SOURCE_SHA = "95fdc839ff1561622ac06c465cfc9a01370130d99be1e832b3a33c79bcb37930"

INPUTS = {
    "parent_fit_terminal_check_v3.py": PREPARATION / "parent_fit_terminal_check_v3.py",
    "teacher_s5_terminal_check_v1.py": PREPARATION / "teacher_s5_terminal_check_v1.py",
    "propensity_terminal_support.py": PREPARATION / "propensity_terminal_support.py",
    "SOURCE_MANIFEST.json": PREPARATION / "teacher_cache_cpu_repair_v2/SOURCE_MANIFEST.json",
    "ADMISSION.json": RECORDS / "ADMISSION.json",
    "LAUNCH.json": RECORDS / "LAUNCH.json",
    "LAUNCH_ACCEPTANCE.json": RECORDS / "LAUNCH_ACCEPTANCE.json",
    "TERMINAL_EVIDENCE.json": RECORDS / "TERMINAL_EVIDENCE.json",
    "NATIVE_STDOUT.raw.json": RECORDS / "terminal_native_capture_v1/STDOUT.raw.json",
    "CACHE_ACCEPTANCE.json": CACHE_RECORDS / "TERMINAL_ACCEPTANCE.json",
    "CACHE_EXACT_STOP.json": CACHE_RECORDS / "exact_stop/STDOUT.raw.json",
    "RNG_DIAGNOSTIC.json": RNG_RECORDS / "capture/RNG_DIAGNOSTIC.json",
    "RNG_DIAGNOSTIC_ACCEPTANCE.json": RNG_RECORDS / "CAPTURE_ACCEPTANCE.json",
    "RNG_DIAGNOSTIC_EXACT_STOP.json": RNG_RECORDS / "exact_stop/STDOUT.raw.json",
}
EXPECTED_SHA256 = {
    "parent_fit_terminal_check_v3.py": "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41",
    "teacher_s5_terminal_check_v1.py": "cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5",
    "propensity_terminal_support.py": "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409",
    "SOURCE_MANIFEST.json": MANIFEST_SHA,
    "ADMISSION.json": "42b5f3fe7142d931b189ff86fa24031426506e60c8a41440b0483d97dc50026b",
    "LAUNCH.json": "012c7ba3b484f16f587c61cb39f880380d22a6199e0a105773dbffc423bd22ea",
    "LAUNCH_ACCEPTANCE.json": "89b8199b6431c8959fe3bab41a630632210b30ae017dd9128f9ad2491c87a3bf",
    "TERMINAL_EVIDENCE.json": TERMINAL_EVIDENCE_SHA,
    "NATIVE_STDOUT.raw.json": TERMINAL_RAW_SHA,
    "CACHE_ACCEPTANCE.json": "88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7",
    "CACHE_EXACT_STOP.json": "286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4",
    "RNG_DIAGNOSTIC.json": RNG_REPORT_SHA,
    "RNG_DIAGNOSTIC_ACCEPTANCE.json": RNG_ACCEPTANCE_SHA,
    "RNG_DIAGNOSTIC_EXACT_STOP.json": RNG_STOP_SHA,
}
MAX_INPUT_BYTES = {
    "parent_fit_terminal_check_v3.py": 75_000,
    "teacher_s5_terminal_check_v1.py": 40_000,
    "propensity_terminal_support.py": 15_000,
    "SOURCE_MANIFEST.json": 30_000,
    "ADMISSION.json": 10_000,
    "LAUNCH.json": 10_000,
    "LAUNCH_ACCEPTANCE.json": 2_000,
    "TERMINAL_EVIDENCE.json": 2_000,
    "NATIVE_STDOUT.raw.json": 400_000,
    "CACHE_ACCEPTANCE.json": 4_000,
    "CACHE_EXACT_STOP.json": 2_000,
    "RNG_DIAGNOSTIC.json": 5_000,
    "RNG_DIAGNOSTIC_ACCEPTANCE.json": 5_000,
    "RNG_DIAGNOSTIC_EXACT_STOP.json": 5_000,
}


def _sha256(raw: bytes) -> str:
    """Return a SHA-256 digest for an input binding."""
    return hashlib.sha256(raw).hexdigest()


def _json(raw: bytes, name: str) -> dict[str, object]:
    """Parse one metadata input and require an object at the root."""
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {name}")
    return value


def _validate_native_capture(payload: dict[str, bytes]) -> None:
    """Join typed evidence, raw capture, source, and launch metadata locally."""
    evidence = _json(payload["TERMINAL_EVIDENCE.json"], "terminal evidence")
    raw = payload["NATIVE_STDOUT.raw.json"]
    if _sha256(raw) != TERMINAL_RAW_SHA or _sha256(raw) != evidence.get("raw_sha256"):
        raise ValueError("captured raw terminal bytes differ from typed evidence")
    if len(raw) != evidence.get("raw_bytes"):
        raise ValueError("captured raw terminal byte count differs")
    capture = _json(raw, "raw terminal capture")
    native = capture.get("unit")
    if not isinstance(native, dict) or capture.get("terminal_success") is not True:
        raise ValueError("raw terminal capture is not a successful native exit")
    expected_native = (INVOCATION, "active", "exited", "0", "1", "0")
    actual_native = tuple(
        native.get(key)
        for key in (
            "InvocationID",
            "ActiveState",
            "SubState",
            "MainPID",
            "ExecMainCode",
            "ExecMainStatus",
        )
    )
    if actual_native != expected_native:
        raise ValueError("raw native unit identity differs")
    if (
        evidence.get("schema"),
        evidence.get("run_name"),
        evidence.get("unit"),
        evidence.get("invocation_id"),
        evidence.get("active_state"),
        evidence.get("sub_state"),
    ) != (
        "fg19.parent-fit-r2-native-terminal-evidence.v1",
        RUN_NAME,
        NATIVE_UNIT,
        INVOCATION,
        "active",
        "exited",
    ):
        raise ValueError("typed terminal evidence identity differs")
    for key, expected in (("main_pid", 0), ("exec_main_code", 1), ("exec_main_status", 0)):
        if type(evidence.get(key)) is not int or evidence[key] != expected:
            raise ValueError(f"typed native exit field differs: {key}")
    captured_files = capture.get("files")
    if not isinstance(captured_files, dict):
        raise ValueError("raw terminal capture file identities are missing")
    for name in ("ADMISSION.json", "LAUNCH.json"):
        record = captured_files.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"raw terminal capture omits {name}")
        content = base64.b64decode(record["base64"], validate=True)
        if content != payload[name] or _sha256(content) != record.get("sha256"):
            raise ValueError(f"raw terminal capture does not bind {name}")
    manifest = _json(payload["SOURCE_MANIFEST.json"], "source manifest")
    entries = manifest.get("files")
    if (
        manifest.get("source_commit") != COMMIT
        or not isinstance(entries, list)
        or len(entries) != 167
    ):
        raise ValueError("source manifest identity differs")
    source_map = {row["path"]: row["sha256"] for row in entries if isinstance(row, dict)}
    admission = _json(payload["ADMISSION.json"], "parent admission")
    if (
        admission.get("source_files_verified") != 167
        or admission.get("source_manifest_sha256") != MANIFEST_SHA
    ):
        raise ValueError("parent admission/source manifest map differs")
    if capture.get("source_files") != source_map:
        raise ValueError("native capture source map differs")
    launch = _json(payload["LAUNCH.json"], "parent launch")
    accepted = _json(payload["LAUNCH_ACCEPTANCE.json"], "parent launch acceptance")
    if launch.get("unit") != NATIVE_UNIT or launch.get("submission", {}).get("returncode") != 0:
        raise ValueError("parent raw launch identity differs")
    if (
        accepted.get("schema"),
        accepted.get("status"),
        accepted.get("invocation_id"),
        accepted.get("admission_sha256"),
        accepted.get("launch_sha256"),
    ) != (
        "fg19-parent-fit-r2-launch-acceptance.v1",
        "ACTUAL_LAUNCHED_FULL_FIT_PENDING",
        INVOCATION,
        EXPECTED_SHA256["ADMISSION.json"],
        EXPECTED_SHA256["LAUNCH.json"],
    ):
        raise ValueError("parent launch acceptance links differ")


def _validate_rng_diagnostic(payload: dict[str, bytes]) -> None:
    """Bind the actual V2 RNG report, capture acceptance and exact stop."""
    report = _json(payload["RNG_DIAGNOSTIC.json"], "RNG diagnostic")
    acceptance = _json(payload["RNG_DIAGNOSTIC_ACCEPTANCE.json"], "RNG diagnostic acceptance")
    stop = _json(payload["RNG_DIAGNOSTIC_EXACT_STOP.json"], "RNG diagnostic exact stop")
    if (
        report.get("schema"),
        report.get("status"),
        report.get("run_name"),
        report.get("torch_version"),
        report.get("next_epoch"),
    ) != (
        "fg19.parent-fit-r2-rng-diagnostic.v2",
        "DIAGNOSTIC_ONLY_NO_ACCEPTANCE",
        RUN_NAME,
        "2.14.0+cu130",
        4,
    ):
        raise ValueError("RNG diagnostic report identity differs")
    if (
        acceptance.get("schema"),
        acceptance.get("status"),
        acceptance.get("unit"),
        acceptance.get("invocation_id"),
        acceptance.get("report_sha256"),
        acceptance.get("raw_sha256"),
        acceptance.get("quality_score"),
        acceptance.get("native_untouched"),
    ) != (
        "fg19.parent-rng-diagnostic-capture-acceptance.v2",
        "CAPTURE_VALIDATED_DIAGNOSTIC_ONLY_NOT_PARENT_ACCEPTANCE",
        "computer-vision-homework-fg19-parent-rng-diag-v2.service",
        "b9dcbfc0b8054e51a49fc3792fd037a0",
        RNG_REPORT_SHA,
        "2f6ced10d3812a3b3ae8354268a193323df6a22ea8523fdd2458060f4b8fcf69",
        None,
        True,
    ):
        raise ValueError("RNG diagnostic capture acceptance differs")
    before = stop.get("before")
    after = stop.get("after")
    action = stop.get("stop")
    if not isinstance(before, dict) or not isinstance(after, dict) or not isinstance(action, dict):
        raise ValueError("RNG diagnostic exact-stop record is incomplete")
    if (
        stop.get("unit"),
        stop.get("invocation_id"),
        stop.get("capture_acceptance_sha256"),
        stop.get("diagnostic_report_sha256"),
        stop.get("raw_capture_sha256"),
        before.get("InvocationID"),
        before.get("ActiveState"),
        before.get("SubState"),
        before.get("MainPID"),
        before.get("ExecMainCode"),
        before.get("ExecMainStatus"),
        action.get("returncode"),
        after.get("LoadState"),
        after.get("ActiveState"),
        after.get("SubState"),
        after.get("InvocationID"),
        stop.get("populated_before"),
        stop.get("populated_after"),
    ) != (
        "computer-vision-homework-fg19-parent-rng-diag-v2.service",
        "b9dcbfc0b8054e51a49fc3792fd037a0",
        RNG_ACCEPTANCE_SHA,
        RNG_REPORT_SHA,
        "2f6ced10d3812a3b3ae8354268a193323df6a22ea8523fdd2458060f4b8fcf69",
        "b9dcbfc0b8054e51a49fc3792fd037a0",
        "active",
        "exited",
        "0",
        "1",
        "0",
        0,
        "not-found",
        "inactive",
        "dead",
        "",
        {},
        {},
    ):
        raise ValueError("RNG diagnostic exact-stop record differs")


def _read_payload() -> dict[str, bytes]:
    """Read only the bounded checker/helper and frozen JSON evidence inputs."""
    if set(INPUTS) != set(EXPECTED_SHA256) or set(INPUTS) != set(MAX_INPUT_BYTES):
        raise ValueError("fixed payload mapping differs")
    payload = {}
    for name, path in INPUTS.items():
        raw = path.read_bytes()
        if len(raw) > MAX_INPUT_BYTES[name]:
            raise ValueError(f"input exceeds its local cap: {name}")
        if _sha256(raw) != EXPECTED_SHA256[name]:
            raise ValueError(f"frozen input digest differs: {name}")
        payload[name] = raw
    _validate_native_capture(payload)
    manifest = _json(payload["SOURCE_MANIFEST.json"], "source manifest")
    resource_rows = [
        row
        for row in manifest.get("files", [])
        if isinstance(row, dict) and row.get("path") == "src/wnn/resources.py"
    ]
    if len(resource_rows) != 1 or resource_rows[0].get("sha256") != CUDA_POLICY_SOURCE_SHA:
        raise ValueError("CUDA resource-policy source pin differs")
    _validate_rng_diagnostic(payload)
    cache = _json(payload["CACHE_ACCEPTANCE.json"], "cache acceptance")
    if (
        cache.get("status"),
        cache.get("metadata_sha256"),
        cache.get("source_commit"),
        cache.get("source_files_verified"),
    ) != ("ACCEPTED", CACHE_METADATA_SHA, COMMIT, 167):
        raise ValueError("accepted cache identity differs")
    return payload


def main() -> None:
    """Create one local rendered body and preparation receipt without transmission."""
    payload = _read_payload()
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("PAYLOAD = {}") != 1:
        raise ValueError("template payload slot must appear exactly once")
    packaged = {
        name: {
            "bytes": len(raw),
            "sha256": _sha256(raw),
            "base64": base64.b64encode(raw).decode("ascii"),
        }
        for name, raw in payload.items()
    }
    body = template.replace("PAYLOAD = {}", "PAYLOAD = " + repr(packaged), 1)
    compile(body, "REMOTE_SUBMISSION.py", "exec")
    required = (
        "fg19_parent_fit_r2_verify_v3",
        VERIFIER_UNIT,
        "MemoryHigh=1400000000",
        "MemoryMax=1500000000",
        "RuntimeMaxSec=600",
        "NATIVE_STDOUT.raw.json",
        "CACHE_EXACT_STOP.json",
        "RNG_DIAGNOSTIC.json",
        "--rng-diagnostic-exact-stop",
    )
    if any(token not in body for token in required):
        raise ValueError("rendered body omits an exact identity or resource binding")
    body_path = OUTPUT / "REMOTE_SUBMISSION.py"
    if body_path.exists() or (OUTPUT / "PREPARATION.json").exists():
        raise FileExistsError("local verifier packet already exists; inspect before replacement")
    body_path.write_text(body, encoding="utf-8", newline="\n")
    root_preparation_sha = _sha256(ROOT_PREPARATION.read_bytes())
    receipt = {
        "schema": "fg19.parent-fit-r2-terminal-verifier-preparation.v3",
        "status": "PREPARED_LOCALLY_NOT_TRANSMITTED_OR_LAUNCHED",
        "run_name": RUN_NAME,
        "native_unit": NATIVE_UNIT,
        "native_invocation_id": INVOCATION,
        "native_evidence_sha256": TERMINAL_EVIDENCE_SHA,
        "native_raw_capture_sha256": TERMINAL_RAW_SHA,
        "rng_diagnostic_sha256": RNG_REPORT_SHA,
        "rng_diagnostic_acceptance_sha256": RNG_ACCEPTANCE_SHA,
        "rng_diagnostic_exact_stop_sha256": RNG_STOP_SHA,
        "source_commit": COMMIT,
        "source_file_count": 167,
        "source_manifest_sha256": MANIFEST_SHA,
        "root_preparation_manifest_at_render_sha256": root_preparation_sha,
        "verifier_unit": VERIFIER_UNIT,
        "remote_output": REMOTE_OUTPUT,
        "remote_body_sha256": _sha256(body.encode()),
        "remote_body_bytes": len(body.encode()),
        "template_sha256": _sha256(template.encode()),
        "generator_sha256": _sha256(Path(__file__).read_bytes()),
        "embedded_inputs": {name: EXPECTED_SHA256[name] for name in sorted(payload)},
        "code_dependencies": {
            name: EXPECTED_SHA256[name]
            for name in (
                "parent_fit_terminal_check_v3.py",
                "teacher_s5_terminal_check_v1.py",
                "propensity_terminal_support.py",
            )
        },
        "cache_metadata_sha256": CACHE_METADATA_SHA,
        "resource_limits": {
            "cpu_quota": "50%",
            "memory_high_bytes": 1_400_000_000,
            "memory_max_bytes": 1_500_000_000,
            "swap_max_bytes": 0,
            "runtime_seconds": 600,
            "cuda_visible_devices": "",
            "blas_threads": 1,
        },
        "fresh_remote_gates": [
            "Output path and new verifier unit absent; exact native invocation still active/exited with typed successful status.",
            "All-populated parent descendants empty, two-core/18GB/20GB/swap-zero envelope, one-CPU-support headroom under the 17GB target, physical scratch, and at least 25 GiB free on project storage.",
            "Exact 167-file source manifest/tree, split, registered data stats, run artifact stats, accepted cache metadata/stats and cache acceptance/exact-stop receipts.",
            "Create-only staging of the three pinned Python files and bounded JSON metadata; no cache prediction array bytes are read or copied by admission.",
        ],
        "evidence_limits": [
            "The prepared body does not verify the fit; the pinned checker runs once after admission under the CPU-only caps.",
            "The checker reads bounded model and training-state files under its restricted loader; neither the preparation step nor this body reads cache prediction array payloads.",
            "The checker verifies the required counters and epoch-3 endpoint; no quality score is requested or inferred.",
            "Profile ancestry is transitively bound by the exact parent-fit admission hash; this packet does not re-evaluate the profile.",
            "The CUDA RNG length and bytes are source-qualified by the actual pinned V2 diagnostic and stop receipts; the diagnostic does not substitute for full runtime provenance checks.",
        ],
        "unknown_outcome_policy": "Durably retain verifier launch outcome; inspect exact unit and output before any further action; never retry blindly.",
        "remote_action_by_this_preparation": False,
    }
    prep_path = OUTPUT / "PREPARATION.json"
    prep_path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "body_sha256": receipt["remote_body_sha256"],
                "body_bytes": receipt["remote_body_bytes"],
                "preparation": str(prep_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
