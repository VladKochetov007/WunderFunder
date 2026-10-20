"""Render one local, identity-bound RNG diagnostic admission packet."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[2]
PREPARATION = CAMPAIGN / "preparation"
RECORDS = CAMPAIGN / "records/parent_fit_r2_v1"
FAILED_CAPTURE_DIR = RECORDS / "terminal_verification_v1/capture"
OLD_PACKET = PREPARATION / "parent_fit_terminal_verifier_v1"
OUTPUT = Path(__file__).resolve().parent
OLD_TEMPLATE = OLD_PACKET / "REMOTE_SUBMISSION_TEMPLATE.py"
RNG_SCRIPT = OUTPUT / "rng_diagnostic_v1.py"
TEMPLATE = OUTPUT / "REMOTE_SUBMISSION_TEMPLATE.py"
BODY = OUTPUT / "REMOTE_SUBMISSION.py"
PREP_RECEIPT = OUTPUT / "PREPARATION.json"

OLD_TEMPLATE_SHA = "c07446cc2841c53fb2e9dc28bdfae03dc3657a217ead1d7168a614909c562dd2"
RENDERED_TEMPLATE_SHA = "19bc9373b2424b5b8d0603a9469bbbd4a8a3852533ef8dc6503f3a2ce5265fd9"
FAILED_RAW_SHA = "7761927b6bcacd034af294ee5f37c30561b6ff3b07b846f0e7393324fea1a359"
FAILED_ADMISSION_SHA = "b89cd063f2409c5551af68819ac4d9f9274d362e2074cd413dce0df0877027d7"
FAILED_LAUNCH_SHA = "ad718e0cb72c11e3a4ea90d4ab5ac694d705d81166f020f10fe0acab9a6e1e12"
FAILED_LOG_SHA = "536e2019e084a40eeb0848d9288751f721f609a49a514d629f1d8de5515a2d70"
FAILED_INVOCATION = "7ecaf59d01ba4a07b0971a1c9902d87e"
FAILED_UNIT = "computer-vision-homework-fg19-parent-fit-r2-verify-v1.service"
NEW_UNIT = "computer-vision-homework-fg19-parent-rng-diag-v1.service"
RUN = "fg19_parent_h256x2_f0_s1001_fit_r2_v1"
STATE_RELATIVE = "training_state.pt"

FAILED_INPUTS = {
    "FAILED_VERIFIER_CAPTURE.raw.json": (
        FAILED_CAPTURE_DIR / "STDOUT.raw.json",
        FAILED_RAW_SHA,
        100_000,
    ),
    "FAILED_VERIFIER_ADMISSION.json": (
        FAILED_CAPTURE_DIR / "VERIFIER_ADMISSION.json",
        FAILED_ADMISSION_SHA,
        20_000,
    ),
    "FAILED_VERIFIER_LAUNCH.json": (
        FAILED_CAPTURE_DIR / "VERIFIER_LAUNCH.json",
        FAILED_LAUNCH_SHA,
        10_000,
    ),
    "FAILED_VERIFIER_UNIT_LOG.raw": (FAILED_CAPTURE_DIR / "unit.log", FAILED_LOG_SHA, 5_000),
}


def _sha256(raw: bytes) -> str:
    """Return SHA-256 for bounded local source or metadata bytes."""
    return hashlib.sha256(raw).hexdigest()


def _object(raw: bytes, label: str) -> dict[str, object]:
    """Decode a JSON object from source-qualified metadata bytes."""
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {label}")
    return value


def _load_old_preparer():
    """Load the frozen predecessor's stdlib-only metadata binder."""
    path = OLD_PACKET / "prepare_parent_fit_terminal_verifier_v1.py"
    spec = importlib.util.spec_from_file_location("parent_fit_verifier_preparer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen predecessor preparer cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_payload() -> dict[str, bytes]:
    """Join the frozen native/cache package to the exact failed-verifier capture."""
    old = _load_old_preparer()
    payload = dict(old._read_payload())
    for name, (path, expected_sha, cap) in FAILED_INPUTS.items():
        raw = path.read_bytes()
        if len(raw) > cap or _sha256(raw) != expected_sha:
            raise ValueError(f"failed-verifier metadata pin differs: {name}")
        payload[name] = raw

    failed_admission = _object(payload["FAILED_VERIFIER_ADMISSION.json"], "failed admission")
    state_stat = (
        failed_admission.get("terminal_files_stat_only", {}).get("files", {}).get(STATE_RELATIVE)
    )
    if not isinstance(state_stat, dict) or set(state_stat) != {
        "bytes",
        "device",
        "inode",
        "mtime_ns",
    }:
        raise ValueError("failed admission does not contain the exact state stat pin")
    payload["RNG_STATE_STAT.json"] = (
        json.dumps(state_stat, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    rng_script = RNG_SCRIPT.read_bytes()
    if len(rng_script) > 12_000:
        raise ValueError("RNG diagnostic script exceeds its source cap")
    payload["RNG_DIAGNOSTIC.py"] = rng_script

    raw_capture = _object(payload["FAILED_VERIFIER_CAPTURE.raw.json"], "failed capture")
    captured_files = raw_capture.get("files")
    if not isinstance(captured_files, dict):
        raise ValueError("failed capture omits bounded file identities")
    for captured_name, payload_name in (
        ("VERIFIER_ADMISSION.json", "FAILED_VERIFIER_ADMISSION.json"),
        ("VERIFIER_LAUNCH.json", "FAILED_VERIFIER_LAUNCH.json"),
        ("unit.log", "FAILED_VERIFIER_UNIT_LOG.raw"),
    ):
        record = captured_files.get(captured_name)
        if not isinstance(record, dict):
            raise ValueError(f"failed capture omits {captured_name}")
        decoded = base64.b64decode(record.get("base64", ""), validate=True)
        if decoded != payload[payload_name] or _sha256(decoded) != record.get("sha256"):
            raise ValueError(f"failed capture file binding differs: {captured_name}")
    return payload


def _failed_attempt_helper() -> str:
    """Return the remote check that binds the previous failed verifier without querying it."""
    return r'''
FAILED_UNIT = "computer-vision-homework-fg19-parent-fit-r2-verify-v1.service"
FAILED_INVOCATION = "7ecaf59d01ba4a07b0971a1c9902d87e"
FAILED_RAW_SHA = "7761927b6bcacd034af294ee5f37c30561b6ff3b07b846f0e7393324fea1a359"
FAILED_ADMISSION_SHA = "b89cd063f2409c5551af68819ac4d9f9274d362e2074cd413dce0df0877027d7"
FAILED_LAUNCH_SHA = "ad718e0cb72c11e3a4ea90d4ab5ac694d705d81166f020f10fe0acab9a6e1e12"
FAILED_LOG_SHA = "536e2019e084a40eeb0848d9288751f721f609a49a514d629f1d8de5515a2d70"


def checked_failed_attempt(payload: dict[str, bytes], outputs: dict[str, object]) -> dict[str, object]:
    """Bind the captured RNG guard failure and unchanged training-state stat."""
    raw = payload["FAILED_VERIFIER_CAPTURE.raw.json"]
    require(digest(raw) == FAILED_RAW_SHA, "failed verifier capture digest differs")
    capture = json_bytes(raw, "failed verifier capture")
    failed_unit = capture.get("unit")
    require(
        capture.get("invocation_id") == FAILED_INVOCATION
        and capture.get("terminal_success") is True
        and isinstance(failed_unit, dict)
        and (
            failed_unit.get("InvocationID"),
            failed_unit.get("ActiveState"),
            failed_unit.get("SubState"),
            failed_unit.get("MainPID"),
            failed_unit.get("ExecMainCode"),
            failed_unit.get("ExecMainStatus"),
            failed_unit.get("Result"),
        )
        == (FAILED_INVOCATION, "failed", "failed", "0", "1", "1", "exit-code"),
        "captured prior verifier identity/status differs",
    )
    failed_admission_raw = payload["FAILED_VERIFIER_ADMISSION.json"]
    failed_launch_raw = payload["FAILED_VERIFIER_LAUNCH.json"]
    failed_log = payload["FAILED_VERIFIER_UNIT_LOG.raw"]
    require(digest(failed_admission_raw) == FAILED_ADMISSION_SHA, "failed admission digest differs")
    require(digest(failed_launch_raw) == FAILED_LAUNCH_SHA, "failed launch digest differs")
    require(digest(failed_log) == FAILED_LOG_SHA, "failed unit log digest differs")
    admission = json_bytes(failed_admission_raw, "failed verifier admission")
    launch = json_bytes(failed_launch_raw, "failed verifier launch")
    require(
        (
            admission.get("schema"),
            admission.get("run_name"),
            admission.get("unit"),
            admission.get("invocation_id"),
        )
        == (
            "fg19.parent-fit-r2-terminal-verifier-admission.v1",
            "fg19_parent_h256x2_f0_s1001_fit_r2_v1",
            FAILED_UNIT,
            "5186b3016ff54b1fa754c10a1f87b3e4",
        ),
        "failed verifier admission identity differs",
    )
    require(
        (
            launch.get("schema"),
            launch.get("unit"),
            launch.get("invocation_id"),
            launch.get("submission", {}).get("post", {}).get("InvocationID"),
        )
        == (
            "fg19.parent-fit-r2-terminal-verifier-launch.v1",
            FAILED_UNIT,
            FAILED_INVOCATION,
            FAILED_INVOCATION,
        ),
        "failed verifier launch identity differs",
    )
    require(
        admission.get("terminal_files_stat_only") == outputs,
        "training outputs changed since the failed verifier's stat-only capture",
    )
    state_stat = json_bytes(payload["RNG_STATE_STAT.json"], "training-state stat pin")
    require(
        state_stat == outputs.get("files", {}).get("training_state.pt"),
        "training-state stat pin differs from fresh output inventory",
    )
    require(
        b"single visible CUDA RNG state is malformed" in failed_log,
        "captured failure is not the expected RNG guard diagnostic",
    )
    return {
        "prior_verifier_invocation_id": FAILED_INVOCATION,
        "prior_verifier_terminal_status": "failed",
        "prior_verifier_capture_sha256": FAILED_RAW_SHA,
        "training_state_stat_only": state_stat,
        "scope": "Failure metadata and training-state stat only; no prior unit query or payload read.",
    }
'''


def _diagnostic_main() -> str:
    """Return a single-gated, create-only diagnostic submission entrypoint."""
    return r'''
def main() -> None:
    """Recheck all parent/cache/source gates, then launch one bounded RNG diagnostic."""
    output = D
    require(B.is_dir() and not B.is_symlink() and B.resolve(strict=True) == B, "project root unavailable")
    physical(D.parent)
    require(D.parent == B / "research_tmp" and not os.path.lexists(D), "new diagnostic output exists or is redirected")
    require(not os.path.lexists(B / "runtime/STOP_ADMISSION.json"), "global stop-admission marker exists")
    require(show(UNIT, ("LoadState",)).get("LoadState") == "not-found", "new diagnostic unit already exists")
    physical(SOURCE)
    payload = payload_bytes()
    expected_names = {
        "parent_fit_terminal_check_v1.py",
        "teacher_s5_terminal_check_v1.py",
        "propensity_terminal_support.py",
        "SOURCE_MANIFEST.json",
        "ADMISSION.json",
        "LAUNCH.json",
        "LAUNCH_ACCEPTANCE.json",
        "TERMINAL_EVIDENCE.json",
        "NATIVE_STDOUT.raw.json",
        "CACHE_ACCEPTANCE.json",
        "CACHE_EXACT_STOP.json",
        "FAILED_VERIFIER_CAPTURE.raw.json",
        "FAILED_VERIFIER_ADMISSION.json",
        "FAILED_VERIFIER_LAUNCH.json",
        "FAILED_VERIFIER_UNIT_LOG.raw",
        "RNG_STATE_STAT.json",
        "RNG_DIAGNOSTIC.py",
    }
    require(set(payload) == expected_names, "fixed diagnostic input set differs")
    manifest = json_bytes(payload["SOURCE_MANIFEST.json"], "source manifest")
    admission = json_bytes(payload["ADMISSION.json"], "parent admission")
    launch = json_bytes(payload["LAUNCH.json"], "parent launch")
    acceptance = json_bytes(payload["LAUNCH_ACCEPTANCE.json"], "parent launch acceptance")
    require(
        (
            admission.get("run_name"),
            admission.get("unit"),
            admission.get("source_commit"),
            admission.get("source_files_verified"),
            admission.get("source_manifest_sha256"),
        )
        == ("fg19_parent_h256x2_f0_s1001_fit_r2_v1", NATIVE_UNIT, COMMIT, 167, MANIFEST_SHA),
        "parent admission identity differs",
    )
    require(launch.get("unit") == NATIVE_UNIT and acceptance.get("invocation_id") == INVOCATION, "parent launch identity differs")
    require(acceptance.get("status") == "ACTUAL_LAUNCHED_FULL_FIT_PENDING", "parent fit is outside the pre-acceptance diagnostic stage")
    require(
        acceptance.get("admission_sha256") == digest(payload["ADMISSION.json"])
        and acceptance.get("launch_sha256") == digest(payload["LAUNCH.json"]),
        "parent acceptance lineage differs",
    )
    require(
        (manifest.get("source_commit"), len(manifest.get("files", []))) == (COMMIT, 167),
        "source manifest identity differs",
    )
    source = checked_source(manifest, admission)
    split = checked_split(admission)
    data = checked_data(admission)
    outputs = checked_run_artifacts()
    cache = checked_cache(admission, split, payload)
    native, terminal = checked_terminal(payload, manifest, admission)
    parent = checked_parent()
    prior_failure = checked_failed_attempt(payload, outputs)

    output.mkdir(mode=0o700, exist_ok=False)
    staged = {}
    for name, raw in payload.items():
        write_exclusive(output / name, raw)
        staged[name] = {"bytes": len(raw), "sha256": digest(raw)}
    diagnostic_admission = {
        "schema": "fg19.parent-fit-r2-rng-diagnostic-admission.v1",
        "status": "DIAGNOSTIC_ONLY_NO_ACCEPTANCE",
        "created_utc": utc_now(),
        "run_name": RUN.name,
        "unit": UNIT,
        "native_unit": NATIVE_UNIT,
        "native_invocation_id": INVOCATION,
        "source": source,
        "split_stat_only": split,
        "registered_data_stat_only": data,
        "run_artifacts_stat_only": outputs,
        "cache_stat_only": cache,
        "native_terminal": terminal,
        "native_unit_snapshot": native,
        "parent_resources": parent,
        "prior_failed_verifier": prior_failure,
        "staged_metadata_and_source": staged,
        "resource_limits": {
            "cpu_quota": "50%",
            "memory_high_bytes": 1400000000,
            "memory_max_bytes": 1500000000,
            "swap_max_bytes": 0,
            "runtime_seconds": 600,
            "cuda_visible_devices": "",
            "blas_threads": 1,
        },
        "scope": "Restricted RNG metadata diagnostic only; no checkpoint/model or data-array values are reported and no acceptance is claimed.",
    }
    write_json_exclusive(output / "RNG_DIAGNOSTIC_ADMISSION.json", diagnostic_admission)

    python = "/home/grinchenkoeg/WunderFunder/.venv/bin/python"
    command = [
        "systemd-run",
        "--user",
        "--unit=" + UNIT,
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        "--property=Slice=wnn-campaign.slice",
        "--property=CPUQuota=50%",
        "--property=MemoryHigh=1400000000",
        "--property=MemoryMax=1500000000",
        "--property=MemorySwapMax=0",
        "--property=RuntimeMaxSec=600",
        "--property=TimeoutStopSec=45",
        "--property=StandardOutput=append:" + str(output / "unit.log"),
        "--property=StandardError=append:" + str(output / "unit.log"),
        "--working-directory=" + str(output),
    ]
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TMPDIR": str(B / "research_tmp"),
        "PYTHONPATH": str(output) + ":" + str(SOURCE / "src"),
        "WNN_PROJECT_ROOT": str(B / "data"),
    }
    command.extend("--setenv=" + key + "=" + value for key, value in environment.items())
    command.extend(
        [
            python,
            "-u",
            str(output / "RNG_DIAGNOSTIC.py"),
            "--run-dir",
            str(RUN),
            "--state-stat",
            str(output / "RNG_STATE_STAT.json"),
            "--output",
            str(output / "RNG_DIAGNOSTIC.json"),
        ]
    )
    submission = {"outcome": "unknown", "command": command}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=25, check=False)
        submission.update({"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode == 0:
            try:
                submission["post"] = show(
                    UNIT,
                    ("InvocationID", "ActiveState", "SubState", "MainPID", "ControlGroup", "ExecMainCode", "ExecMainStatus"),
                    timeout=10,
                )
                submission["outcome"] = "submitted"
            except subprocess.TimeoutExpired as error:
                submission["post_show_timeout"] = {"stdout": text_value(error.stdout), "stderr": text_value(error.stderr)}
            except subprocess.CalledProcessError as error:
                submission["post_show_error"] = {
                    "returncode": error.returncode,
                    "stdout": text_value(error.stdout),
                    "stderr": text_value(error.stderr),
                    "command": error.cmd,
                }
            except (OSError, RuntimeError) as error:
                submission["post_show_error"] = {"returncode": None, "stdout": None, "stderr": None, "error": repr(error)}
    except subprocess.TimeoutExpired as error:
        submission.update({"stdout": text_value(error.stdout), "stderr": text_value(error.stderr), "submit_timeout": True})
    except OSError as error:
        submission.update({"os_error": repr(error), "submit_failed_before_confirmation": True})
    launch_record = {
        "schema": "fg19.parent-fit-r2-rng-diagnostic-launch.v1",
        "created_utc": utc_now(),
        "unit": UNIT,
        "invocation_id": submission.get("post", {}).get("InvocationID") if isinstance(submission.get("post"), dict) else None,
        "submission": submission,
        "unknown_outcome_policy": "Never retry; inspect this exact unit and durable files before any further action.",
    }
    write_json_exclusive(output / "RNG_DIAGNOSTIC_LAUNCH.json", launch_record)
    print(json.dumps({"unit": UNIT, "launch_outcome": submission["outcome"], "launch_record": str(output / "RNG_DIAGNOSTIC_LAUNCH.json")}, sort_keys=True))


main()
'''


def _render_template() -> str:
    """Reuse the reviewed gate functions and substitute only this diagnostic identity."""
    raw = OLD_TEMPLATE.read_bytes()
    if _sha256(raw) != OLD_TEMPLATE_SHA:
        raise ValueError("frozen parent verifier gate template digest differs")
    source = raw.decode("utf-8")
    prefix, marker, suffix = source.partition("\ndef main() -> None:\n")
    if marker != "\ndef main() -> None:\n" or not suffix.rstrip().endswith("main()"):
        raise ValueError("frozen template entrypoint shape differs")
    replacements = (
        (
            'D = B / "research_tmp/fg19_parent_fit_r2_verify_v1"',
            'D = B / "research_tmp/fg19_parent_rng_diag_v1"',
        ),
        (
            'UNIT = "computer-vision-homework-fg19-parent-fit-r2-verify-v1.service"',
            'UNIT = "computer-vision-homework-fg19-parent-rng-diag-v1.service"',
        ),
    )
    for before, after in replacements:
        if prefix.count(before) != 1:
            raise ValueError(f"expected one frozen template identity: {before}")
        prefix = prefix.replace(before, after, 1)
    return prefix + "\n" + _failed_attempt_helper() + "\n" + _diagnostic_main()


def _payload_objects(payload: dict[str, bytes]) -> dict[str, object]:
    """Encode local pinned bytes into the standalone remote body."""
    return {
        name: {
            "bytes": len(raw),
            "sha256": _sha256(raw),
            "base64": base64.b64encode(raw).decode("ascii"),
        }
        for name, raw in sorted(payload.items())
    }


def main() -> None:
    """Create new packet files exclusively; never overwrite the failed verifier packet."""
    if BODY.exists() or PREP_RECEIPT.exists():
        raise FileExistsError(
            "rendered diagnostic packet already exists; inspect before replacement"
        )
    payload = _load_payload()
    if TEMPLATE.exists():
        template = TEMPLATE.read_text(encoding="utf-8")
        if _sha256(template.encode()) != RENDERED_TEMPLATE_SHA:
            raise ValueError("frozen rendered diagnostic template digest differs")
    else:
        template = _render_template()
    rng_script = payload["RNG_DIAGNOSTIC.py"]
    if len(rng_script) > 12_000:
        raise ValueError("RNG diagnostic script exceeds cap")
    compiled = template.replace("PAYLOAD = {}", "PAYLOAD = " + repr(_payload_objects(payload)), 1)
    if "PAYLOAD = {}" in compiled:
        raise ValueError("template payload slot was not replaced")
    compile(compiled, "REMOTE_SUBMISSION.py", "exec")
    if not TEMPLATE.exists():
        with TEMPLATE.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(template)
    body_path = OUTPUT / "REMOTE_SUBMISSION.py"
    with body_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(compiled)
    inputs = {name: _sha256(raw) for name, raw in sorted(payload.items())}
    receipt = {
        "schema": "fg19.parent-fit-r2-rng-diagnostic-preparation.v1",
        "status": "PREPARED_LOCALLY_NOT_TRANSMITTED_OR_LAUNCHED",
        "run_name": RUN,
        "native_invocation_id": "5186b3016ff54b1fa754c10a1f87b3e4",
        "failed_verifier_invocation_id": FAILED_INVOCATION,
        "failed_verifier_capture_sha256": FAILED_RAW_SHA,
        "source_commit": "68ff682977b822f38351aac0f621146f5f13b440",
        "source_manifest_sha256": "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621",
        "frozen_gate_template_sha256": OLD_TEMPLATE_SHA,
        "generator_sha256": _sha256(Path(__file__).read_bytes()),
        "diagnostic_script_sha256": _sha256(rng_script),
        "template_sha256": _sha256(template.encode()),
        "remote_body_sha256": _sha256(compiled.encode()),
        "remote_body_bytes": len(compiled.encode()),
        "remote_unit": NEW_UNIT,
        "remote_output": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_parent_rng_diag_v1",
        "input_sha256": inputs,
        "training_state_stat_only": _object(payload["RNG_STATE_STAT.json"], "state stat"),
        "resource_limits": {
            "cpu_quota": "50%",
            "memory_high_bytes": 1400000000,
            "memory_max_bytes": 1500000000,
            "swap_max_bytes": 0,
            "runtime_seconds": 600,
            "cuda_visible_devices": "",
            "blas_threads": 1,
        },
        "scope": "Only format/next_epoch, installed torch version and RNG structure/tensor-byte hashes are emitted. No model/optimizer values, dataset/cache arrays, quality score, or acceptance claim.",
        "remote_action_by_preparation": False,
    }
    (OUTPUT / "PREPARATION.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "body_sha256": receipt["remote_body_sha256"],
                "template_sha256": receipt["template_sha256"],
                "output": str(OUTPUT),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
