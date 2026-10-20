"""Render and locally pin the one-shot CONTROL terminal verifier admission body."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import stat
from datetime import UTC, datetime
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[2]
REPOSITORY = Path(__file__).resolve().parents[5]
PREPARATION = CAMPAIGN / "preparation"
RECORDS = CAMPAIGN / "records"
OUTPUT = Path(__file__).resolve().parent
TEMPLATE = OUTPUT / "REMOTE_SUBMISSION_TEMPLATE.py"
BODY = OUTPUT / "REMOTE_SUBMISSION.py"
ROOT_PREPARATION = PREPARATION / "CHILD_CONTROL_TERMINAL_ROOT_PREPARATION_V1.json"
TERMINAL_DIR = RECORDS / "child_control_fit_v1/terminal_native_capture_v1"
INPUTS = {
    "child_control_terminal_check_v3.py": OUTPUT / "child_control_terminal_check_v3.py",
    "parent_fit_terminal_check_v3.py": PREPARATION / "parent_fit_terminal_check_v3.py",
    "propensity_terminal_support.py": PREPARATION / "propensity_terminal_support.py",
    "teacher_s5_terminal_check_v1.py": PREPARATION / "teacher_s5_terminal_check_v1.py",
    "SOURCE_MANIFEST.json": PREPARATION / "teacher_cache_cpu_repair_v2/SOURCE_MANIFEST.json",
    "ADMISSION.json": RECORDS / "child_control_fit_v1/ADMISSION.json",
    "LAUNCH.json": RECORDS / "child_control_fit_v1/LAUNCH.json",
    "LAUNCH_ACCEPTANCE.json": RECORDS / "child_control_fit_v1/LAUNCH_ACCEPTANCE.json",
    "TERMINAL_EVIDENCE.json": RECORDS / "child_control_fit_v1/TERMINAL_EVIDENCE.json",
    "NATIVE_STDOUT.raw.json": TERMINAL_DIR / "STDOUT.raw.json",
    "CACHE_ACCEPTANCE.json": RECORDS / "teacher_cache_cpu_build_v2/TERMINAL_ACCEPTANCE.json",
    "CACHE_EXACT_STOP.json": RECORDS / "teacher_cache_cpu_build_v2/exact_stop/STDOUT.raw.json",
}
EXPECTED_SHA256 = {
    "child_control_terminal_check_v3.py": "618b9b0fe7c0cf85e94848b24a837a63e27df46f81e35f2f8536c049072e5167",
    "parent_fit_terminal_check_v3.py": "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41",
    "propensity_terminal_support.py": "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409",
    "teacher_s5_terminal_check_v1.py": "cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5",
    "SOURCE_MANIFEST.json": "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621",
    "ADMISSION.json": "f749d0c4ba4925506320f1a480d7e76164a2b0f84b7efac5b432cb18522c5973",
    "LAUNCH.json": "afd19428c8b6034a9dd40fac78d6adb78b70f3d95f1f964917cbadc828ed27e9",
    "LAUNCH_ACCEPTANCE.json": "caa7e1560171ee6813a6766649b82dc512432c240b400a286a11cea39ad29a15",
    "TERMINAL_EVIDENCE.json": "fefa2e85469495b89e5c9a502f4bd0b9f0b2bcd7cb8928364b46f6a9b2b7a70a",
    "NATIVE_STDOUT.raw.json": "33133c140cf9c8676c87f518230c7aa48ac2485ecdf3dafc5e8c45edf16c0f43",
    "CACHE_ACCEPTANCE.json": "88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7",
    "CACHE_EXACT_STOP.json": "286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4",
}
MAX_BYTES = {
    "child_control_terminal_check_v3.py": 80_000,
    "parent_fit_terminal_check_v3.py": 70_000,
    "propensity_terminal_support.py": 15_000,
    "teacher_s5_terminal_check_v1.py": 40_000,
    "SOURCE_MANIFEST.json": 30_000,
    "ADMISSION.json": 10_000,
    "LAUNCH.json": 10_000,
    "LAUNCH_ACCEPTANCE.json": 4_000,
    "TERMINAL_EVIDENCE.json": 2_000,
    "NATIVE_STDOUT.raw.json": 75_000,
    "CACHE_ACCEPTANCE.json": 4_000,
    "CACHE_EXACT_STOP.json": 2_000,
}
READONLY_RECEIPTS = {
    "PARENT_TERMINAL_ACCEPTANCE.json": (
        RECORDS / "parent_fit_r2_v1/TERMINAL_ACCEPTANCE.json",
        "5b9ddc32f871f4dae29334b4f573d1e38efd9f8435cf2d7fde47e47f97e658b8",
    ),
    "PARENT_EXACT_STOP.json": (
        RECORDS / "parent_fit_r2_v1/exact_stop/STDOUT.raw.json",
        "15a1e42a90a0962c9b45b6043a7e1e152057848725391fffcd611862e5c6ffa8",
    ),
    "DICTIONARY_ACCEPTANCE.json": (
        RECORDS / "dictionary_f0_v1/ACCEPTANCE.json",
        "4802815134d09bbbfeb697689a17ac82bcbbd6541e7969d5fec56867ecd3066f",
    ),
    "DICTIONARY_EXACT_STOP.json": (
        RECORDS / "dictionary_f0_v1/exact_stop/STDOUT.raw.json",
        "2eb44a4381de06f28ed3d7438ace320ef3274eae37c89074fc481f58f5d88158",
    ),
    "PROFILE_TERMINAL_ACCEPTANCE.json": (
        RECORDS / "child_allthree_profile_v1/TERMINAL_ACCEPTANCE.json",
        "9235b1670e80de8e18d322ab6e40dac102a7a8d8c028250f7b08262d8692c045",
    ),
    "PROFILE_EXACT_STOP.json": (
        RECORDS / "child_allthree_profile_v1/exact_stop/STDOUT.raw.json",
        "a0437f4da0b2534721909087bab1ddf40e068c4ba6911870472b7374e4bb7dce",
    ),
    "PROFILE_ADMISSION.json": (
        RECORDS / "child_allthree_profile_v1/ADMISSION.json",
        "908d066878b6a5b69549077a0002c6d6f1ed306bc10f5b059fa6ca423d2da2c5",
    ),
    "PROFILE_LAUNCH.json": (
        RECORDS / "child_allthree_profile_v1/LAUNCH.json",
        "df171f46017dbb4b4b7bdfaaeb374db651d6d972eb11e8ffcebe917a4b464d44",
    ),
    "PROFILE_UNIT_LOG": (
        RECORDS / "child_allthree_profile_v1/capture/unit.log",
        "b4480cd8529ecbb0d5ec1cf840f0c9a312b541f9598cc4cd4cfa19084350c289",
    ),
}
REQUIRED_BODY_TOKENS = (
    "fg19_child_control_f0_s1001_fit_v1",
    "computer-vision-homework-fg19-child-control-fit-v1.service",
    "computer-vision-homework-fg19-child-control-verify-v3.service",
    "MemoryHigh=1400000000",
    "MemoryMax=1500000000",
    "MemorySwapMax=0",
    "RuntimeMaxSec=600",
    "--terminal-evidence",
    "--launch-acceptance",
    "--cache-exact-stop",
    "--profile-evidence-root",
    "--run-dir",
    "--source-root",
    "--source-manifest",
    "--split",
    "--evidence-root",
    "--cache-root",
    "--cache-acceptance",
    "--output",
    "CUDA_VISIBLE_DEVICES",
    "WNN_PROJECT_ROOT",
    "admission_base64",
    "launch_base64",
    "/home/grinchenkoeg/WunderFunder/.venv/bin/python",
    "PARENT_CHECKPOINT_SHA",
    "DICTIONARY_SHA",
    "FG19_APPROVED_PATH_COMPAT_EXCEPTION",
    "pathlib._local.PosixPath:exact-dictionary-path-only",
    "source tree inventory differs",
    "accepted parent checkpoint stat differs",
    "accepted dictionary artifact stat differs",
    "not os.path.lexists(D)",
    "empty/unscoped parent cgroup is NA; refusing host-root fallback",
    "never retry",
)


def sha256(raw: bytes) -> str:
    """Return a SHA-256 digest for exact local input bytes."""
    return hashlib.sha256(raw).hexdigest()


def stable_read(path: Path, cap: int) -> bytes:
    """Read one bounded local input without following a symlink."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_size > cap:
        raise ValueError(f"unsafe or oversized local input: {path}")
    raw = path.read_bytes()
    after = path.stat()
    if len(raw) != before.st_size or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"local input changed while reading: {path}")
    return raw


def json_object(raw: bytes, name: str) -> dict[str, object]:
    """Parse bounded metadata only; callers preserve the original bytes."""
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {name}")
    return value


def check_receipts(root_manifest: dict[str, object]) -> dict[str, str]:
    """Verify all existing parent/profile/cache/dictionary receipt pins read-only."""
    pins = {row["path"]: row["sha256"] for row in root_manifest["metadata_and_proof_pins"]}
    observed: dict[str, str] = {}
    for name, (path, expected) in READONLY_RECEIPTS.items():
        raw = stable_read(path, 1_000_000)
        if sha256(raw) != expected:
            raise ValueError(f"read-only predecessor digest differs: {name}")
        if name != "PROFILE_UNIT_LOG":
            relative = str(path.relative_to(REPOSITORY))
            if pins.get(relative) != expected:
                raise ValueError(f"root preparation receipt pin differs: {name}")
        observed[name] = expected
    profile_acceptance = json_object(
        stable_read(READONLY_RECEIPTS["PROFILE_TERMINAL_ACCEPTANCE.json"][0], 12_000),
        "profile acceptance",
    )
    profile_files = profile_acceptance.get("files")
    expected_profile_files = {
        "ADMISSION.json": (7817, READONLY_RECEIPTS["PROFILE_ADMISSION.json"][1]),
        "LAUNCH.json": (9691, READONLY_RECEIPTS["PROFILE_LAUNCH.json"][1]),
        "unit.log": (7624, READONLY_RECEIPTS["PROFILE_UNIT_LOG"][1]),
    }
    if not isinstance(profile_files, dict):
        raise ValueError("profile acceptance has no captured-file map")
    observed_profile_files = {
        name: (profile_files.get(name, {}).get("bytes"), profile_files.get(name, {}).get("sha256"))
        for name in expected_profile_files
    }
    if observed_profile_files != expected_profile_files:
        raise ValueError("profile file pins differ from accepted profile receipt")
    return observed


def validate_inputs(payload: dict[str, bytes], root_raw: bytes) -> dict[str, object]:
    """Join typed terminal evidence, raw capture, launch inputs and ancestry pins."""
    root_manifest = json_object(root_raw, "root preparation manifest")
    identity = root_manifest.get("identity")
    if not isinstance(identity, dict) or (
        identity.get("run_name"),
        identity.get("unit"),
        identity.get("invocation_id"),
        identity.get("source_files"),
        identity.get("source_manifest_sha256"),
    ) != (
        "fg19_child_control_f0_s1001_fit_v1",
        "computer-vision-homework-fg19-child-control-fit-v1.service",
        "e66e0d0703314e4fb5d47dcdbfcaf202",
        167,
        EXPECTED_SHA256["SOURCE_MANIFEST.json"],
    ):
        raise ValueError("root preparation identity differs")
    if root_manifest.get("metadata_and_proof_pins") is None:
        raise ValueError("root preparation receipt pins are missing")
    admission = json_object(payload["ADMISSION.json"], "CONTROL admission")
    launch = json_object(payload["LAUNCH.json"], "CONTROL launch")
    accepted = json_object(payload["LAUNCH_ACCEPTANCE.json"], "CONTROL launch acceptance")
    terminal = json_object(payload["TERMINAL_EVIDENCE.json"], "CONTROL terminal evidence")
    capture = json_object(payload["NATIVE_STDOUT.raw.json"], "native raw capture")
    if (
        admission.get("run_name"),
        admission.get("unit"),
        admission.get("source_commit"),
        admission.get("source_files_verified"),
        admission.get("source_manifest_sha256"),
        admission.get("split_sha256"),
    ) != (
        identity["run_name"],
        identity["unit"],
        "68ff682977b822f38351aac0f621146f5f13b440",
        167,
        EXPECTED_SHA256["SOURCE_MANIFEST.json"],
        "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7",
    ):
        raise ValueError("CONTROL admission source or run identity differs")
    if (launch.get("unit"), launch.get("submission", {}).get("returncode")) != (
        identity["unit"],
        0,
    ):
        raise ValueError("CONTROL launch receipt differs")
    if (
        accepted.get("schema"),
        accepted.get("status"),
        accepted.get("invocation_id"),
        accepted.get("admission_sha256"),
        accepted.get("launch_sha256"),
    ) != (
        "fg19-child-control-fit-launch-acceptance.v1",
        "ACTUAL_LAUNCH_CAPTURED_NOT_COMPLETION",
        identity["invocation_id"],
        EXPECTED_SHA256["ADMISSION.json"],
        EXPECTED_SHA256["LAUNCH.json"],
    ):
        raise ValueError("CONTROL launch acceptance differs")
    if (
        terminal.get("schema"),
        terminal.get("run_name"),
        terminal.get("unit"),
        terminal.get("invocation_id"),
        terminal.get("active_state"),
        terminal.get("sub_state"),
        terminal.get("main_pid"),
        terminal.get("exec_main_code"),
        terminal.get("exec_main_status"),
        terminal.get("raw_bytes"),
        terminal.get("raw_sha256"),
    ) != (
        "fg19.child-control-fit-native-terminal-evidence.v1",
        identity["run_name"],
        identity["unit"],
        identity["invocation_id"],
        "active",
        "exited",
        0,
        1,
        0,
        73_636,
        "33133c140cf9c8676c87f518230c7aa48ac2485ecdf3dafc5e8c45edf16c0f43",
    ):
        raise ValueError("actual typed terminal evidence differs")
    if (
        capture.get("schema"),
        capture.get("expected_run"),
        capture.get("expected_unit"),
        capture.get("expected_invocation_id"),
        capture.get("captured_utc"),
    ) != (
        "fg19-child-control-startup-capture.v2",
        identity["run_name"],
        identity["unit"],
        identity["invocation_id"],
        terminal.get("captured_utc"),
    ):
        raise ValueError("native capture schema or identity differs")
    native = capture.get("unit_properties")
    if not isinstance(native, dict) or tuple(
        native.get(key)
        for key in (
            "InvocationID",
            "ActiveState",
            "SubState",
            "MainPID",
            "ExecMainCode",
            "ExecMainStatus",
            "Result",
        )
    ) != (identity["invocation_id"], "active", "exited", "0", "1", "0", "success"):
        raise ValueError("native capture has no exact successful exit")
    receipts = capture.get("receipts")
    if not isinstance(receipts, dict):
        raise ValueError("captured B receipts are missing")
    for name in ("ADMISSION.json", "LAUNCH.json"):
        record = receipts.get(name)
        if (
            not isinstance(record, dict)
            or base64.b64decode(record.get("base64", ""), validate=True) != payload[name]
        ):
            raise ValueError(f"captured {name} bytes differ")
    cache = admission.get("cache")
    if not isinstance(cache, dict) or (
        cache.get("metadata_sha256"),
        cache.get("acceptance_sha256"),
        cache.get("exact_stop_sha256"),
    ) != (
        "03d9cf1fbd20b8bb3ad7b7d0c4513293fd1d5128e816256e4bf3cdfbfa4b43e3",
        EXPECTED_SHA256["CACHE_ACCEPTANCE.json"],
        EXPECTED_SHA256["CACHE_EXACT_STOP.json"],
    ):
        raise ValueError("CONTROL admitted cache identity differs")
    readonly_receipts = check_receipts(root_manifest)
    return {
        "typed_terminal_sha256": EXPECTED_SHA256["TERMINAL_EVIDENCE.json"],
        "native_capture_sha256": EXPECTED_SHA256["NATIVE_STDOUT.raw.json"],
        "native_capture_bytes": len(payload["NATIVE_STDOUT.raw.json"]),
        "launch_inputs": {
            name: EXPECTED_SHA256[name]
            for name in ("ADMISSION.json", "LAUNCH.json", "LAUNCH_ACCEPTANCE.json")
        },
        "read_only_predecessor_receipts": readonly_receipts,
        "source_identity": {
            "commit": identity["source_commit"],
            "files": identity["source_files"],
            "manifest_sha256": identity["source_manifest_sha256"],
        },
    }


def validate_body(body: str) -> list[str]:
    """Compile and check the narrow admission adapter's identity and gate tokens."""
    compile(body, "REMOTE_SUBMISSION.py", "exec", flags=0, dont_inherit=True)
    module = ast.parse(body)
    main = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_statements = main.body[1:] if ast.get_docstring(main) else main.body
    first = main_statements[0]
    if not (
        isinstance(first, ast.Assign)
        and isinstance(first.value, ast.Call)
        and isinstance(first.value.func, ast.Name)
        and first.value.func.id == "require_path_exception"
    ):
        raise ValueError("exception switch must be the first admission-body action")
    missing = [token for token in REQUIRED_BODY_TOKENS if token not in body]
    if missing:
        raise ValueError(f"rendered body omits required identity or gate tokens: {missing}")
    forbidden = ('["systemctl", "stop"', "torch.load(", "np.load(")
    if any(token in body for token in forbidden):
        raise ValueError("admission body contains a stop or payload loader")
    return [
        "python_compile",
        "exact_run_and_invocation",
        "resource_caps_and_cpu_only",
        "explicit_allowlist_exception_before_any_mutation",
        "required_checker_cli",
        "no_native_stop_or_payload_loader",
        "create_only_and_no_retry",
    ]


def main() -> None:
    """Create immutable rendered body and local preparation receipts once."""
    if (
        BODY.exists()
        or (OUTPUT / "PREPARATION.json").exists()
        or (OUTPUT / "LOCAL_CHECKS.json").exists()
    ):
        raise FileExistsError("verifier packet already rendered; inspect before replacement")
    template = stable_read(TEMPLATE, 200_000).decode("utf-8")
    root_raw = stable_read(ROOT_PREPARATION, 50_000)
    payload_raw: dict[str, bytes] = {}
    for name, path in INPUTS.items():
        raw = stable_read(path, MAX_BYTES[name])
        if sha256(raw) != EXPECTED_SHA256[name]:
            raise ValueError(f"local input digest differs: {name}")
        payload_raw[name] = raw
    input_checks = validate_inputs(payload_raw, root_raw)
    packaged = {
        name: {
            "bytes": len(raw),
            "sha256": EXPECTED_SHA256[name],
            "base64": base64.b64encode(raw).decode("ascii"),
        }
        for name, raw in payload_raw.items()
    }
    if template.count("PAYLOAD = {}") != 1:
        raise ValueError("template payload slot must appear exactly once")
    body = template.replace("PAYLOAD = {}", "PAYLOAD = " + repr(packaged), 1)
    local_checks = validate_body(body)
    body_raw = body.encode("utf-8")
    BODY.write_text(body, encoding="utf-8", newline="\n")
    root_pins = json_object(root_raw, "root preparation manifest").get(
        "metadata_and_proof_pins", []
    )
    prep = {
        "schema": "fg19.child-control-terminal-verifier-preparation.v3",
        "status": "PREPARED_LOCALLY_NOT_TRANSMITTED_OR_STAGED",
        "utc": datetime.now(UTC).isoformat(),
        "run_name": "fg19_child_control_f0_s1001_fit_v1",
        "native_unit": "computer-vision-homework-fg19-child-control-fit-v1.service",
        "native_invocation_id": "e66e0d0703314e4fb5d47dcdbfcaf202",
        "native_terminal_evidence_sha256": EXPECTED_SHA256["TERMINAL_EVIDENCE.json"],
        "native_raw_capture_sha256": EXPECTED_SHA256["NATIVE_STDOUT.raw.json"],
        "native_raw_capture_bytes": len(payload_raw["NATIVE_STDOUT.raw.json"]),
        "source_commit": "68ff682977b822f38351aac0f621146f5f13b440",
        "source_files": 167,
        "source_manifest_sha256": EXPECTED_SHA256["SOURCE_MANIFEST.json"],
        "root_preparation_sha256_at_render": sha256(root_raw),
        "root_preparation_terminal_state_was_stale": json_object(root_raw, "root prep")
        .get("execution_status", {})
        .get("actual_control_native_terminal_captured")
        is False,
        "remote_output": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_child_control_verify_v3",
        "support_unit": "computer-vision-homework-fg19-child-control-verify-v3.service",
        "proposed_path_compatibility_exception": {
            "environment_variable": "FG19_APPROVED_PATH_COMPAT_EXCEPTION",
            "required_value": "pathlib._local.PosixPath:exact-dictionary-path-only",
            "status": "EXPLICIT_HUMAN_EXCEPTION_REQUIRED; absent switch fails before B-side staging",
            "scope": "One inert tuple-mapped pathlib._local.PosixPath constructor, only for the exact source-qualified dictionary-cache Path in training_state.pt; weights_only=True remains enabled.",
        },
        "remote_body_sha256": sha256(body_raw),
        "remote_body_bytes": len(body_raw),
        "template_sha256": sha256(template.encode()),
        "generator_sha256": sha256(Path(__file__).read_bytes()),
        "embedded_inputs": {
            name: {"bytes": len(raw), "sha256": EXPECTED_SHA256[name]}
            for name, raw in sorted(payload_raw.items())
        },
        "read_only_receipt_pin_count_in_root_preparation": len(root_pins),
        "input_checks": input_checks,
        "resource_limits": {
            "cpu_quota": "50%",
            "memory_high_bytes": 1_400_000_000,
            "memory_max_bytes": 1_500_000_000,
            "swap_max_bytes": 0,
            "runtime_seconds": 600,
            "cuda_hidden": True,
            "blas_threads": 1,
        },
        "remote_gate_outline": [
            "explicit exact-path PosixPath compatibility exception switch before any B-side staging or support submission",
            "fresh exact native active/exited MainPID0 ExecMainCode1 Status0",
            "source manifest and all167 source file hashes",
            "split, registered paths, accepted checkpoint/dictionary and cache stat gates",
            "native/parent/profile/cache/dictionary receipt hash gates",
            "empty campaign descendants, two CPU cores, high18GB max20GB swap0, single support reservation and17GB target",
            "25GiB physical scratch floor",
            "exclusive staging and one support submission",
        ],
        "local_checks": local_checks,
        "remote_action_by_preparation": False,
    }
    prep_raw = (json.dumps(prep, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    (OUTPUT / "PREPARATION.json").write_bytes(prep_raw)
    checks = {
        "schema": "fg19.child-control-terminal-verifier-local-checks.v3",
        "status": "PASS_LOCAL_STATIC_AND_METADATA_CHECKS_ONLY",
        "remote_body_sha256": sha256(body_raw),
        "remote_body_bytes": len(body_raw),
        "checks": local_checks,
        "metadata_inputs_hashed": len(payload_raw),
        "model_state_cache_array_dictionary_payloads_read_or_hashed": False,
        "source_tree_read_or_hashed_locally": False,
        "remote_action": False,
    }
    (OUTPUT / "LOCAL_CHECKS.json").write_text(
        json.dumps(checks, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "body_sha256": sha256(body_raw),
                "body_bytes": len(body_raw),
                "preparation": str(OUTPUT / "PREPARATION.json"),
                "local_checks": str(OUTPUT / "LOCAL_CHECKS.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
