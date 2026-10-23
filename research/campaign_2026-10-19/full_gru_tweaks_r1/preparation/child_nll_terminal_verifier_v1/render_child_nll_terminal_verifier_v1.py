"""Render only after the exact native exit proof and final review bindings exist."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import stat
from datetime import datetime
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[2]
REPOSITORY = Path(__file__).resolve().parents[5]
OUTPUT = Path(__file__).resolve().parent
TEMPLATE = OUTPUT / "REMOTE_SUBMISSION_TEMPLATE.py"
BODY = OUTPUT / "REMOTE_SUBMISSION.py"
PREPARATION = CAMPAIGN / "preparation"
RECORDS = CAMPAIGN / "records"
ROOT_PREPARATION = PREPARATION / "child_nll_terminal_check_v1/ROOT_PREPARATION.json"
TERMINAL = RECORDS / "child_nll_fit_v1/TERMINAL_EVIDENCE.json"
NATIVE_RAW = RECORDS / "child_nll_fit_v1/terminal_native_capture_v1/STDOUT.raw.json"
FIT_ACCEPTANCE = RECORDS / "child_nll_fit_v1/LAUNCH_ACCEPTANCE.json"
SOURCE_MANIFEST = PREPARATION / "teacher_cache_cpu_repair_v2/SOURCE_MANIFEST.json"
CHECKER = PREPARATION / "child_nll_terminal_check_v1/child_nll_terminal_check_v1.py"
NATIVE_SCHEMA = "fg19.child-nll-fit-native-terminal-evidence.v1"
ROOT_SCHEMA = "fg19.child-nll-terminal-root-preparation.v1"
RUN = "fg19_child_nll_f0_s1001_fit_v1"
UNIT = "computer-vision-homework-fg19-child-nll-fit-v1.service"
INVOCATION = "c833c613fb92426981ab255a5faf4452"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
MANIFEST_SHA = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
TEMPLATE_BASE_SHA = "afc61d97ee559338344aa35f32765c826c0d148bd5f9f80c03fa63e988ded10e"
IMPORT_BOUNDARY_PROOF_V2 = {
    "path": (
        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
        "child_nll_terminal_check_v1/IMPORT_BOUNDARY_PROOF_V2.json"
    ),
    "bytes": 11_291,
    "sha256": "2855c3710c1275df71d659392e26b70b617b55aa00243bf3a1f7e7b2b368259e",
}
MAX_NATIVE_RAW_BYTES = 2_000_000
MAX_METADATA_BYTES = 250_000

CODE_INPUTS = {
    "child_nll_terminal_check_v1.py": CHECKER,
    "child_control_terminal_check_v3.py": (
        PREPARATION / "child_control_terminal_verifier_v3/child_control_terminal_check_v3.py"
    ),
    "parent_fit_terminal_check_v3.py": PREPARATION / "parent_fit_terminal_check_v3.py",
    "propensity_terminal_support.py": PREPARATION / "propensity_terminal_support.py",
    "teacher_s5_terminal_check_v1.py": PREPARATION / "teacher_s5_terminal_check_v1.py",
    "child_ema_terminal_check_v3.py": (
        PREPARATION / "child_ema_terminal_verifier_v3/child_ema_terminal_check_v3.py"
    ),
}
FIXED_CODE_PINS = {
    "child_control_terminal_check_v3.py": (
        79_972,
        "618b9b0fe7c0cf85e94848b24a837a63e27df46f81e35f2f8536c049072e5167",
    ),
    "parent_fit_terminal_check_v3.py": (
        64_550,
        "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41",
    ),
    "propensity_terminal_support.py": (
        13_107,
        "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409",
    ),
    "teacher_s5_terminal_check_v1.py": (
        37_448,
        "cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5",
    ),
    "child_ema_terminal_check_v3.py": (
        38_267,
        "676d548c1cd25c3c1379548999cca1300ef135dd41b4a8f73689ef8e59de3b5d",
    ),
}
METADATA_INPUTS = {
    "SOURCE_MANIFEST.json": SOURCE_MANIFEST,
    "LAUNCH_ACCEPTANCE.json": FIT_ACCEPTANCE,
    "NLL_LAUNCH_STDOUT.raw.json": RECORDS / "child_nll_fit_v1/STDOUT.raw.json",
    "NLL_TRANSPORT.json": RECORDS / "child_nll_fit_v1/TRANSPORT.json",
    "TERMINAL_EVIDENCE.json": TERMINAL,
    "NATIVE_STDOUT.raw.json": NATIVE_RAW,
}
EXPECTED_CODE_PATHS = {
    name: str(path.relative_to(CAMPAIGN)).replace("\\", "/") for name, path in CODE_INPUTS.items()
}


def _sha256(raw: bytes) -> str:
    """Return the SHA-256 digest of exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def _read_regular(path: Path, cap: int) -> bytes:
    """Read a bounded regular file without following a final symlink."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_size > cap:
        raise ValueError(f"unsafe or oversized conditional-render input: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ValueError(f"conditional-render input changed before read: {path}")
        chunks: list[bytes] = []
        remaining = cap + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or len(raw) > cap:
        raise ValueError(f"conditional-render input size changed: {path}")
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise ValueError(f"conditional-render input changed while read: {path}")
    return raw


def _json_object(raw: bytes, label: str) -> dict[str, object]:
    """Parse a small JSON object while preserving its separately pinned bytes."""
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {label}")
    return value


def _safe_repo_path(value: object) -> Path:
    """Resolve a repository-relative metadata path without allowing traversal."""
    if not isinstance(value, str):
        raise ValueError("root preparation path must be a string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe repository-relative review path: {value!r}")
    path = REPOSITORY / relative
    if path.resolve(strict=True) != path or path.is_symlink():
        raise ValueError(f"review input is redirected: {path}")
    return path


def _check_code_closure(root: dict[str, object]) -> dict[str, dict[str, object]]:
    """Prehash the exact six code files named by final root preparation."""
    entries = root.get("code_dependencies")
    if not isinstance(entries, list) or len(entries) != 6:
        raise ValueError("final root preparation must bind exactly six code files")
    by_name: dict[str, dict[str, object]] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("malformed final code dependency entry")
        path_text = item.get("path")
        if not isinstance(path_text, str):
            raise ValueError("final code dependency path is missing")
        path = _safe_repo_path(path_text)
        name = path.name
        if name in by_name or name not in CODE_INPUTS:
            raise ValueError(f"unexpected or duplicate private code dependency: {name}")
        if path != CODE_INPUTS[name]:
            raise ValueError(f"private dependency path is not the fixed reviewed path: {name}")
        raw = _read_regular(path, 100_000)
        size, sha = item.get("bytes"), item.get("sha256")
        if type(size) is not int or len(raw) != size or _sha256(raw) != sha:
            raise ValueError(f"final checker/helper pin differs from exact local bytes: {name}")
        if name in FIXED_CODE_PINS and (size, sha) != FIXED_CODE_PINS[name]:
            raise ValueError(f"frozen helper pin differs: {name}")
        by_name[name] = {"path": path_text, "bytes": size, "sha256": sha, "raw": raw}
    if set(by_name) != set(CODE_INPUTS):
        raise ValueError("final private code closure names differ")
    return by_name


def _check_source_review(root: dict[str, object], checker: dict[str, object]) -> dict[str, object]:
    """Require the exact final checker source review and no blocking findings."""
    item = root.get("source_review")
    if not isinstance(item, dict) or item.get("status") != "PASS_SOURCE_ONLY":
        raise ValueError("root preparation has no final source-only PASS")
    path = _safe_repo_path(item.get("path"))
    raw = _read_regular(path, MAX_METADATA_BYTES)
    if type(item.get("bytes")) is not int or len(raw) != item.get("bytes"):
        raise ValueError("final checker source-review byte count differs")
    sha = _sha256(raw)
    if sha != item.get("sha256"):
        raise ValueError("final checker source-review digest differs")
    review = _json_object(raw, "checker source review")
    status = review.get("status")
    if not isinstance(status, str) or not status.startswith("PASS"):
        raise ValueError("checker source review itself does not report PASS")
    blockers = review.get("blocking_findings")
    if blockers not in (None, []):
        raise ValueError("checker source review contains blocking findings")
    checker_name = Path(str(checker["path"])).name
    checker_binding_found = False
    stack: list[object] = [review]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            path_value = value.get("path")
            if (
                isinstance(path_value, str)
                and Path(path_value).name == checker_name
                and (value.get("bytes"), value.get("sha256"))
                == (checker.get("bytes"), checker.get("sha256"))
            ):
                checker_binding_found = True
            keyed_pin = value.get(checker_name)
            if isinstance(keyed_pin, dict) and (
                keyed_pin.get("bytes"),
                keyed_pin.get("sha256"),
            ) == (checker.get("bytes"), checker.get("sha256")):
                checker_binding_found = True
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    if not checker_binding_found:
        raise ValueError("final source review does not bind the exact NLL checker bytes")
    return {"path": item["path"], "bytes": len(raw), "sha256": sha, "status": item["status"]}


def _check_proof_pins(root: dict[str, object]) -> list[dict[str, object]]:
    """Recheck the root-preparation metadata-only support pins, excluding payloads."""
    entries = root.get("metadata_and_proof_pins")
    if not isinstance(entries, list) or not entries:
        raise ValueError("root preparation omits its proof and metadata pins")
    observed: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("malformed metadata/proof pin")
        path_text = item.get("path")
        if not isinstance(path_text, str) or path_text in seen:
            raise ValueError("missing or duplicate metadata/proof pin path")
        seen.add(path_text)
        if Path(path_text).suffix.lower() in {".pt", ".pth", ".npy", ".npz", ".parquet"}:
            raise ValueError("root metadata pin unexpectedly names a model or data payload")
        path = _safe_repo_path(path_text)
        raw = _read_regular(path, MAX_METADATA_BYTES)
        if type(item.get("bytes")) is not int or len(raw) != item.get("bytes"):
            raise ValueError(f"root metadata byte count differs: {path_text}")
        if _sha256(raw) != item.get("sha256"):
            raise ValueError(f"root metadata digest differs: {path_text}")
        observed.append({"path": path_text, "bytes": len(raw), "sha256": _sha256(raw)})
    return observed


def _check_root_preparation(
    root_raw: bytes,
) -> tuple[
    dict[str, object], dict[str, dict[str, object]], dict[str, object], list[dict[str, object]]
]:
    """Require the final checker review and code/proof pins before rendering."""
    root = _json_object(root_raw, "NLL checker root preparation")
    identity = root.get("identity")
    expected_identity = {
        "run_name": RUN,
        "unit": UNIT,
        "invocation_id": INVOCATION,
        "source_commit": COMMIT,
        "source_files": 167,
        "source_manifest_sha256": MANIFEST_SHA,
    }
    if (
        root.get("schema"),
        root.get("status"),
        identity,
        root.get("dependency_binding_condition_satisfied"),
        root.get("native_terminal_evidence_present"),
        root.get("actual_B_staging_admission_or_execution"),
    ) != (
        ROOT_SCHEMA,
        "PASS_LOCAL_PREPARATION_ONLY_NOT_STAGED_ADMITTED_OR_EXECUTED",
        expected_identity,
        True,
        False,
        False,
    ):
        raise ValueError("final NLL checker root-preparation identity/status differs")
    if root.get("accepted_import_boundary_proof") != IMPORT_BOUNDARY_PROOF_V2:
        raise ValueError("root preparation does not select the exact accepted V2 import proof")
    code = _check_code_closure(root)
    review = _check_source_review(root, code["child_nll_terminal_check_v1.py"])
    proof_pins = _check_proof_pins(root)
    if not any(
        (item.get("path"), item.get("bytes"), item.get("sha256"))
        == (
            IMPORT_BOUNDARY_PROOF_V2["path"],
            IMPORT_BOUNDARY_PROOF_V2["bytes"],
            IMPORT_BOUNDARY_PROOF_V2["sha256"],
        )
        for item in proof_pins
    ):
        raise ValueError("accepted V2 import proof is absent from verified metadata pins")
    return root, code, review, proof_pins


def _check_native_proof(terminal_raw: bytes, capture_raw: bytes) -> dict[str, object]:
    """Join actual strict-integer NLL exit evidence to its raw-first capture."""
    terminal = _json_object(terminal_raw, "typed NLL native terminal evidence")
    if (
        terminal.get("schema"),
        terminal.get("run_name"),
        terminal.get("unit"),
        terminal.get("invocation_id"),
        terminal.get("active_state"),
        terminal.get("sub_state"),
    ) != (NATIVE_SCHEMA, RUN, UNIT, INVOCATION, "active", "exited"):
        raise ValueError("typed NLL native terminal identity/state differs")
    for key, expected in (("main_pid", 0), ("exec_main_code", 1), ("exec_main_status", 0)):
        value = terminal.get(key)
        if type(value) is not int or value != expected:
            raise ValueError(f"typed NLL exit field {key} is not the exact integer {expected}")
    if type(terminal.get("raw_bytes")) is not int or terminal["raw_bytes"] != len(capture_raw):
        raise ValueError("typed NLL raw capture byte count differs")
    if terminal["raw_bytes"] <= 0:
        raise ValueError("typed NLL raw capture byte count must be positive")
    raw_sha = _sha256(capture_raw)
    if terminal.get("raw_sha256") != raw_sha or len(capture_raw) > MAX_NATIVE_RAW_BYTES:
        raise ValueError("typed NLL raw capture digest/cap differs")
    if terminal.get("raw_path") != (
        "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_nll_fit_v1/"
        "terminal_native_capture_v1/STDOUT.raw.json"
    ):
        raise ValueError("typed NLL raw capture path differs")
    capture = _json_object(capture_raw, "raw-first NLL native capture")
    if (
        capture.get("schema"),
        capture.get("expected_run"),
        capture.get("expected_unit"),
        capture.get("expected_invocation_id"),
        capture.get("captured_utc"),
    ) != (
        "fg19-child-nll-startup-capture.v1",
        RUN,
        UNIT,
        INVOCATION,
        terminal.get("captured_utc"),
    ):
        raise ValueError("raw-first NLL capture identity differs")
    unit = capture.get("unit_properties")
    if not isinstance(unit, dict) or tuple(
        unit.get(key)
        for key in (
            "InvocationID",
            "ActiveState",
            "SubState",
            "MainPID",
            "ExecMainCode",
            "ExecMainStatus",
            "Result",
        )
    ) != (INVOCATION, "active", "exited", "0", "1", "0", "success"):
        raise ValueError("raw-first NLL capture unit properties differ")
    timestamps: dict[str, datetime] = {}
    for key in ("captured_utc", "native_exit_utc"):
        value = terminal.get(key)
        if not isinstance(value, str):
            raise ValueError(f"typed NLL {key} is missing")
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"typed NLL {key} is malformed") from error
        if stamp.tzinfo is None:
            raise ValueError(f"typed NLL {key} has no timezone")
        timestamps[key] = stamp
    if timestamps["native_exit_utc"] > timestamps["captured_utc"]:
        raise ValueError("typed NLL exit timestamp is after its raw capture")
    return {
        "typed_sha256": _sha256(terminal_raw),
        "typed_bytes": len(terminal_raw),
        "raw_sha256": raw_sha,
        "raw_bytes": len(capture_raw),
        "captured_utc": terminal.get("captured_utc"),
        "native_exit_utc": terminal.get("native_exit_utc"),
    }


def _load_all_inputs(
    root: dict[str, object],
    code: dict[str, dict[str, object]],
    proof: dict[str, object],
) -> tuple[dict[str, bytes], dict[str, object]]:
    """Read the six exact code files and bounded metadata, never fit payloads."""
    payloads: dict[str, bytes] = {
        name: entry["raw"] for name, entry in code.items() if isinstance(entry.get("raw"), bytes)
    }
    for name, path in METADATA_INPUTS.items():
        cap = MAX_NATIVE_RAW_BYTES if name == "NATIVE_STDOUT.raw.json" else MAX_METADATA_BYTES
        raw = _read_regular(path, cap)
        payloads[name] = raw
    fixed = {
        "SOURCE_MANIFEST.json": (27_569, MANIFEST_SHA),
        "LAUNCH_ACCEPTANCE.json": (
            2_931,
            "44a708fc5a32b101c97a8a28c22e3b15e815059582de564e17f528f5369dc7a5",
        ),
        "NLL_LAUNCH_STDOUT.raw.json": (
            27_426,
            "90c7fcd2a916ffd6a1fa63ee9a7192f6cbc439f989bb7ade25435ecd1902efbb",
        ),
        "NLL_TRANSPORT.json": (
            1_213,
            "5864d2ecacb9d8287002e061598560e71c65a73d49054eadc81c3842e101f7a3",
        ),
    }
    for name, (size, sha) in fixed.items():
        if (len(payloads[name]), _sha256(payloads[name])) != (size, sha):
            raise ValueError(f"fixed NLL launch/source metadata differs: {name}")
    if _sha256(payloads["TERMINAL_EVIDENCE.json"]) != proof["typed_sha256"]:
        raise ValueError("typed native evidence changed after proof validation")
    if _sha256(payloads["NATIVE_STDOUT.raw.json"]) != proof["raw_sha256"]:
        raise ValueError("raw native capture changed after proof validation")
    return payloads, root


def _check_embedded_body(body: str, payloads: dict[str, bytes]) -> list[str]:
    """Parse the candidate and bind every literal payload without executing it."""
    tree = ast.parse(body)
    if not tree.body or not (
        isinstance(tree.body[-1], ast.If)
        and isinstance(tree.body[-1].test, ast.Compare)
        and isinstance(tree.body[-1].test.left, ast.Name)
        and tree.body[-1].test.left.id == "__name__"
    ):
        raise ValueError("conditional body must have an inert main guard")
    if "Path(__file__)" in body or "stable_file(__file__" in body:
        raise ValueError("stdin body cannot self-read or claim a worker-side body hash")
    payload_node = next(
        (
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "PAYLOAD" for target in node.targets
            )
        ),
        None,
    )
    if not isinstance(payload_node, ast.Dict):
        raise ValueError("rendered body lacks a literal fixed PAYLOAD map")
    packaged = ast.literal_eval(payload_node)
    if set(packaged) != set(payloads):
        raise ValueError("rendered body payload names differ from the bounded inputs")
    for name, raw in payloads.items():
        item = packaged[name]
        decoded = base64.b64decode(item["base64"], validate=True)
        if (len(decoded), _sha256(decoded), decoded) != (len(raw), _sha256(raw), raw):
            raise ValueError(f"rendered body payload differs from exact local bytes: {name}")
    checker_text = payloads["child_nll_terminal_check_v1.py"].decode("utf-8")
    source_cli_call = checker_text.find("launch_args = _launch_cli_args(launch, source_root)")
    source_module_import = checker_text.find(
        'representation_module = importlib.import_module("wnn.training.input_representation")'
    )
    if source_cli_call < 0 or source_module_import <= source_cli_call:
        raise ValueError("pinned checker does not insert and validate source root before imports")
    markers = (
        'result.stdout.strip() == "ext4"',
        "project_device == scratch_device == 64_512",
        '"child_nll_terminal_check_v1.py"',
        '"DICTIONARY_CHILD_TERMINAL_ACCEPTANCE.json"',
        '"DICTIONARY_CHILD_EXACT_STOP.json"',
        '"CONTROL_TERMINAL_ACCEPTANCE.json"',
        '"EMA_TERMINAL_ACCEPTANCE.json"',
        '"PARENT_TERMINAL_ACCEPTANCE.json"',
        '"PROFILE_TERMINAL_ACCEPTANCE.json"',
        '"worker_body_sha256_claimed": False',
        '"worker_self_read": False',
        '"-E"',
    )
    if not all(marker in body for marker in markers):
        raise ValueError("rendered body is missing an NLL fresh gate or raw-transport boundary")
    return [
        "rendered body parses without production execution",
        "six exact checker/helper files and bounded launch/native metadata are embedded",
        "all twelve accepted predecessor receipts remain exact/read-only dependencies",
        "native proof has strict identity, exit types, raw join, and current-unit recheck",
        "canonical source root is inserted and origin-checked before representation imports",
        "the canonical worker interpreter runs with environment isolation",
        "support caps, physical scratch, and no-self-read stdin authority are present",
    ]


def render_body() -> tuple[bytes, dict[str, object]]:
    """Require actual native proof and final reviews, then render one body create-only."""
    if BODY.exists() or BODY.is_symlink():
        raise FileExistsError("conditional renderer output already exists; never overwrite")
    root_raw = _read_regular(ROOT_PREPARATION, MAX_METADATA_BYTES)
    root, code, source_review, proof_pins = _check_root_preparation(root_raw)
    terminal_raw = _read_regular(TERMINAL, 16_000)
    capture_raw = _read_regular(NATIVE_RAW, MAX_NATIVE_RAW_BYTES)
    proof = _check_native_proof(terminal_raw, capture_raw)
    payloads, _ = _load_all_inputs(root, code, proof)
    template_raw = _read_regular(TEMPLATE, 100_000)
    if _sha256(template_raw) != TEMPLATE_BASE_SHA:
        raise ValueError("conditional NLL template differs from the exact reviewed template pin")
    template = template_raw.decode("utf-8")
    if template.count("PAYLOAD = {}") != 1 or template.count("FINAL_REVIEW_BINDING = {}") != 1:
        raise ValueError("template must have one empty payload and binding slot")
    substitutions = {
        'TERMINAL_SHA = "__RENDER_TERMINAL_SHA256__"': f'TERMINAL_SHA = "{proof["typed_sha256"]}"',
        'NATIVE_RAW_SHA = "__RENDER_NATIVE_RAW_SHA256__"': f'NATIVE_RAW_SHA = "{proof["raw_sha256"]}"',
        "NATIVE_RAW_BYTES = 0": f"NATIVE_RAW_BYTES = {proof['raw_bytes']}",
        'CHECKER_SHA = "__RENDER_CHECKER_SHA256__"': f'CHECKER_SHA = "{code["child_nll_terminal_check_v1.py"]["sha256"]}"',
        "CHECKER_BYTES = 0": f"CHECKER_BYTES = {code['child_nll_terminal_check_v1.py']['bytes']}",
    }
    body = template
    for old, new in substitutions.items():
        if body.count(old) != 1:
            raise ValueError(f"template placeholder is missing or duplicated: {old}")
        body = body.replace(old, new, 1)
    packaged = {
        name: {
            "bytes": len(raw),
            "sha256": _sha256(raw),
            "base64": base64.b64encode(raw).decode("ascii"),
        }
        for name, raw in sorted(payloads.items())
    }
    body = body.replace("PAYLOAD = {}", "PAYLOAD = " + repr(packaged), 1)
    render_binding = {
        "schema": "fg19.child-nll-terminal-verifier-render-binding.v1",
        "status": "ROOT_PREPARATION_AND_SOURCE_REVIEW_AND_NATIVE_EXIT0_BOUND",
        "checker": {
            "path": code["child_nll_terminal_check_v1.py"]["path"],
            "bytes": code["child_nll_terminal_check_v1.py"]["bytes"],
            "sha256": code["child_nll_terminal_check_v1.py"]["sha256"],
        },
        "source_review": source_review,
        "root_preparation": {
            "path": str(ROOT_PREPARATION.relative_to(REPOSITORY)),
            "bytes": len(root_raw),
            "sha256": _sha256(root_raw),
            "status": root.get("status"),
        },
        "native_terminal": proof,
        "body_template": {
            "path": str(TEMPLATE.relative_to(REPOSITORY)),
            "bytes": len(template_raw),
            "sha256": _sha256(template_raw),
        },
        "root_metadata_proof_pin_count": len(proof_pins),
    }
    binding_assignment = "FINAL_REVIEW_BINDING = {}"
    if body.count(binding_assignment) != 1:
        raise ValueError("final review binding slot is missing or duplicated")
    body = body.replace(binding_assignment, "FINAL_REVIEW_BINDING = " + repr(render_binding), 1)
    check_names = _check_embedded_body(body, payloads)
    body_raw = body.encode("utf-8")
    ast.parse(body_raw.decode("utf-8"))
    report = {
        "status": "RENDERED_CREATE_ONLY_LOCAL_BODY_NOT_STAGED_OR_ADMITTED",
        "body_bytes": len(body_raw),
        "body_sha256": _sha256(body_raw),
        "payload_count": len(payloads),
        "code_count": len(code),
        "native_proof": proof,
        "root_preparation_sha256": _sha256(root_raw),
        "source_review": source_review,
        "proof_pin_count": len(proof_pins),
        "checks": check_names,
        "body_executed_or_imported": False,
        "remote_or_B_action": False,
    }
    descriptor = os.open(BODY, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(body_raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(OUTPUT, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return body_raw, report


def main() -> int:
    """Report readiness or render only when explicitly requested and fully bound."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render", action="store_true", help="emit the body after every exact proof gate passes"
    )
    args = parser.parse_args()
    if not args.render:
        pending = [
            str(path) for path in (ROOT_PREPARATION, TERMINAL, NATIVE_RAW) if not path.exists()
        ]
        report = {
            "status": "WAITING_FOR_FINAL_ROOT_PREPARATION_AND_NATIVE_EXIT0_PROOF"
            if pending
            else "READY_FOR_EXPLICIT_RENDER_CHECK",
            "missing_inputs": pending,
            "body_rendered": False,
            "remote_or_B_action": False,
        }
        print(json.dumps(report, sort_keys=True))
        return 0
    body_raw, report = render_body()
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
