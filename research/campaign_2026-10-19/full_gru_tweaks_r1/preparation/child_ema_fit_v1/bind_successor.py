"""Bind the EMA successor body after final CONTROL receipts exist locally."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import pprint
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
REPOSITORY = HERE.parents[4]
SOURCE_CHECKOUT = Path("/Users/nb/Documents/wunder/WunderFunder-strong-gru-tweaks-20261019")
CONTROL_BODY = CAMPAIGN / "preparation/child_control_fit_v1/REMOTE_SUBMISSION.py"
CONTROL_RECIPE = CAMPAIGN / "preparation/child_control_fit_v1/CONTROL_RECIPE_BINDING_V1.json"
EMA_RECIPE = HERE / "EMA_RECIPE_BINDING_V1.json"
OUTPUT_BODY = HERE / "REMOTE_SUBMISSION.py"
EXPECTED_CONTROL_BODY_SHA256 = "124b90f9cc76c17c9a62e22b934dc15e579417f0b769eb86f44334ba55f6b954"
EXPECTED_CONTROL_RECIPE_SHA256 = "c00c8f5e478326ee72c3651aec0c23bf911344e8984009ad3953abcd5d3057e8"
EXPECTED_SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
EXPECTED_SOURCE_MANIFEST_SHA256 = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
EXPECTED_CONTROL_RUN = "fg19_child_control_f0_s1001_fit_v1"
EXPECTED_CONTROL_UNIT = "computer-vision-homework-fg19-child-control-fit-v1.service"
EXPECTED_EMA_RUN = "fg19_child_ema_f0_s1001_fit_v1"
EXPECTED_EMA_UNIT = "computer-vision-homework-fg19-child-ema-fit-v1.service"
CONTROL_ACCEPTANCE_SCHEMA = "fg19-child-control-fit-terminal-acceptance.v1"


def _sha256(raw: bytes) -> str:
    """Return a lowercase SHA-256 digest for exact file bytes."""
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    """Load one local JSON metadata document and require an object root."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _literal_assignment(source: str, name: str) -> tuple[ast.AST, object]:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return node, ast.literal_eval(node.value)
    raise ValueError(f"missing literal assignment {name}")


def _successor_argv(control_argv: list[str]) -> list[str]:
    """Change the two permitted argv values: run identity and representation mode."""
    if not control_argv or control_argv[0] != "--name":
        raise ValueError("CONTROL argv does not begin with its required run identity")
    argv = list(control_argv)
    name_index = argv.index("--name")
    mode_index = argv.index("--representation-mode")
    if argv[name_index + 1] != EXPECTED_CONTROL_RUN or argv[mode_index + 1] != "base":
        raise ValueError("CONTROL argv identity or base mode differs from the frozen recipe")
    argv[name_index + 1] = EXPECTED_EMA_RUN
    argv[mode_index + 1] = "ema"
    return argv


def _assert_argv_delta(control: list[str], successor: list[str]) -> None:
    changed = [
        (index, left, right)
        for index, (left, right) in enumerate(zip(control, successor, strict=True))
        if left != right
    ]
    expected = [
        (1, EXPECTED_CONTROL_RUN, EXPECTED_EMA_RUN),
        (successor.index("--representation-mode") + 1, "base", "ema"),
    ]
    if changed != expected or len(control) != len(successor):
        raise ValueError(
            f"successor argv has changes beyond run and representation identities: {changed!r}"
        )


def _source_component_hashes() -> dict[str, dict[str, object]]:
    recipe = _load_json(EMA_RECIPE)
    semantics = _object(recipe.get("source_feature_semantics"), "source_feature_semantics")
    expected = _object(semantics.get("source_component_bindings"), "source component bindings")
    actual: dict[str, dict[str, object]] = {}
    for relative, item in expected.items():
        if not isinstance(relative, str):
            raise ValueError("source component path must be text")
        pin = _object(item, f"source component pin {relative}")
        path = SOURCE_CHECKOUT / relative
        raw = path.read_bytes()
        digest = _sha256(raw)
        if pin.get("sha256") != digest or pin.get("bytes") != len(raw):
            raise ValueError(f"canonical source component changed: {relative}")
        actual[relative] = {"bytes": len(raw), "sha256": digest}
    return actual


def _canonical_cli_check(argv: list[str]) -> dict[str, object]:
    """Parse the fixed argv with source commit 68ff's canonical parser only."""
    revision = subprocess.run(
        ["git", "-C", str(SOURCE_CHECKOUT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if revision != EXPECTED_SOURCE_COMMIT:
        raise ValueError(f"canonical source checkout revision changed: {revision}")
    sys.path.insert(0, str(SOURCE_CHECKOUT / "src"))
    from wnn.training.cli import build_parser

    parsed = build_parser().parse_args(argv)
    expected = {
        "name": EXPECTED_EMA_RUN,
        "representation_mode": "ema",
        "mean_variance_objective": "wmse",
        "mean_variance_head": 1,
        "loss": "wmse",
        "batch": 16,
        "chunk": 2000,
        "epochs": 6,
        "checkpoint_epoch": 5,
        "seed": 1001,
    }
    actual = {key: getattr(parsed, key, None) for key in expected}
    if actual != expected:
        raise ValueError(f"canonical CLI parsed unexpected EMA recipe values: {actual!r}")
    return {"source_commit": revision, "parsed": actual}


def _source_mode_semantics_check() -> dict[str, object]:
    """Check the pinned source spells the fixed EMA feature layout and recurrence."""
    path = SOURCE_CHECKOUT / "src/wnn/training/input_representation.py"
    source = path.read_text(encoding="utf-8")
    required = (
        "RAW_WIDTH = 112",
        "CODE_WIDTH = 16",
        "SLOT_WIDTH = 128",
        "EXTRA_WIDTH = 2 * SLOT_WIDTH",
        "HALF_LIFE = 32",
        'elif mode == "ema":\n'
        "        output[..., :SLOT_WIDTH].zero_()\n"
        "        causal_signed_ema(\n"
        "            raw_fp32,\n"
        "            output=output[..., SLOT_WIDTH : SLOT_WIDTH + RAW_WIDTH],\n"
        "        )\n"
        "        output[..., SLOT_WIDTH + RAW_WIDTH :].zero_()",
        "    result.zero_()\n"
        "    alpha = torch.tensor(EMA_ALPHA, device=values.device, dtype=torch.float32)",
        "        delta = source[:, start:stop] - source[:, start - 1 : stop - 1]",
    )
    missing = [snippet for snippet in required if snippet not in source]
    if missing:
        raise ValueError(f"pinned EMA source semantics changed: {missing!r}")
    return {
        "source_path": "src/wnn/training/input_representation.py",
        "source_sha256": _sha256(path.read_bytes()),
        "raw_width": 112,
        "code_width": 16,
        "representation_slot_width": 128,
        "appended_width": 256,
        "ema_half_life_rows": 32,
        "mode_ema_layout": {
            "columns_0_128": "exact zero dictionary/sparse slots",
            "columns_128_240": "causal signed EMA of 112 raw input innovations",
            "columns_240_256": "exact zero padding",
        },
        "innovation": "current raw row minus previous raw row; row zero is zero; history resets for each input sequence before recurrent chunking",
        "parameter_ema": False,
    }


def _receipt_counter(receipt: dict[str, object], names: tuple[str, ...]) -> object:
    for candidate in (receipt, receipt.get("fit_counters"), receipt.get("fit")):
        if isinstance(candidate, dict):
            for name in names:
                if name in candidate:
                    return candidate[name]
    return None


def _source_identity(acceptance: dict[str, object]) -> tuple[object, object, object]:
    source = acceptance.get("source")
    nested = source if isinstance(source, dict) else {}
    commit = acceptance.get("source_commit", nested.get("commit"))
    file_count = acceptance.get("source_file_count", nested.get("file_count"))
    manifest = acceptance.get("source_manifest_sha256", nested.get("manifest_sha256"))
    return commit, file_count, manifest


def validate_control_receipts(acceptance_raw: bytes, stop_raw: bytes) -> dict[str, object]:
    """Require one accepted native CONTROL fit and its exact successful stop."""
    if not acceptance_raw or len(acceptance_raw) > 256_000 or len(stop_raw) > 64_000:
        raise ValueError("CONTROL receipt bytes are empty or exceed the bounded metadata size")
    acceptance = _object(json.loads(acceptance_raw), "CONTROL acceptance")
    stop = _object(json.loads(stop_raw), "CONTROL exact stop")
    schema = acceptance.get("schema")
    if schema != CONTROL_ACCEPTANCE_SCHEMA:
        raise ValueError(f"unexpected CONTROL acceptance schema: {schema!r}")
    if (
        acceptance.get("status") != "ACCEPTED"
        or acceptance.get("run_name") != EXPECTED_CONTROL_RUN
        or acceptance.get("unit") != EXPECTED_CONTROL_UNIT
    ):
        raise ValueError(
            "CONTROL acceptance status/run/unit do not identify the frozen predecessor"
        )
    invocation = acceptance.get("invocation_id")
    if not isinstance(invocation, str) or not re.fullmatch(r"[0-9a-f]{32}", invocation):
        raise ValueError("CONTROL acceptance has no exact 32-character invocation identity")
    if _source_identity(acceptance) != (
        EXPECTED_SOURCE_COMMIT,
        167,
        EXPECTED_SOURCE_MANIFEST_SHA256,
    ):
        raise ValueError(
            "CONTROL acceptance source identity differs from the frozen canonical source"
        )
    counts = {
        "successful_updates": _receipt_counter(
            acceptance,
            ("successful_updates", "optimizer_completed_updates", "observed_optimizer_updates"),
        ),
        "completed_batches": _receipt_counter(
            acceptance, ("completed_batches", "observed_batches")
        ),
        "amp_skipped_updates": _receipt_counter(acceptance, ("amp_skipped_updates",)),
        "nonfinite_updates": _receipt_counter(
            acceptance, ("nonfinite_updates", "gradient_stats_nonfinite_steps")
        ),
        "next_epoch": _receipt_counter(acceptance, ("next_epoch",)),
        "selected_epoch": _receipt_counter(acceptance, ("selected_epoch",)),
    }
    expected_counts = {
        "successful_updates": 44_760,
        "completed_batches": 4_476,
        "amp_skipped_updates": 0,
        "nonfinite_updates": 0,
        "next_epoch": 6,
        "selected_epoch": 5,
    }
    if counts != expected_counts:
        raise ValueError(
            f"CONTROL acceptance counters do not match the fixed six-epoch fit: {counts!r}"
        )
    stop_schema = stop.get("schema")
    if stop_schema is not None and not isinstance(stop_schema, str):
        raise ValueError("CONTROL exact-stop schema must be text when present")
    acceptance_sha = _sha256(acceptance_raw)
    if (
        stop.get("unit") != EXPECTED_CONTROL_UNIT
        or stop.get("invocation_id") != invocation
        or stop.get("acceptance_sha256") != acceptance_sha
    ):
        raise ValueError(
            "CONTROL exact stop does not bind the exact accepted unit, invocation, and bytes"
        )
    before = _object(stop.get("before"), "CONTROL exact-stop before state")
    after = _object(stop.get("after"), "CONTROL exact-stop after state")
    if (
        before.get("InvocationID") != invocation
        or before.get("MainPID") not in ("0", 0)
        or before.get("ExecMainCode") not in ("1", 1)
        or before.get("ExecMainStatus") not in ("0", 0)
        or before.get("ActiveState") != "active"
        or before.get("SubState") != "exited"
        or after.get("LoadState") != "not-found"
        or after.get("ActiveState") != "inactive"
        or after.get("SubState") != "dead"
        or after.get("MainPID") not in ("0", 0)
        or after.get("InvocationID") != ""
        or stop.get("populated_before") != {}
        or stop.get("populated_after") != {}
    ):
        raise ValueError("CONTROL exact stop is not the exact successful exited unit stop")
    stop_result = stop.get("stop")
    if isinstance(stop_result, dict) and (
        stop_result.get("returncode") != 0
        or stop_result.get("stdout") != ""
        or stop_result.get("stderr") != ""
    ):
        raise ValueError("CONTROL stop command did not return a clean successful result")
    return {
        "acceptance_sha256": acceptance_sha,
        "exact_stop_sha256": _sha256(stop_raw),
        "run_name": EXPECTED_CONTROL_RUN,
        "unit": EXPECTED_CONTROL_UNIT,
        "invocation_id": invocation,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_file_count": 167,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "successful_updates": 44_760,
        "completed_batches": 4_476,
        "amp_skipped_updates": 0,
        "nonfinite_updates": 0,
        "next_epoch": 6,
        "selected_epoch": 5,
    }


def _read_receipt(path_value: str, expected_name: str) -> bytes:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPOSITORY / path
    resolved = path.resolve(strict=True)
    control_records = (CAMPAIGN / "records/child_control_fit_v1").resolve(strict=True)
    expected_path = control_records / (
        "TERMINAL_ACCEPTANCE.json"
        if expected_name == "TERMINAL_ACCEPTANCE.json"
        else "exact_stop/STDOUT.raw.json"
    )
    if path.is_symlink() or path != resolved or not resolved.is_relative_to(control_records):
        raise ValueError(
            f"receipt must be a physical file beneath the fixed CONTROL records folder: {path}"
        )
    if resolved != expected_path or resolved.name != expected_name or not resolved.is_file():
        raise ValueError(f"unexpected CONTROL receipt path: {resolved}")
    raw = resolved.read_bytes()
    if len(raw) > 256_000:
        raise ValueError(f"CONTROL receipt is outside the bounded metadata size: {resolved}")
    return raw


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected one {label} source anchor, found {count}")
    return source.replace(old, new, 1)


def _replace_argv_assignment(source: str, argv: list[str]) -> str:
    node, _ = _literal_assignment(source, "TRAINER_ARGV")
    if not isinstance(node, ast.Assign) or node.end_lineno is None:
        raise ValueError("trainer argv assignment is not a simple source block")
    lines = source.splitlines(keepends=True)
    replacement = "TRAINER_ARGV = " + pprint.pformat(argv, width=88, sort_dicts=False) + "\n"
    return "".join(lines[: node.lineno - 1]) + replacement + "".join(lines[node.end_lineno :])


def _control_gate_source(
    identity: dict[str, object], acceptance_raw: bytes, stop_raw: bytes
) -> str:
    acceptance_b64 = base64.b64encode(acceptance_raw).decode("ascii")
    stop_b64 = base64.b64encode(stop_raw).decode("ascii")
    return f"""\nCONTROL_RUN = {EXPECTED_CONTROL_RUN!r}\nCONTROL_UNIT = {EXPECTED_CONTROL_UNIT!r}\nCONTROL_ACCEPTANCE_SHA256 = {identity["acceptance_sha256"]!r}\nCONTROL_EXACT_STOP_SHA256 = {identity["exact_stop_sha256"]!r}\nCONTROL_ACCEPTANCE_B64 = {acceptance_b64!r}\nCONTROL_EXACT_STOP_B64 = {stop_b64!r}\n"""


def _control_validation_source() -> str:
    return """\ndef _control_receipt_bytes() -> tuple[bytes, bytes]:
    acceptance_raw = base64.b64decode(CONTROL_ACCEPTANCE_B64, validate=True)
    stop_raw = base64.b64decode(CONTROL_EXACT_STOP_B64, validate=True)
    if (
        hashlib.sha256(acceptance_raw).hexdigest() != CONTROL_ACCEPTANCE_SHA256
        or hashlib.sha256(stop_raw).hexdigest() != CONTROL_EXACT_STOP_SHA256
    ):
        raise RuntimeError("bound CONTROL acceptance/stop bytes changed")
    return acceptance_raw, stop_raw


def _control_identity(acceptance_raw: bytes, stop_raw: bytes) -> dict[str, object]:
    acceptance = json.loads(acceptance_raw)
    stop = json.loads(stop_raw)
    if (
        not isinstance(acceptance, dict)
        or not isinstance(stop, dict)
        or acceptance.get("schema") != "fg19-child-control-fit-terminal-acceptance.v1"
        or acceptance.get("status") != "ACCEPTED"
        or acceptance.get("run_name") != CONTROL_RUN
        or acceptance.get("unit") != CONTROL_UNIT
    ):
        raise RuntimeError("CONTROL predecessor lacks the accepted frozen run identity")
    invocation = acceptance.get("invocation_id")
    if not isinstance(invocation, str) or not re.fullmatch(r"[0-9a-f]{32}", invocation):
        raise RuntimeError("CONTROL predecessor invocation identity is malformed")
    if (
        stop.get("unit") != CONTROL_UNIT
        or stop.get("invocation_id") != invocation
        or stop.get("acceptance_sha256") != CONTROL_ACCEPTANCE_SHA256
    ):
        raise RuntimeError("CONTROL exact stop does not cite the exact accepted invocation")
    before, after = stop.get("before"), stop.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise RuntimeError("CONTROL exact-stop states are missing")
    if (
        before.get("InvocationID") != invocation
        or before.get("MainPID") != "0"
        or before.get("ExecMainCode") != "1"
        or before.get("ExecMainStatus") != "0"
        or before.get("ActiveState") != "active"
        or before.get("SubState") != "exited"
        or after.get("LoadState") != "not-found"
        or after.get("ActiveState") != "inactive"
        or after.get("SubState") != "dead"
        or after.get("MainPID") != "0"
        or after.get("InvocationID") != ""
        or stop.get("populated_before") != {}
        or stop.get("populated_after") != {}
    ):
        raise RuntimeError("CONTROL predecessor was not accepted and exactly stopped after native exit0")
    return {
        "acceptance_sha256": CONTROL_ACCEPTANCE_SHA256,
        "exact_stop_sha256": CONTROL_EXACT_STOP_SHA256,
        "run_name": CONTROL_RUN,
        "unit": CONTROL_UNIT,
        "invocation_id": invocation,
        "status": "ACCEPTED_AND_EXACT_STOPPED",
    }
"""


def _render_body(acceptance_raw: bytes, stop_raw: bytes) -> str:
    identity = validate_control_receipts(acceptance_raw, stop_raw)
    original = CONTROL_BODY.read_text(encoding="utf-8")
    if _sha256(original.encode()) != EXPECTED_CONTROL_BODY_SHA256:
        raise ValueError("reviewed CONTROL body template bytes changed")
    control_recipe = _load_json(CONTROL_RECIPE)
    if _sha256(CONTROL_RECIPE.read_bytes()) != EXPECTED_CONTROL_RECIPE_SHA256:
        raise ValueError("CONTROL recipe binding bytes changed")
    ema_recipe = _load_json(EMA_RECIPE)
    control_argv = control_recipe.get("trainer_argv")
    ema_argv = ema_recipe.get("trainer_argv")
    if not isinstance(control_argv, list) or not isinstance(ema_argv, list):
        raise ValueError("CONTROL/EMA recipe argv is missing")
    derived = _successor_argv(control_argv)
    _assert_argv_delta(control_argv, derived)
    if derived != ema_argv:
        raise ValueError("EMA recipe argv differs from the exact allowed CONTROL delta")
    argv_sha = hashlib.sha256(json.dumps(derived, separators=(",", ":")).encode()).hexdigest()
    if argv_sha != ema_recipe.get("trainer_argv_sha256"):
        raise ValueError("EMA canonical trainer argv digest differs from its recipe binding")
    body = _replace_argv_assignment(original, derived)
    body = _replace_once(
        body,
        'OUTPUT = B / "research_tmp/fg19_child_control_fit_v1"',
        'OUTPUT = B / "research_tmp/fg19_child_ema_fit_v1"',
        "evidence path",
    )
    body = _replace_once(
        body,
        'RUN = "fg19_child_control_f0_s1001_fit_v1"',
        f'RUN = "{EXPECTED_EMA_RUN}"',
        "run name",
    )
    body = _replace_once(
        body,
        'UNIT = "computer-vision-homework-fg19-child-control-fit-v1.service"',
        f'UNIT = "{EXPECTED_EMA_UNIT}"',
        "systemd unit",
    )
    body = body.replace("CONTROL", "EMA").replace("control", "ema")
    body = _replace_once(
        body, "import os\n", "import os\nimport re\n", "receipt identity regex import"
    )
    body = _replace_once(
        body,
        'TRAINER_ARGV_SHA256 = "69e735294550802364f29a94a3dbc68873aa2e4500fe543e4ffb1b90f594b9a2"',
        f'TRAINER_ARGV_SHA256 = "{argv_sha}"',
        "trainer argv digest",
    )
    body = _replace_once(
        body,
        '"representation_mode": "base"',
        '"representation_mode": "ema"',
        "admission representation mode",
    )
    unit_marker = f'UNIT = "{EXPECTED_EMA_UNIT}"\n'
    body = _replace_once(
        body,
        unit_marker,
        unit_marker + _control_gate_source(identity, acceptance_raw, stop_raw),
        "predecessor receipt constants",
    )
    trainer_marker = "def _trainer_argv() -> list[str]:\n"
    body = _replace_once(
        body,
        trainer_marker,
        _control_validation_source() + "\n\n" + trainer_marker,
        "predecessor receipt validator",
    )
    main_marker = (
        "    profile_acceptance_raw, profile_stop_raw = _profile_receipt_bytes()\n"
        "    receipt_bytes, staged_pins = _decode_staged_receipts()\n"
    )
    main_replacement = (
        "    profile_acceptance_raw, profile_stop_raw = _profile_receipt_bytes()\n"
        "    control_acceptance_raw, control_stop_raw = _control_receipt_bytes()\n"
        "    control_predecessor = _control_identity(control_acceptance_raw, control_stop_raw)\n"
        "    receipt_bytes, staged_pins = _decode_staged_receipts()\n"
        '    receipt_bytes.update({"CONTROL_TERMINAL_ACCEPTANCE.json": control_acceptance_raw, '
        '"CONTROL_EXACT_STOP.json": control_stop_raw})\n'
        '    staged_pins.update({"CONTROL_TERMINAL_ACCEPTANCE.json": CONTROL_ACCEPTANCE_SHA256, '
        '"CONTROL_EXACT_STOP.json": CONTROL_EXACT_STOP_SHA256})\n'
    )
    body = _replace_once(body, main_marker, main_replacement, "early predecessor receipt gate")
    body = _replace_once(
        body,
        '        "child_profile": profile,\n',
        '        "child_profile": profile,\n        "control_predecessor": control_predecessor,\n',
        "admission predecessor linkage",
    )
    body = _replace_once(
        body,
        '        "submission": submission,\n',
        '        "submission": submission,\n        "control_predecessor": control_predecessor,\n',
        "launch predecessor linkage",
    )
    ast.parse(body)
    return body


def _check_patch_anchors() -> dict[str, object]:
    body = CONTROL_BODY.read_text(encoding="utf-8")
    if _sha256(body.encode()) != EXPECTED_CONTROL_BODY_SHA256:
        raise ValueError("reviewed CONTROL body template bytes changed")
    anchors = {
        "unit_assignment": 'UNIT = "computer-vision-homework-fg19-child-control-fit-v1.service"',
        "main_receipt_gate": (
            "    profile_acceptance_raw, profile_stop_raw = _profile_receipt_bytes()\n"
            "    receipt_bytes, staged_pins = _decode_staged_receipts()\n"
        ),
        "admission_link": '        "child_profile": profile,\n',
        "launch_link": '        "submission": submission,\n',
    }
    for label, anchor in anchors.items():
        if body.count(anchor) != 1:
            raise ValueError(f"conditional renderer anchor {label} is not unique")
    _literal_assignment(body, "TRAINER_ARGV")
    _literal_assignment(body, "TRAINER_ARGV_SHA256")
    return {"base_body_sha256": _sha256(body.encode()), "unique_patch_anchors": sorted(anchors)}


def _synthetic_receipt_pair() -> tuple[bytes, bytes]:
    acceptance = {
        "schema": CONTROL_ACCEPTANCE_SCHEMA,
        "status": "ACCEPTED",
        "run_name": EXPECTED_CONTROL_RUN,
        "unit": EXPECTED_CONTROL_UNIT,
        "invocation_id": "0123456789abcdef0123456789abcdef",
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_file_count": 167,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "successful_updates": 44_760,
        "completed_batches": 4_476,
        "amp_skipped_updates": 0,
        "nonfinite_updates": 0,
        "next_epoch": 6,
        "selected_epoch": 5,
    }
    acceptance_raw = (json.dumps(acceptance, sort_keys=True) + "\n").encode()
    stop = {
        "unit": EXPECTED_CONTROL_UNIT,
        "invocation_id": acceptance["invocation_id"],
        "acceptance_sha256": _sha256(acceptance_raw),
        "before": {
            "InvocationID": acceptance["invocation_id"],
            "MainPID": "0",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "ActiveState": "active",
            "SubState": "exited",
        },
        "after": {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "MainPID": "0",
            "InvocationID": "",
        },
        "populated_before": {},
        "populated_after": {},
        "stop": {"returncode": 0, "stdout": "", "stderr": ""},
    }
    return acceptance_raw, (json.dumps(stop, sort_keys=True) + "\n").encode()


def run_local_checks() -> dict[str, object]:
    """Check the frozen recipe, source semantics, anchors, and fail-closed receipt gate."""
    control = _load_json(CONTROL_RECIPE)
    ema = _load_json(EMA_RECIPE)
    if _sha256(CONTROL_RECIPE.read_bytes()) != EXPECTED_CONTROL_RECIPE_SHA256:
        raise ValueError("CONTROL recipe binding bytes changed")
    control_argv = control.get("trainer_argv")
    ema_argv = ema.get("trainer_argv")
    if not isinstance(control_argv, list) or not isinstance(ema_argv, list):
        raise ValueError("recipe argv is missing")
    absent = ema.get("arguments_deliberately_absent")
    if not isinstance(absent, list) or "--resume" not in absent:
        raise ValueError("EMA recipe must explicitly prohibit resume")
    if "--resume" in ema_argv:
        raise ValueError("EMA trainer argv must not resume a prior run")
    binder = _object(ema.get("binder"), "binder")
    binder_sha = _sha256(Path(__file__).read_bytes())
    if binder.get("sha256") != binder_sha:
        raise ValueError("frozen recipe does not pin these conditional binder bytes")
    derived = _successor_argv(control_argv)
    _assert_argv_delta(control_argv, derived)
    if derived != ema_argv:
        raise ValueError(
            "successor recipe contains argv changes outside the two authorized identity values"
        )
    digest = hashlib.sha256(json.dumps(derived, separators=(",", ":")).encode()).hexdigest()
    if digest != ema.get("trainer_argv_sha256"):
        raise ValueError("frozen EMA trainer argv digest does not match")
    source_hashes = _source_component_hashes()
    mode_semantics = _source_mode_semantics_check()
    cli = _canonical_cli_check(derived)
    anchors = _check_patch_anchors()
    accepted_raw, stopped_raw = _synthetic_receipt_pair()
    synthetic_identity = validate_control_receipts(accepted_raw, stopped_raw)
    invalid_acceptance = json.loads(accepted_raw)
    invalid_acceptance["status"] = "PENDING"
    invalid_raw = (json.dumps(invalid_acceptance, sort_keys=True) + "\n").encode()
    try:
        validate_control_receipts(invalid_raw, stopped_raw)
    except ValueError:
        rejected_pending = True
    else:
        rejected_pending = False
    if not rejected_pending:
        raise ValueError("synthetic pending CONTROL receipt was not rejected")
    return {
        "schema": "fg19-child-ema-fit-local-checks.v1",
        "status": "PASS_LOCAL_PREPARATION_ONLY_CONTROL_RECEIPTS_UNBOUND",
        "recipe_sha256": _sha256(EMA_RECIPE.read_bytes()),
        "control_recipe_sha256": _sha256(CONTROL_RECIPE.read_bytes()),
        "control_body_sha256": anchors["base_body_sha256"],
        "ema_trainer_argv_sha256": digest,
        "binder_sha256": binder_sha,
        "changed_argv_values_only": [
            "--name: control to EMA run identity",
            "--representation-mode: base to ema",
        ],
        "canonical_cli": cli,
        "source_component_hashes": source_hashes,
        "ema_mode_source_semantics": mode_semantics,
        "template_patch_anchor_check": anchors,
        "synthetic_control_receipt_identity": synthetic_identity,
        "synthetic_pending_receipt_rejected": rejected_pending,
        "actual_control_acceptance_sha256": "UNBOUND_PENDING_ROOT_RECEIPT",
        "actual_control_exact_stop_sha256": "UNBOUND_PENDING_ROOT_RECEIPT",
        "rendered_body_exists": OUTPUT_BODY.exists(),
        "remote_action_performed": False,
        "payload_reads_or_hashes": 0,
        "profile_or_model_suite_repeated": False,
    }


def render_bound_body(acceptance_path: str, exact_stop_path: str) -> Path:
    """Render one local remote-admission body after both actual receipts validate."""
    if OUTPUT_BODY.exists() or OUTPUT_BODY.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing successor body: {OUTPUT_BODY}")
    acceptance_raw = _read_receipt(acceptance_path, "TERMINAL_ACCEPTANCE.json")
    exact_stop_raw = _read_receipt(exact_stop_path, "STDOUT.raw.json")
    validate_control_receipts(acceptance_raw, exact_stop_raw)
    body = _render_body(acceptance_raw, exact_stop_raw)
    descriptor = os.open(OUTPUT_BODY, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(body.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return OUTPUT_BODY


def main() -> None:
    """Run local static checks, or explicitly render after actual receipt binding."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render", action="store_true", help="render only after real CONTROL receipts pass"
    )
    parser.add_argument("--control-acceptance")
    parser.add_argument("--control-exact-stop")
    args = parser.parse_args()
    if args.render:
        if not args.control_acceptance or not args.control_exact_stop:
            parser.error("--render requires both --control-acceptance and --control-exact-stop")
        path = render_bound_body(args.control_acceptance, args.control_exact_stop)
        print(
            json.dumps({"status": "LOCAL_BODY_RENDERED_FOR_INDEPENDENT_REVIEW", "path": str(path)})
        )
        return
    if args.control_acceptance or args.control_exact_stop:
        parser.error("receipt paths are accepted only with --render")
    print(json.dumps(run_local_checks(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
