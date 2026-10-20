"""Bind the fixed F0 NLL child to the actual dictionary child's terminal receipts."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import pprint
import re
import stat
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
REPOSITORY = HERE.parents[4]
DICTIONARY_PREPARATION = CAMPAIGN / "preparation/child_dictionary_fit_v1"
DICTIONARY_BODY = DICTIONARY_PREPARATION / "REMOTE_SUBMISSION.py"
SOURCE_MANIFEST = CAMPAIGN / "preparation/teacher_cache_cpu_repair_v2/SOURCE_MANIFEST.json"
RECIPE = HERE / "NLL_RECIPE_BINDING_V1.json"
TEMPLATE_RECORD = HERE / "NLL_ADMISSION_TEMPLATE_V1.json"
TEMPLATE_BODY = HERE / "NLL_LAUNCH_TEMPLATE.py"
FINAL_BODY = HERE / "REMOTE_SUBMISSION.py"
FINAL_BINDING = HERE / "NLL_FINAL_BINDING_RECEIPT_V1.json"
CHILD_ACCEPTANCE = CAMPAIGN / "records/child_dictionary_fit_v1/TERMINAL_ACCEPTANCE.json"
CHILD_EXACT_STOP = CAMPAIGN / "records/child_dictionary_fit_v1/exact_stop/EXACT_STOP.json"

EXPECTED_DICTIONARY_BODY_BYTES = 196_111
EXPECTED_DICTIONARY_BODY_SHA256 = "9f545489dd80c13831b1a02c75dddf4059f3cbfb84311b2400db4830b4fdd77e"
EXPECTED_SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
EXPECTED_SOURCE_MANIFEST_SHA256 = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
EXPECTED_SOURCE_FILES = {
    "scripts/train_rnn.py": (
        1790,
        "74530d3499bba1b5647e459105cb263bfc7205d7f3aa928684dfb279c2677345",
    ),
    "src/wnn/training/cli_features.py": (
        4409,
        "680687de564a7c96ac96296a729ccb6aa37a8ecb913e525153e769fa14abd79a",
    ),
    "src/wnn/training/cli_targets.py": (
        5996,
        "ffc05685353085f2ab627d0b4800af81e06520df94f652b6a888e546176c1b57",
    ),
    "src/wnn/training/input_representation.py": (
        11714,
        "934bac99544c04ac3b77f0b9f78fa4b40ce5782a785fee04d9450d9d568b116f",
    ),
    "src/wnn/training/inputs.py": (
        9077,
        "b433ad56965e4272f3209e02f9108527fdaeabebefae07616dff61a4bbe5c9ea",
    ),
    "src/wnn/training/outputs.py": (
        10325,
        "35aab70e8aeb1ddabfa53f0f274ddc7aaa8c34d2e60ee92094a2221fcd20706a",
    ),
    "src/wnn/training/losses.py": (
        10493,
        "5859a4d40f13e91f50d34dfaed42d12dcfa2e229a85721ffc68bb5d69e1cfa66",
    ),
    "src/wnn/training/mean_variance.py": (
        5445,
        "e4a067bb9d467d121720854876cfffb083ab803a4ed440ec96025779553e15c5",
    ),
    "src/wnn/training/representation_validation.py": (
        1418,
        "983e63fecc8886a67ec3452e21ca3637a25588eba02ab225ce5e1b6f9d8fb887",
    ),
    "src/wnn/training/setup_model.py": (
        12302,
        "cec9748ad67c995a3c73f6b0288780ba52d0952445251c410113f6670c79fe33",
    ),
    "src/wnn/training/setup_preprocessing.py": (
        12381,
        "a02a47186da846c1dbc82d49d5fc0f35f56e7812120576febeda7988e38ce623",
    ),
}

RUN = "fg19_child_nll_f0_s1001_fit_v1"
UNIT = "computer-vision-homework-fg19-child-nll-fit-v1.service"
MODE = "base"
OBJECTIVE = "nll"
DICTIONARY_RUN = "fg19_child_dictionary_f0_s1001_fit_v1"
DICTIONARY_UNIT = "computer-vision-homework-fg19-child-dictionary-fit-v1.service"
DICTIONARY_INVOCATION = "73b10b699e034a1b9952b2377b845095"
DICTIONARY_LAUNCH_ACCEPTANCE_SHA256 = (
    "33fa4fd4cb37c16374e615c2eb6d698f5f3065b252803377260fdae6dd913a95"
)
DICTIONARY_BODY_SHA256 = EXPECTED_DICTIONARY_BODY_SHA256
UNBOUND_ACCEPTANCE = "UNBOUND_PENDING_ACTUAL_DICTIONARY_CHILD_TERMINAL_ACCEPTANCE"
UNBOUND_EXACT_STOP = "UNBOUND_PENDING_ACTUAL_DICTIONARY_CHILD_EXACT_STOP"
ACCEPTANCE_B64_SENTINEL = "UNBOUND_ACCEPTANCE_BYTES"
EXACT_STOP_B64_SENTINEL = "UNBOUND_EXACT_STOP_BYTES"

EXPECTED_RECIPE_BYTES = 11_391
EXPECTED_RECIPE_SHA256 = "51f753f6f2163d162eb10776258ec794770c243fe4b627b0c6f033d02b83ec43"
EXPECTED_TEMPLATE_RECORD_BYTES = 4_231
EXPECTED_TEMPLATE_RECORD_SHA256 = "96a8422463a24f3d772e18f07fedb20e35eeb746180f35c6cfc0aded168021c6"
EXPECTED_TEMPLATE_BODY_BYTES = 203_526
EXPECTED_TEMPLATE_BODY_SHA256 = "7f75525a14527c9cc11e0b44da7969846120fe5231ed2d4da0f6169d82fd8dc3"


def _sha256(raw: bytes) -> str:
    """Return the lowercase SHA-256 digest for exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def _read_regular(path: Path, maximum: int, label: str) -> bytes:
    """Read one bounded, physical regular local file without following a symlink."""
    info = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or path.resolve(strict=True) != path:
        raise ValueError(f"{label} must be a physical regular file: {path}")
    if info.st_size <= 0 or info.st_size > maximum:
        raise ValueError(f"{label} has an invalid bounded size: {info.st_size}")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError(f"{label} changed while being read: {path}")
    return raw


def _literal_assignment(source: str, name: str) -> tuple[ast.Assign, Any]:
    """Read one module-level literal assignment without importing the module."""
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return node, ast.literal_eval(node.value)
    raise ValueError(f"missing module assignment {name}")


def _replace_assignment(source: str, name: str, value: object) -> str:
    """Replace exactly one module-level assignment using its parsed source span."""
    node, _ = _literal_assignment(source, name)
    if node.end_lineno is None:
        raise ValueError(f"assignment has no source end: {name}")
    lines = source.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: node.lineno - 1])
    end = sum(len(line) for line in lines[: node.end_lineno])
    replacement = f"{name} = {pprint.pformat(value, width=88, sort_dicts=False)}\n"
    return source[:start] + replacement + source[end:]


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    """Replace one fixed source anchor and reject drift or ambiguous matches."""
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected one {label} anchor, found {count}")
    return source.replace(old, new, 1)


def _replace_cli_identity(source: str) -> tuple[str, list[str], list[str]]:
    """Change only run identity, representation mode, and mean-variance objective."""
    _, original = _literal_assignment(source, "TRAINER_ARGV")
    _, original_sha = _literal_assignment(source, "TRAINER_ARGV_SHA256")
    if (
        not isinstance(original, list)
        or _sha256(json.dumps(original, separators=(",", ":")).encode()) != original_sha
    ):
        raise ValueError("dictionary trainer argv or its canonical digest changed")
    expected_sha = "096b036c8af6220512e51b2002f3598729274b482d5db4b1df8099b26f91ab6d"
    if original_sha != expected_sha:
        raise ValueError("dictionary trainer argv differs from the reviewed V2 launch")
    argv = list(original)
    for index, expected in {1: DICTIONARY_RUN, 48: "sparse", 54: "wmse"}.items():
        if argv[index] != expected:
            raise ValueError(f"dictionary argv identity differs at index {index}")
    argv[1] = RUN
    argv[48] = MODE
    argv[54] = OBJECTIVE
    if len(original) != len(argv):
        raise ValueError("NLL trainer argv length differs from dictionary child")
    changed = [
        index
        for index, (left, right) in enumerate(zip(original, argv, strict=True))
        if left != right
    ]
    if changed != [1, 48, 54]:
        raise ValueError(f"NLL argv differs beyond its fixed three values: {changed}")
    if argv[45] != "fg19_parent_h256x2_f0_s1001_fit_r2_v1@ep3":
        raise ValueError("NLL initializer must remain the accepted parent checkpoint")
    if (
        argv[50]
        != "/home/grinchenkoeg/computer-vision-homework-20261018/data/artifacts/research/fg19_dictionary_f0_v1/input_representation.pt"
    ):
        raise ValueError("NLL must retain the exact shared dictionary cache path")
    if argv[55:59] != ["--loss", "wmse", "--loss-norm", "chunk"]:
        raise ValueError("NLL must retain the supported chunk-normalized --loss wmse gate")
    if argv[51:55] != ["--mean-variance-head", "1", "--mean-variance-objective", "nll"]:
        raise ValueError("NLL must use the source-qualified mean/logvariance NLL objective")
    digest = _sha256(json.dumps(argv, separators=(",", ":")).encode())
    source = _replace_assignment(source, "TRAINER_ARGV", argv)
    source = _replace_assignment(source, "TRAINER_ARGV_SHA256", digest)
    return source, original, argv


def _insert_predecessor_function(source: str) -> str:
    """Add a strict reader for the actual dictionary child's accepted/exact-stopped receipts."""
    anchor = "def main() -> None:\n"
    function = '''def _dictionary_child_terminal_predecessor() -> tuple[bytes, bytes, dict[str, object]]:
    """Require actual accepted terminal and exact-stop bytes for the dictionary child."""
    if (
        NLL_DICTIONARY_CHILD_ACCEPTANCE_SHA256.startswith("UNBOUND_")
        or NLL_DICTIONARY_CHILD_EXACT_STOP_SHA256.startswith("UNBOUND_")
        or NLL_DICTIONARY_CHILD_ACCEPTANCE_B64.startswith("UNBOUND_")
        or NLL_DICTIONARY_CHILD_EXACT_STOP_B64.startswith("UNBOUND_")
    ):
        raise RuntimeError("dictionary child terminal acceptance and exact stop are unbound")
    if not re.fullmatch(r"[0-9a-f]{64}", NLL_DICTIONARY_CHILD_ACCEPTANCE_SHA256):
        raise RuntimeError("dictionary child acceptance SHA-256 pin is malformed")
    if not re.fullmatch(r"[0-9a-f]{64}", NLL_DICTIONARY_CHILD_EXACT_STOP_SHA256):
        raise RuntimeError("dictionary child exact-stop SHA-256 pin is malformed")
    acceptance_raw = base64.b64decode(NLL_DICTIONARY_CHILD_ACCEPTANCE_B64, validate=True)
    stop_raw = base64.b64decode(NLL_DICTIONARY_CHILD_EXACT_STOP_B64, validate=True)
    if hashlib.sha256(acceptance_raw).hexdigest() != NLL_DICTIONARY_CHILD_ACCEPTANCE_SHA256:
        raise RuntimeError("dictionary child acceptance bytes do not match their pin")
    if hashlib.sha256(stop_raw).hexdigest() != NLL_DICTIONARY_CHILD_EXACT_STOP_SHA256:
        raise RuntimeError("dictionary child exact-stop bytes do not match their pin")
    acceptance = json.loads(acceptance_raw)
    exact_stop = json.loads(stop_raw)
    if not isinstance(acceptance, dict) or not isinstance(exact_stop, dict):
        raise RuntimeError("dictionary child predecessor receipts must be JSON objects")
    expected_acceptance = {
        "schema": "fg19-child-dictionary-fit-terminal-acceptance.v1",
        "status": "ACCEPTED",
        "run_name": NLL_DICTIONARY_CHILD_RUN,
        "unit": NLL_DICTIONARY_CHILD_UNIT,
        "source_commit": P["source_commit"],
        "source_file_count": 167,
        "source_manifest_sha256": P["source_manifest_sha256"],
        "representation_mode": "sparse",
        "mean_variance_objective": "wmse",
        "successful_updates": 44760,
        "completed_batches": 4476,
        "amp_skipped_updates": 0,
        "nonfinite_updates": 0,
        "next_epoch": 6,
        "selected_epoch": 5,
        "epochs": 6,
        "main_pid": 0,
        "exec_main_code": 1,
        "exec_main_status": 0,
        "strict_fp32_finite_model_optimizer_verified": True,
        "selected_model_epoch5_and_training_state_tensor_identity_verified": True,
    }
    if any(acceptance.get(key) != value for key, value in expected_acceptance.items()):
        raise RuntimeError("dictionary child terminal acceptance identity or exact-stop status differs")
    for key in (
        "successful_updates",
        "completed_batches",
        "amp_skipped_updates",
        "nonfinite_updates",
        "next_epoch",
        "selected_epoch",
        "epochs",
        "main_pid",
        "exec_main_code",
        "exec_main_status",
    ):
        if type(acceptance.get(key)) is not int:
            raise RuntimeError(f"dictionary child acceptance field must be an integer: {key}")
    invocation = acceptance.get("invocation_id")
    if (
        not isinstance(invocation, str)
        or not re.fullmatch(r"[0-9a-f]{32}", invocation)
        or invocation != NLL_DICTIONARY_CHILD_INVOCATION
    ):
        raise RuntimeError("dictionary child acceptance invocation identity is malformed")
    files = acceptance.get("files")
    if not isinstance(files, dict) or not {"model_ep5.pt", "training_state.pt", "provenance.json"}.issubset(files):
        raise RuntimeError("dictionary child acceptance omits selected checkpoint/state provenance")
    for name, pin in files.items():
        if (
            not isinstance(name, str)
            or not isinstance(pin, dict)
            or type(pin.get("bytes")) is not int
            or pin["bytes"] <= 0
            or not isinstance(pin.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])
        ):
            raise RuntimeError("dictionary child acceptance has a malformed file pin")
    if (
        exact_stop.get("acceptance_sha256") != NLL_DICTIONARY_CHILD_ACCEPTANCE_SHA256
        or exact_stop.get("unit") != NLL_DICTIONARY_CHILD_UNIT
        or exact_stop.get("invocation_id") != invocation
    ):
        raise RuntimeError("dictionary child exact stop is not bound to its accepted invocation")
    before = exact_stop.get("before")
    after = exact_stop.get("after")
    stop = exact_stop.get("stop")
    if not isinstance(before, dict) or not isinstance(after, dict) or not isinstance(stop, dict):
        raise RuntimeError("dictionary child exact stop lacks before/after/stop evidence")
    if (
        before.get("LoadState") != "loaded"
        or before.get("ActiveState") != "active"
        or before.get("SubState") != "exited"
        or before.get("MainPID") != "0"
        or before.get("ExecMainCode") != "1"
        or before.get("ExecMainStatus") != "0"
        or before.get("InvocationID") != invocation
        or after.get("LoadState") != "not-found"
        or after.get("ActiveState") != "inactive"
        or after.get("SubState") != "dead"
        or after.get("MainPID") != "0"
        or after.get("InvocationID") not in ("", None)
        or type(stop.get("returncode")) is not int
        or stop.get("returncode") != 0
        or exact_stop.get("populated_before") != {}
        or exact_stop.get("populated_after") != {}
    ):
        raise RuntimeError("dictionary child exact-stop state or population guard differs")
    return acceptance_raw, stop_raw, {
        "run_name": NLL_DICTIONARY_CHILD_RUN,
        "unit": NLL_DICTIONARY_CHILD_UNIT,
        "invocation_id": invocation,
        "launch_acceptance_sha256": NLL_DICTIONARY_CHILD_LAUNCH_ACCEPTANCE_SHA256,
        "acceptance_sha256": NLL_DICTIONARY_CHILD_ACCEPTANCE_SHA256,
        "exact_stop_sha256": NLL_DICTIONARY_CHILD_EXACT_STOP_SHA256,
        "selected_model_ep5_sha256": files["model_ep5.pt"]["sha256"],
        "initialization_run": "fg19_parent_h256x2_f0_s1001_fit_r2_v1@ep3",
        "dictionary_artifact_sha256": DICTIONARY_SHA256,
    }


'''
    return _replace_once(
        source, anchor, function + anchor, "dictionary child terminal predecessor helper"
    )


def _transform_dictionary_body(base: bytes) -> tuple[bytes, dict[str, object]]:
    """Derive a visibly unbound NLL launch template from the reviewed dictionary body."""
    if (
        len(base) != EXPECTED_DICTIONARY_BODY_BYTES
        or _sha256(base) != EXPECTED_DICTIONARY_BODY_SHA256
    ):
        raise ValueError("reviewed dictionary launch body changed")
    source = base.decode("utf-8")
    _, base_argv = _literal_assignment(source, "TRAINER_ARGV")
    source, old_argv, nll_argv = _replace_cli_identity(source)
    replacements = {
        '"""Fail-closed admission template for the F0 seed-1001 dictionary fit."""': '"""UNBOUND local NLL launch template; no final submission body is admitted."""',
        'OUTPUT = B / "research_tmp/fg19_child_dictionary_fit_v1"': 'OUTPUT = B / "research_tmp/fg19_child_nll_fit_v1"',
        'RUN = "fg19_child_dictionary_f0_s1001_fit_v1"': f'RUN = "{RUN}"',
        'UNIT = "computer-vision-homework-fg19-child-dictionary-fit-v1.service"': f'UNIT = "{UNIT}"',
        '"fg19-child-dictionary-fit-admission.v1"': '"fg19-child-nll-fit-admission.v1"',
        '"fg19-child-dictionary-fit-launch.v1"': '"fg19-child-nll-fit-launch.v1"',
        '"representation_mode": "sparse",\n            "objective": "wmse",': '"representation_mode": "base",\n            "objective": "nll",',
        '"dictionary output/run path exists"': '"NLL output/run path exists"',
        '"new dictionary unit exists"': '"new NLL unit exists"',
        '"dictionary destination appeared during admission"': '"NLL destination appeared during admission"',
        '"dictionary unit appeared during admission"': '"NLL unit appeared during admission"',
        '"never retry; inspect this exact unit and durable dictionary output"': '"never retry; inspect this exact unit and durable NLL output"',
        '"dictionary canonical trainer argv digest changed"': '"NLL canonical trainer argv digest changed"',
        '"""Return the frozen direct canonical trainer argv for the dictionary fit."""': '"""Return the frozen direct canonical trainer argv for the NLL fit."""',
        '"""Run fresh read-only gates, then admit one create-only dictionary fit."""': '"""Fail closed unless the dictionary child is accepted and exactly stopped."""',
    }
    for old, new in replacements.items():
        source = _replace_once(source, old, new, f"NLL identity patch {old[:48]!r}")
    source = _replace_once(
        source,
        'UNIT = "computer-vision-homework-fg19-child-nll-fit-v1.service"\n',
        'UNIT = "computer-vision-homework-fg19-child-nll-fit-v1.service"\n'
        f'NLL_DICTIONARY_CHILD_RUN = "{DICTIONARY_RUN}"\n'
        f'NLL_DICTIONARY_CHILD_UNIT = "{DICTIONARY_UNIT}"\n'
        f'NLL_DICTIONARY_CHILD_INVOCATION = "{DICTIONARY_INVOCATION}"\n'
        f'NLL_DICTIONARY_CHILD_LAUNCH_ACCEPTANCE_SHA256 = "{DICTIONARY_LAUNCH_ACCEPTANCE_SHA256}"\n'
        f'NLL_DICTIONARY_CHILD_ACCEPTANCE_SHA256 = "{UNBOUND_ACCEPTANCE}"\n'
        f'NLL_DICTIONARY_CHILD_EXACT_STOP_SHA256 = "{UNBOUND_EXACT_STOP}"\n'
        f'NLL_DICTIONARY_CHILD_ACCEPTANCE_B64 = "{ACCEPTANCE_B64_SENTINEL}"\n'
        f'NLL_DICTIONARY_CHILD_EXACT_STOP_B64 = "{EXACT_STOP_B64_SENTINEL}"\n',
        "NLL child receipt placeholder constants",
    )
    source = _insert_predecessor_function(source)
    source = _replace_once(
        source,
        "    profile_acceptance_raw, profile_stop_raw = _profile_receipt_bytes()\n",
        "    dictionary_child_acceptance_raw, dictionary_child_stop_raw, dictionary_child_predecessor = _dictionary_child_terminal_predecessor()\n"
        "    profile_acceptance_raw, profile_stop_raw = _profile_receipt_bytes()\n",
        "early fail-closed child receipt check",
    )
    source = _replace_once(
        source,
        'receipt_bytes.update({"CONTROL_TERMINAL_ACCEPTANCE.json": control_acceptance_raw, "CONTROL_EXACT_STOP.json": control_stop_raw, "EMA_TERMINAL_ACCEPTANCE.json": ema_acceptance_raw, "EMA_EXACT_STOP.json": ema_stop_raw})',
        'receipt_bytes.update({"CONTROL_TERMINAL_ACCEPTANCE.json": control_acceptance_raw, "CONTROL_EXACT_STOP.json": control_stop_raw, "EMA_TERMINAL_ACCEPTANCE.json": ema_acceptance_raw, "EMA_EXACT_STOP.json": ema_stop_raw, "DICTIONARY_CHILD_TERMINAL_ACCEPTANCE.json": dictionary_child_acceptance_raw, "DICTIONARY_CHILD_EXACT_STOP.json": dictionary_child_stop_raw})',
        "staged child terminal receipts",
    )
    source = _replace_once(
        source,
        'staged_pins.update({"CONTROL_TERMINAL_ACCEPTANCE.json": CONTROL_ACCEPTANCE_SHA256, "CONTROL_EXACT_STOP.json": CONTROL_EXACT_STOP_SHA256, "EMA_TERMINAL_ACCEPTANCE.json": EMA_PREDECESSOR_ACCEPTANCE_SHA256, "EMA_EXACT_STOP.json": EMA_PREDECESSOR_EXACT_STOP_SHA256})',
        'staged_pins.update({"CONTROL_TERMINAL_ACCEPTANCE.json": CONTROL_ACCEPTANCE_SHA256, "CONTROL_EXACT_STOP.json": CONTROL_EXACT_STOP_SHA256, "EMA_TERMINAL_ACCEPTANCE.json": EMA_PREDECESSOR_ACCEPTANCE_SHA256, "EMA_EXACT_STOP.json": EMA_PREDECESSOR_EXACT_STOP_SHA256, "DICTIONARY_CHILD_TERMINAL_ACCEPTANCE.json": NLL_DICTIONARY_CHILD_ACCEPTANCE_SHA256, "DICTIONARY_CHILD_EXACT_STOP.json": NLL_DICTIONARY_CHILD_EXACT_STOP_SHA256})',
        "child terminal receipt pins",
    )
    source = _replace_once(
        source,
        '        "ema_predecessor": ema_predecessor,\n        "staged_receipt_sha256": {',
        '        "ema_predecessor": ema_predecessor,\n'
        '        "dictionary_child_fit_predecessor": dictionary_child_predecessor,\n'
        '        "staged_receipt_sha256": {',
        "NLL admission child predecessor link",
    )
    source = _replace_once(
        source,
        '        "ema_predecessor": ema_predecessor,\n        "systemctl_after_submission": post_submit,',
        '        "ema_predecessor": ema_predecessor,\n'
        '        "dictionary_child_fit_predecessor": dictionary_child_predecessor,\n'
        '        "systemctl_after_submission": post_submit,',
        "NLL launch child predecessor link",
    )
    body = source.encode("utf-8")
    info = {
        "dictionary_body_bytes": len(base),
        "dictionary_body_sha256": _sha256(base),
        "template_body_bytes": len(body),
        "template_body_sha256": _sha256(body),
        "argv_changed_indices": [1, 48, 54],
        "dictionary_argv_sha256": _sha256(json.dumps(old_argv, separators=(",", ":")).encode()),
        "nll_argv_sha256": _sha256(json.dumps(nll_argv, separators=(",", ":")).encode()),
    }
    _check_unbound_template(body, info)
    return body, info


def _check_source_manifest() -> dict[str, object]:
    """Bind NLL CLI semantics to the frozen 167-file source manifest."""
    raw = _read_regular(SOURCE_MANIFEST, 64_000, "source manifest")
    if _sha256(raw) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("canonical 167-file source manifest changed")
    manifest = json.loads(raw)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "fg19.teacher-cache-source-manifest.v1"
        or manifest.get("source_commit") != EXPECTED_SOURCE_COMMIT
        or not isinstance(manifest.get("files"), list)
        or len(manifest["files"]) != 167
    ):
        raise ValueError("canonical source manifest identity differs")
    found: dict[str, tuple[int, str]] = {}
    for record in manifest["files"]:
        if isinstance(record, dict) and record.get("path") in EXPECTED_SOURCE_FILES:
            found[record["path"]] = (record.get("bytes"), record.get("sha256"))
    if found != EXPECTED_SOURCE_FILES:
        raise ValueError("NLL source-qualified CLI/objective file pins differ from source manifest")
    return {
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_manifest_bytes": len(raw),
        "source_manifest_sha256": _sha256(raw),
        "source_file_count": len(manifest["files"]),
        "selected_source_file_pins": {
            path: {"bytes": pin[0], "sha256": pin[1]} for path, pin in sorted(found.items())
        },
    }


def _check_unbound_template(
    body: bytes, info: dict[str, object] | None = None
) -> dict[str, object]:
    """Assert identity, gate ordering, unbound receipt state, and no final body."""
    source = body.decode("utf-8")
    _, argv = _literal_assignment(source, "TRAINER_ARGV")
    _, argv_sha = _literal_assignment(source, "TRAINER_ARGV_SHA256")
    if _sha256(json.dumps(argv, separators=(",", ":")).encode()) != argv_sha:
        raise ValueError("NLL template argv digest mismatch")
    if (
        argv[1] != RUN
        or argv[48] != MODE
        or argv[54] != OBJECTIVE
        or argv[45] != "fg19_parent_h256x2_f0_s1001_fit_r2_v1@ep3"
        or argv[50]
        != "/home/grinchenkoeg/computer-vision-homework-20261018/data/artifacts/research/fg19_dictionary_f0_v1/input_representation.pt"
        or argv[55:59] != ["--loss", "wmse", "--loss-norm", "chunk"]
    ):
        raise ValueError("NLL template changed a protected recipe argument")
    if (
        _literal_assignment(source, "NLL_DICTIONARY_CHILD_ACCEPTANCE_SHA256")[1]
        != UNBOUND_ACCEPTANCE
    ):
        raise ValueError("template must remain unbound to dictionary terminal acceptance")
    if (
        _literal_assignment(source, "NLL_DICTIONARY_CHILD_EXACT_STOP_SHA256")[1]
        != UNBOUND_EXACT_STOP
    ):
        raise ValueError("template must remain unbound to dictionary exact stop")
    if (
        _literal_assignment(source, "NLL_DICTIONARY_CHILD_ACCEPTANCE_B64")[1]
        != ACCEPTANCE_B64_SENTINEL
    ):
        raise ValueError("template must not contain fabricated child acceptance bytes")
    if (
        _literal_assignment(source, "NLL_DICTIONARY_CHILD_EXACT_STOP_B64")[1]
        != EXACT_STOP_B64_SENTINEL
    ):
        raise ValueError("template must not contain fabricated child exact-stop bytes")
    main = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    guard_statement = main.body[1] if len(main.body) > 1 else None
    if (
        not isinstance(guard_statement, ast.Assign)
        or not isinstance(guard_statement.value, ast.Call)
        or not isinstance(guard_statement.value.func, ast.Name)
        or guard_statement.value.func.id != "_dictionary_child_terminal_predecessor"
    ):
        raise ValueError(
            "unbound child predecessor guard must run before any other admission action"
        )
    if FINAL_BODY.exists() or FINAL_BINDING.exists():
        raise ValueError("actual NLL body/binding must remain absent until receipts are supplied")
    return {
        "status": "PASS_UNBOUND_NO_FINAL_BODY",
        "run_name": RUN,
        "unit": UNIT,
        "argv_sha256": argv_sha,
        "argv_changed_indices": [1, 48, 54],
        "child_acceptance": UNBOUND_ACCEPTANCE,
        "child_exact_stop": UNBOUND_EXACT_STOP,
        "body_output_absent": True,
        **(info or {}),
    }


def _render_final_body(acceptance_raw: bytes, stop_raw: bytes) -> bytes:
    """Bind validated actual dictionary receipts into one create-only final body."""
    if len(acceptance_raw) > 64_000 or len(stop_raw) > 64_000:
        raise ValueError("dictionary child receipt exceeds the fixed local bound")
    source = TEMPLATE_BODY.read_text(encoding="utf-8")
    _check_unbound_template(source.encode("utf-8"))
    acceptance_sha = _sha256(acceptance_raw)
    stop_sha = _sha256(stop_raw)
    _validate_child_receipts(acceptance_raw, stop_raw, acceptance_sha, stop_sha)
    substitutions = {
        f'NLL_DICTIONARY_CHILD_ACCEPTANCE_SHA256 = "{UNBOUND_ACCEPTANCE}"': f'NLL_DICTIONARY_CHILD_ACCEPTANCE_SHA256 = "{acceptance_sha}"',
        f'NLL_DICTIONARY_CHILD_EXACT_STOP_SHA256 = "{UNBOUND_EXACT_STOP}"': f'NLL_DICTIONARY_CHILD_EXACT_STOP_SHA256 = "{stop_sha}"',
        f'NLL_DICTIONARY_CHILD_ACCEPTANCE_B64 = "{ACCEPTANCE_B64_SENTINEL}"': f'NLL_DICTIONARY_CHILD_ACCEPTANCE_B64 = "{base64.b64encode(acceptance_raw).decode("ascii")}"',
        f'NLL_DICTIONARY_CHILD_EXACT_STOP_B64 = "{EXACT_STOP_B64_SENTINEL}"': f'NLL_DICTIONARY_CHILD_EXACT_STOP_B64 = "{base64.b64encode(stop_raw).decode("ascii")}"',
    }
    for old, new in substitutions.items():
        source = _replace_once(source, old, new, "bound dictionary terminal receipt")
    rendered = source.encode("utf-8")
    if UNBOUND_ACCEPTANCE.encode() in rendered or UNBOUND_EXACT_STOP.encode() in rendered:
        raise ValueError("final NLL body retains an unbound predecessor marker")
    _validate_child_receipts(acceptance_raw, stop_raw, acceptance_sha, stop_sha)
    if FINAL_BODY.exists():
        raise FileExistsError("final NLL body path already exists; refusing overwrite")
    return rendered


def _validate_child_receipts(
    acceptance_raw: bytes, stop_raw: bytes, acceptance_sha: str, stop_sha: str
) -> None:
    """Validate actual dictionary receipt identity and exact exited-stop state."""
    if _sha256(acceptance_raw) != acceptance_sha or _sha256(stop_raw) != stop_sha:
        raise ValueError("dictionary child receipt byte digest mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", acceptance_sha) or not re.fullmatch(
        r"[0-9a-f]{64}", stop_sha
    ):
        raise ValueError("dictionary child receipt digest is malformed")
    acceptance = json.loads(acceptance_raw)
    stop = json.loads(stop_raw)
    if not isinstance(acceptance, dict) or not isinstance(stop, dict):
        raise ValueError("dictionary child receipts must be JSON objects")
    expected = {
        "schema": "fg19-child-dictionary-fit-terminal-acceptance.v1",
        "status": "ACCEPTED",
        "run_name": DICTIONARY_RUN,
        "unit": DICTIONARY_UNIT,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_file_count": 167,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "representation_mode": "sparse",
        "mean_variance_objective": "wmse",
        "successful_updates": 44760,
        "completed_batches": 4476,
        "amp_skipped_updates": 0,
        "nonfinite_updates": 0,
        "next_epoch": 6,
        "selected_epoch": 5,
        "epochs": 6,
        "main_pid": 0,
        "exec_main_code": 1,
        "exec_main_status": 0,
        "strict_fp32_finite_model_optimizer_verified": True,
        "selected_model_epoch5_and_training_state_tensor_identity_verified": True,
    }
    if any(acceptance.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "actual dictionary child acceptance schema/status/recipe/terminal identity differs"
        )
    for key in (
        "successful_updates",
        "completed_batches",
        "amp_skipped_updates",
        "nonfinite_updates",
        "next_epoch",
        "selected_epoch",
        "epochs",
        "main_pid",
        "exec_main_code",
        "exec_main_status",
    ):
        if type(acceptance.get(key)) is not int:
            raise ValueError(f"dictionary child acceptance field must be a strict integer: {key}")
    invocation = acceptance.get("invocation_id")
    if (
        not isinstance(invocation, str)
        or not re.fullmatch(r"[0-9a-f]{32}", invocation)
        or invocation != DICTIONARY_INVOCATION
    ):
        raise ValueError("dictionary child acceptance invocation id is malformed")
    files = acceptance.get("files")
    if not isinstance(files, dict) or not {
        "model_ep5.pt",
        "training_state.pt",
        "provenance.json",
    }.issubset(files):
        raise ValueError(
            "dictionary child acceptance does not bind selected model/state/provenance"
        )
    for name, pin in files.items():
        if (
            not isinstance(name, str)
            or not isinstance(pin, dict)
            or type(pin.get("bytes")) is not int
            or pin["bytes"] <= 0
            or not isinstance(pin.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])
        ):
            raise ValueError("dictionary child acceptance file map is malformed")
    if (
        stop.get("acceptance_sha256") != acceptance_sha
        or stop.get("unit") != DICTIONARY_UNIT
        or stop.get("invocation_id") != invocation
    ):
        raise ValueError("dictionary exact stop is not bound to its actual accepted invocation")
    before = stop.get("before")
    after = stop.get("after")
    stop_call = stop.get("stop")
    if (
        not isinstance(before, dict)
        or not isinstance(after, dict)
        or not isinstance(stop_call, dict)
    ):
        raise ValueError("dictionary exact stop lacks before/after/stop fields")
    if (
        before.get("LoadState") != "loaded"
        or before.get("ActiveState") != "active"
        or before.get("SubState") != "exited"
        or before.get("MainPID") != "0"
        or before.get("ExecMainCode") != "1"
        or before.get("ExecMainStatus") != "0"
        or before.get("InvocationID") != invocation
        or after.get("LoadState") != "not-found"
        or after.get("ActiveState") != "inactive"
        or after.get("SubState") != "dead"
        or after.get("MainPID") != "0"
        or after.get("InvocationID") not in ("", None)
        or type(stop_call.get("returncode")) is not int
        or stop_call.get("returncode") != 0
        or stop.get("populated_before") != {}
        or stop.get("populated_after") != {}
    ):
        raise ValueError("dictionary exact-stop state/population evidence is not accepted")


def _write_final_binding(body: bytes, acceptance_raw: bytes, stop_raw: bytes) -> None:
    """Create the final body and its local binding receipt without overwriting."""
    if FINAL_BODY.exists() or FINAL_BINDING.exists():
        raise FileExistsError("final NLL output already exists; refusing overwrite")
    FINAL_BODY.write_bytes(body)
    FINAL_BODY.chmod(0o700)
    record = {
        "schema": "fg19-child-nll-fit-final-binding.v1",
        "status": "PASS_LOCAL_RENDERED_BODY_AWAITING_INDEPENDENT_SOURCE_REVIEW",
        "body": {
            "path": str(FINAL_BODY.relative_to(CAMPAIGN)),
            "bytes": len(body),
            "sha256": _sha256(body),
        },
        "dictionary_child_terminal_acceptance": {
            "path": str(CHILD_ACCEPTANCE.relative_to(CAMPAIGN)),
            "bytes": len(acceptance_raw),
            "sha256": _sha256(acceptance_raw),
        },
        "dictionary_child_exact_stop": {
            "path": str(CHILD_EXACT_STOP.relative_to(CAMPAIGN)),
            "bytes": len(stop_raw),
            "sha256": _sha256(stop_raw),
        },
        "dictionary_child_is_sequential_predecessor_only": True,
        "initializer_remains": "fg19_parent_h256x2_f0_s1001_fit_r2_v1@ep3",
        "no_b_action": True,
    }
    raw = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    with FINAL_BINDING.open("xb") as stream:
        stream.write(raw)
        stream.flush()


def _check() -> dict[str, object]:
    """Check frozen local inputs, source pins, and the unbound template without rendering."""
    base = _read_regular(DICTIONARY_BODY, 250_000, "reviewed dictionary admission body")
    body, render_info = _transform_dictionary_body(base)
    source_info = _check_source_manifest()
    recipe_raw = _read_regular(RECIPE, 32_000, "NLL recipe")
    template_raw = _read_regular(TEMPLATE_RECORD, 32_000, "NLL admission template record")
    frozen_template = _read_regular(TEMPLATE_BODY, 250_000, "unbound NLL launch template")
    if (
        len(recipe_raw) != EXPECTED_RECIPE_BYTES
        or _sha256(recipe_raw) != EXPECTED_RECIPE_SHA256
        or len(template_raw) != EXPECTED_TEMPLATE_RECORD_BYTES
        or _sha256(template_raw) != EXPECTED_TEMPLATE_RECORD_SHA256
        or len(frozen_template) != EXPECTED_TEMPLATE_BODY_BYTES
        or _sha256(frozen_template) != EXPECTED_TEMPLATE_BODY_SHA256
        or body != frozen_template
    ):
        raise ValueError("frozen NLL recipe/template pins changed or template is not reproducible")
    recipe = json.loads(recipe_raw)
    template = json.loads(template_raw)
    if (
        recipe.get("schema") != "fg19-child-nll-fit-recipe-binding.v1"
        or recipe.get("status") != "LOCAL_RECIPE_FROZEN_NOT_ADMITTED"
    ):
        raise ValueError("NLL recipe schema/status differs")
    if (
        template.get("schema") != "fg19-child-nll-fit-admission-template.v1"
        or template.get("status")
        != "UNBOUND_DICTIONARY_CHILD_ACCEPTANCE_AND_EXACT_STOP__NO_FINAL_BODY"
    ):
        raise ValueError("NLL template must remain explicitly unbound")
    current_artifacts = template.get("current_artifacts")
    if (
        not isinstance(current_artifacts, dict)
        or current_artifacts.get("actual_dictionary_child_acceptance_sha256") != UNBOUND_ACCEPTANCE
        or current_artifacts.get("actual_dictionary_child_exact_stop_sha256") != UNBOUND_EXACT_STOP
        or current_artifacts.get("final_body_exists") is not False
        or current_artifacts.get("final_binding_receipt_exists") is not False
    ):
        raise ValueError("NLL template record is not fail-closed/unbound")
    if (
        recipe.get("identity", {}).get("run_name") != RUN
        or recipe.get("identity", {}).get("unit") != UNIT
        or recipe.get("science", {}).get("representation_mode") != MODE
        or recipe.get("science", {}).get("objective") != OBJECTIVE
        or recipe.get("accepted_unchanged_ancestry", {}).get("dictionary_child_run")
        != DICTIONARY_RUN
        or recipe.get("accepted_unchanged_ancestry", {}).get("dictionary_child_unit")
        != DICTIONARY_UNIT
        or recipe.get("accepted_unchanged_ancestry", {}).get("dictionary_child_invocation_id")
        != DICTIONARY_INVOCATION
        or recipe.get("only_trainer_argv_changes_from_actual_dictionary_child", {}).get(
            "nll_trainer_argv_sha256"
        )
        != _literal_assignment(frozen_template.decode("utf-8"), "TRAINER_ARGV_SHA256")[1]
    ):
        raise ValueError("NLL recipe values differ from the exact rendered local template")
    return {
        "status": "PASS_LOCAL_UNBOUND_TEMPLATE_ONLY",
        "source": source_info,
        "recipe": {"bytes": len(recipe_raw), "sha256": _sha256(recipe_raw)},
        "template_record": {"bytes": len(template_raw), "sha256": _sha256(template_raw)},
        "template_body": {"bytes": len(frozen_template), "sha256": _sha256(frozen_template)},
        "render_reproduction": render_info,
        "final_body_exists": FINAL_BODY.exists(),
        "final_binding_exists": FINAL_BINDING.exists(),
    }


def _render() -> dict[str, object]:
    """Render only after actual dictionary child acceptance and exact-stop files validate."""
    local_check = _check()
    if FINAL_BODY.exists() or FINAL_BINDING.exists():
        raise FileExistsError("final NLL output exists; refusing overwrite")
    acceptance_raw = _read_regular(CHILD_ACCEPTANCE, 64_000, "actual dictionary child acceptance")
    stop_raw = _read_regular(CHILD_EXACT_STOP, 64_000, "actual dictionary child exact stop")
    body = _render_final_body(acceptance_raw, stop_raw)
    _write_final_binding(body, acceptance_raw, stop_raw)
    return {
        "status": "PASS_LOCAL_RENDERED_BODY_AWAITING_INDEPENDENT_SOURCE_REVIEW",
        "local_unbound_packet_check": local_check["status"],
        "body": {"bytes": len(body), "sha256": _sha256(body), "path": str(FINAL_BODY)},
        "binding": {
            "bytes": FINAL_BINDING.stat().st_size,
            "sha256": _sha256(FINAL_BINDING.read_bytes()),
            "path": str(FINAL_BINDING),
        },
    }


def main() -> int:
    """Run a local-only check or render after both actual dictionary receipts exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check", action="store_true", help="verify only frozen local unbound inputs"
    )
    group.add_argument(
        "--prepare-unbound-template",
        action="store_true",
        help="create the fail-closed template from the pinned dictionary body",
    )
    group.add_argument(
        "--render",
        action="store_true",
        help="render after actual accepted and exact-stopped dictionary receipts validate",
    )
    args = parser.parse_args()
    if args.check:
        result = _check()
    elif args.prepare_unbound_template:
        if TEMPLATE_BODY.exists():
            raise FileExistsError("unbound NLL template exists; refusing overwrite")
        base = _read_regular(DICTIONARY_BODY, 250_000, "reviewed dictionary admission body")
        body, _ = _transform_dictionary_body(base)
        with TEMPLATE_BODY.open("xb") as stream:
            stream.write(body)
            stream.flush()
        result = {
            "status": "CREATED_UNBOUND_TEMPLATE_ONLY",
            "bytes": len(body),
            "sha256": _sha256(body),
        }
    else:
        result = _render()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
