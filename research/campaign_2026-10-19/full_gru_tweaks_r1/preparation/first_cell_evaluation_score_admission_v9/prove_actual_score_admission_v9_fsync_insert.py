"""Prove the V9 body adds only one post-submit SCORE_D directory fsync block."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

REPOSITORY = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
CAMPAIGN = REPOSITORY / "research/campaign_2026-10-19/full_gru_tweaks_r1"
V8_DIRECTORY = CAMPAIGN / "preparation/first_cell_evaluation_score_admission_v8"
V9_DIRECTORY = CAMPAIGN / "preparation/first_cell_evaluation_score_admission_v9"
V8_BODY = V8_DIRECTORY / "REMOTE_SUBMISSION_TEMPLATE_V8.py"
V8_BINDING = V8_DIRECTORY / "SCORE_ADMISSION_SOURCE_BINDING_V8.json"
V9_BODY = V9_DIRECTORY / "REMOTE_SUBMISSION_V9.py"
V9_BINDING = V9_DIRECTORY / "SCORE_ADMISSION_SOURCE_BINDING_V9.json"
BUILDER = V9_DIRECTORY / "build_score_admission_v9_from_v8.py"
NEGATIVE = V9_DIRECTORY / "ACTUAL_SCORE_ADMISSION_V8_DURABILITY_NEGATIVE_V1.json"
PROOF = V9_DIRECTORY / "ACTUAL_SCORE_ADMISSION_V9_FSYNC_INSERT_PROOF.json"
V8_BODY_PIN = (
    187_541,
    "cbb339e6225152e12fb044ce60a6c7cf3f7d58d00f395790097ea7a2e037fd11",
)
V8_BINDING_PIN = (
    5_422,
    "b23d12daa1cb1a4d43f84507e1799ef045c0a9c105a5cf04bf3374005482b9e2",
)
BLOCK = b"""    directory_fd = os.open(
        SCORE_D, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
"""
MARKER = b"    launch = submit_once(admission)\n"
FOLLOWING = b'    admission_raw, _ = stable_file(SCORE_D / "ADMISSION.json", 1_000_000)\n'
UNCHANGED_BINDING_KEYS = (
    "body_template",
    "export_authority",
    "export_evidence_root",
    "helper_source",
    "non_staged_readonly_export_inputs",
    "outer_reply_keys",
    "resource_limits",
    "result_path",
    "score_evidence_root",
    "score_unit",
    "scorer_dependency_manifest",
    "scorer_payloads",
    "source_review",
    "staged_worker_files",
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pin(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "path": path.relative_to(REPOSITORY).as_posix(),
        "sha256": sha256(raw),
    }


def _main(tree: ast.Module) -> ast.FunctionDef:
    matches = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    if len(matches) != 1:
        raise AssertionError("expected one top-level main()")
    return matches[0]


def _directory_open(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    if not isinstance(node.targets[0], ast.Name) or node.targets[0].id != "directory_fd":
        return False
    call = node.value
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "os"
        and call.func.attr == "open"
        and bool(call.args)
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "SCORE_D"
    )


def _sync_try(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Try) or node.handlers or node.orelse:
        return False
    if len(node.body) != 1 or len(node.finalbody) != 1:
        return False
    body, final = node.body[0], node.finalbody[0]
    if not isinstance(body, ast.Expr) or not isinstance(body.value, ast.Call):
        return False
    if not isinstance(final, ast.Expr) or not isinstance(final.value, ast.Call):
        return False
    fsync_call = body.value
    close_call = final.value
    return (
        isinstance(fsync_call.func, ast.Attribute)
        and isinstance(fsync_call.func.value, ast.Name)
        and fsync_call.func.value.id == "os"
        and fsync_call.func.attr == "fsync"
        and len(fsync_call.args) == 1
        and isinstance(fsync_call.args[0], ast.Name)
        and fsync_call.args[0].id == "directory_fd"
        and isinstance(close_call.func, ast.Attribute)
        and isinstance(close_call.func.value, ast.Name)
        and close_call.func.value.id == "os"
        and close_call.func.attr == "close"
        and len(close_call.args) == 1
        and isinstance(close_call.args[0], ast.Name)
        and close_call.args[0].id == "directory_fd"
    )


def _literal_payloads(tree: ast.Module) -> object:
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "PAYLOAD_B64"
    ]
    if len(nodes) != 1 or nodes[0].value is None:
        raise AssertionError("body must contain exactly one literal PAYLOAD_B64")
    return ast.literal_eval(nodes[0].value)


def main() -> None:
    old_raw = V8_BODY.read_bytes()
    new_raw = V9_BODY.read_bytes()
    old_binding_raw = V8_BINDING.read_bytes()
    new_binding_raw = V9_BINDING.read_bytes()
    if (len(old_raw), sha256(old_raw)) != V8_BODY_PIN:
        raise AssertionError("frozen V8 actual body pin differs")
    if (len(old_binding_raw), sha256(old_binding_raw)) != V8_BINDING_PIN:
        raise AssertionError("frozen V8 actual binding pin differs")
    if old_raw.count(MARKER + FOLLOWING) != 1:
        raise AssertionError("expected one exact post-submit/reply boundary in V8 body")
    if new_raw.count(MARKER + BLOCK + FOLLOWING) != 1:
        raise AssertionError("expected one exact post-submit SCORE_D fsync insertion")
    if new_raw.count(BLOCK) != 2:
        raise AssertionError(
            "V9 must preserve the old pre-submit block and add one post-submit block"
        )
    restored = new_raw.replace(MARKER + BLOCK + FOLLOWING, MARKER + FOLLOWING, 1)
    if restored != old_raw:
        raise AssertionError(
            "removing only the inserted durability block must restore V8 byte-for-byte"
        )

    old_tree = ast.parse(old_raw.decode("utf-8", errors="strict"))
    new_tree = ast.parse(new_raw.decode("utf-8", errors="strict"))
    _main(old_tree)
    new_main = _main(new_tree)
    launch_indices = [
        index
        for index, statement in enumerate(new_main.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "launch"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "submit_once"
    ]
    if len(launch_indices) != 1:
        raise AssertionError("V9 must have one submit_once result assignment")
    launch_index = launch_indices[0]
    if (
        launch_index + 3 >= len(new_main.body)
        or not _directory_open(new_main.body[launch_index + 1])
        or not _sync_try(new_main.body[launch_index + 2])
    ):
        raise AssertionError(
            "V9 directory fsync does not immediately follow successful submit_once"
        )
    launch_reread_indices = [
        index
        for index, statement in enumerate(new_main.body)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "launch_raw"
            for top_target in statement.targets
            for target in ast.walk(top_target)
        )
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "stable_file"
            and any(
                isinstance(arg, ast.BinOp)
                and isinstance(arg.left, ast.Name)
                and arg.left.id == "SCORE_D"
                and isinstance(arg.right, ast.Constant)
                and arg.right.value == "LAUNCH.json"
                for arg in node.args
            )
            for node in ast.walk(statement)
        )
    ]
    if launch_reread_indices != [launch_index + 4]:
        raise AssertionError("V9 inserted fsync must precede the stable LAUNCH reread")
    reply_indices = [
        index
        for index, statement in enumerate(new_main.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "print"
    ]
    if len(reply_indices) != 1 or reply_indices[0] <= launch_reread_indices[0]:
        raise AssertionError("four-key reply must follow the stable LAUNCH reread")
    del new_main.body[launch_index + 1 : launch_index + 3]
    if ast.dump(new_tree, include_attributes=False) != ast.dump(old_tree, include_attributes=False):
        raise AssertionError("deleting the one V9 post-submit fsync AST block must restore V8 AST")
    if _literal_payloads(old_tree) != _literal_payloads(ast.parse(new_raw.decode("utf-8"))):
        raise AssertionError("the six embedded scorer/receipt payload literals changed")

    old_binding = json.loads(old_binding_raw)
    new_binding = json.loads(new_binding_raw)
    if any(new_binding.get(key) != old_binding.get(key) for key in UNCHANGED_BINDING_KEYS):
        raise AssertionError("V9 binding changed an identity, gate, or payload pin")
    if new_binding.get("schema") != "fg19.first-cell-score-admission-source-binding.v9":
        raise AssertionError("V9 binding schema differs")
    rendered = new_binding.get("rendered_source")
    if rendered != {
        "path": V9_BODY.relative_to(REPOSITORY).as_posix(),
        "bytes": len(new_raw),
        "sha256": sha256(new_raw),
    }:
        raise AssertionError("V9 binding does not pin the exact V9 body")
    negative = json.loads(NEGATIVE.read_text(encoding="utf-8"))
    if negative.get("status") != "REJECT_SOURCE_ONLY_NO_B_ACTION":
        raise AssertionError("V8 durability negative record status differs")
    if (negative["prior_body"]["bytes"], negative["prior_body"]["sha256"]) != V8_BODY_PIN:
        raise AssertionError("V8 source-only negative does not retain exact body pin")

    proof = {
        "schema": "fg19.first-cell-score-admission-v9-fsync-insert-proof.v1",
        "status": "PASS_EXACT_ONE_BLOCK_INVERSE_AND_SIX_PAYLOAD_PINS",
        "scope": "Local byte/AST/metadata proof only; no body execution, imports, tests, B action, or model/array payload access.",
        "prior_body": pin(V8_BODY),
        "prior_binding": pin(V8_BINDING),
        "new_body": pin(V9_BODY),
        "new_binding": pin(V9_BINDING),
        "builder": pin(BUILDER),
        "source_negative": pin(NEGATIVE),
        "single_change": {
            "inserted_block": BLOCK.decode("utf-8"),
            "position": "immediately after successful submit_once(admission), before stable_file rereads LAUNCH.json",
            "byte_inverse_restores_v8": True,
            "ast_inverse_restores_v8": True,
            "same_embedded_payload_b64_mapping": True,
            "unchanged_binding_contract_and_six_payload_pins": True,
        },
    }
    encoded = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode("utf-8")
    PROOF.write_bytes(encoded)
    print(f"wrote {PROOF.relative_to(REPOSITORY)} bytes={len(encoded)} sha256={sha256(encoded)}")


if __name__ == "__main__":
    main()
