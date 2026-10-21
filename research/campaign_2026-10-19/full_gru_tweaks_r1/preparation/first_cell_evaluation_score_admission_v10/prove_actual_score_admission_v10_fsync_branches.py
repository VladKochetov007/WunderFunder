from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V9_DIR = ROOT / "preparation/first_cell_evaluation_score_admission_v9"
V10_DIR = Path(__file__).resolve().parent
V9_BODY = V9_DIR / "REMOTE_SUBMISSION_V9.py"
V9_BINDING = V9_DIR / "SCORE_ADMISSION_SOURCE_BINDING_V9.json"
V10_BODY = V10_DIR / "REMOTE_SUBMISSION_V10.py"
V10_BINDING = V10_DIR / "SCORE_ADMISSION_SOURCE_BINDING_V10.json"
PROOF_PATH = V10_DIR / "ACTUAL_SCORE_ADMISSION_V10_FSYNC_BRANCHES_PROOF.json"
EXPECTED_V9_BODY_BYTES = 187_751
EXPECTED_V9_BODY_SHA256 = "039d52fc297ddfe7b537baab905dbb23933df780f2a815d0ed48aa862c7bd662"
EXPECTED_V9_BINDING_BYTES = 5_516
EXPECTED_V9_BINDING_SHA256 = "2798dc1cee7279425fc4d69c9ef5adcedd8736f3a619392abe40fdab4d448e68"
EXPECTED_V10_BODY_BYTES = 187_989
EXPECTED_V10_BODY_SHA256 = "24787cf037328baabebe19d4eb6ba5c452740d2b5648dbf478544e3ce33cbd53"
EXPECTED_V10_BINDING_BYTES = 5_602
EXPECTED_V10_BINDING_SHA256 = "9767f437d545ee9a6b448a6606be972ea6a7766ea3efebf24cbbb346df56a42e"

BLOCK4 = (
    "    directory_fd = os.open(\n"
    '        SCORE_D, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)\n'
    "    )\n"
    "    try:\n"
    "        os.fsync(directory_fd)\n"
    "    finally:\n"
    "        os.close(directory_fd)\n"
)
BLOCK8 = "".join(
    "    " + line if line.strip() else line for line in BLOCK4.splitlines(keepends=True)
)
MAIN_POST = "    launch = submit_once(admission)\n" + BLOCK4
MAIN_LAUNCH_AND_REREAD = (
    "    launch = submit_once(admission)\n"
    '    admission_raw, _ = stable_file(SCORE_D / "ADMISSION.json", 1_000_000)\n'
)
UNRESOLVED_END = (
    "        )\n"
    "        raise RuntimeError(\n"
    '            "score invocation ID unresolved; durable unknown outcome retained, never retry"\n'
    "        )\n"
)
UNRESOLVED_WITH_FSYNC = "        )\n" + BLOCK8 + UNRESOLVED_END[len("        )\n") :]
LAUNCH_WRITE = '    write_json_exclusive(SCORE_D / "LAUNCH.json", launch)\n'


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_create_only(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"existing proof output differs; refusing overwrite {path}")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def without_binding_metadata(binding: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in binding.items()
        if key not in {"schema", "scope", "rendered_source"}
    }


def main() -> None:
    v9_body = V9_BODY.read_bytes()
    v9_binding_raw = V9_BINDING.read_bytes()
    v10_body = V10_BODY.read_bytes()
    v10_binding_raw = V10_BINDING.read_bytes()
    pins = (
        (v9_body, EXPECTED_V9_BODY_BYTES, EXPECTED_V9_BODY_SHA256, "V9 body"),
        (v9_binding_raw, EXPECTED_V9_BINDING_BYTES, EXPECTED_V9_BINDING_SHA256, "V9 binding"),
        (v10_body, EXPECTED_V10_BODY_BYTES, EXPECTED_V10_BODY_SHA256, "V10 body"),
        (v10_binding_raw, EXPECTED_V10_BINDING_BYTES, EXPECTED_V10_BINDING_SHA256, "V10 binding"),
    )
    for raw, expected_bytes, expected_sha, label in pins:
        if len(raw) != expected_bytes or sha256(raw) != expected_sha:
            raise SystemExit(f"{label} exact pin mismatch")

    v10_text = v10_body.decode("utf-8")
    if v10_text.count(MAIN_LAUNCH_AND_REREAD) != 1:
        raise SystemExit(
            "V10 main must reread immediately after submit_once without an extra block"
        )
    inverse = v10_text.replace(
        MAIN_LAUNCH_AND_REREAD, MAIN_POST + MAIN_LAUNCH_AND_REREAD.split("\n", 1)[1], 1
    )
    success_with_fsync = LAUNCH_WRITE + BLOCK4
    if inverse.count(success_with_fsync) != 1:
        raise SystemExit("V10 successful receipt is not followed by one exact directory fsync")
    inverse = inverse.replace(success_with_fsync, LAUNCH_WRITE, 1)
    if inverse.count(UNRESOLVED_WITH_FSYNC) != 1:
        raise SystemExit("V10 unresolved receipt is not followed by one exact directory fsync")
    inverse = inverse.replace(UNRESOLVED_WITH_FSYNC, UNRESOLVED_END, 1)
    if inverse.encode("utf-8") != v9_body:
        raise SystemExit("inverse did not restore exact frozen V9 body bytes")
    if ast.dump(ast.parse(inverse), include_attributes=False) != ast.dump(
        ast.parse(v9_body), include_attributes=False
    ):
        raise SystemExit("inverse AST differs from frozen V9")

    v9_binding = json.loads(v9_binding_raw)
    v10_binding = json.loads(v10_binding_raw)
    if without_binding_metadata(v9_binding) != without_binding_metadata(v10_binding):
        raise SystemExit("non-version binding metadata changed")
    if v9_binding["scorer_payloads"] != v10_binding["scorer_payloads"]:
        raise SystemExit("the six payload pins changed")
    if len(v10_binding["scorer_payloads"]) != 6:
        raise SystemExit("expected exactly six pinned score payloads")

    v10_ast = ast.parse(v10_body)
    main_node = next(
        node for node in v10_ast.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    submit_node = next(
        node
        for node in v10_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "submit_once"
    )
    main_launch = next(
        index
        for index, node in enumerate(main_node.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "launch" for target in ast.walk(node))
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "submit_once"
    )
    if not (
        isinstance(main_node.body[main_launch + 1], ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "admission_raw"
            for target in ast.walk(main_node.body[main_launch + 1])
        )
    ):
        raise SystemExit("main contains work between submit_once and durable reread")
    if sum(isinstance(node, ast.If) for node in submit_node.body) < 1:
        raise SystemExit("submit_once lost its unresolved-invocation branch")

    proof = {
        "schema": "fg19.first-cell-score-admission-v10-fsync-branches-proof.v1",
        "status": "PASS_EXACT_V9_INVERSE_AND_BOTH_RECEIPT_BRANCH_FSYNC_V10",
        "v9_body": {"bytes": len(v9_body), "sha256": sha256(v9_body)},
        "v10_body": {"bytes": len(v10_body), "sha256": sha256(v10_body)},
        "v9_binding": {"bytes": len(v9_binding_raw), "sha256": sha256(v9_binding_raw)},
        "v10_binding": {"bytes": len(v10_binding_raw), "sha256": sha256(v10_binding_raw)},
        "inverse_exact_v9_body_bytes": True,
        "inverse_ast_equal_v9": True,
        "added_fsync_sites": [
            "immediately after LAUNCH_UNRESOLVED.json create-only write and before unknown-outcome raise",
            "immediately after LAUNCH.json create-only write and before outcome validation/return",
        ],
        "main_post_submit_fsync_removed": True,
        "initial_pre_submit_fsync_retained": True,
        "six_scorer_payload_pins_unchanged": True,
        "scorer_payload_count": len(v10_binding["scorer_payloads"]),
        "worker_identity_and_resource_metadata_unchanged": True,
        "no_worker_or_payload_execution": True,
    }
    proof_bytes = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode()
    write_create_only(PROOF_PATH, proof_bytes)
    print(f"proof_bytes={len(proof_bytes)} proof_sha256={sha256(proof_bytes)}")


if __name__ == "__main__":
    main()
