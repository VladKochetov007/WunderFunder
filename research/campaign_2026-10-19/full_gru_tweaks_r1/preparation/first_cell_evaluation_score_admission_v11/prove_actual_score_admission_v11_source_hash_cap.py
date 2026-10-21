from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V10_DIR = ROOT / "preparation/first_cell_evaluation_score_admission_v10"
V11_DIR = Path(__file__).resolve().parent
V10_BODY = V10_DIR / "REMOTE_SUBMISSION_V10.py"
V10_BINDING = V10_DIR / "SCORE_ADMISSION_SOURCE_BINDING_V10.json"
V11_BODY = V11_DIR / "REMOTE_SUBMISSION_V11.py"
V11_BINDING = V11_DIR / "SCORE_ADMISSION_SOURCE_BINDING_V11.json"
PROOF_PATH = V11_DIR / "ACTUAL_SCORE_ADMISSION_V11_SOURCE_HASH_CAP_PROOF.json"
V10_BODY_BYTES = 187_989
V10_BODY_SHA256 = "24787cf037328baabebe19d4eb6ba5c452740d2b5648dbf478544e3ce33cbd53"
V10_BINDING_BYTES = 5_602
V10_BINDING_SHA256 = "9767f437d545ee9a6b448a6606be972ea6a7766ea3efebf24cbbb346df56a42e"
V11_BODY_BYTES = 188_823
V11_BODY_SHA256 = "9dc1f04741609cb4e80e84308439187f5bef5d82d2808057d92fe11eca4379d5"
V11_BINDING_BYTES = 5_554
V11_BINDING_SHA256 = "ff325182837288d97fb3b1f705afef4b6c6cc5198950f7db21a9f55d53d01e1f"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def function_source(source: str, name: str) -> str:
    start = source.index(f"def {name}(")
    end = source.index("\ndef ", start + 1)
    return source[start:end]


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise SystemExit(f"expected one {label}, found {source.count(old)}")
    return source.replace(old, new, 1)


def without_binding_metadata(binding: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in binding.items()
        if key not in {"schema", "scope", "rendered_source"}
    }


def write_create_only(path: Path, raw: bytes) -> None:
    if path.exists():
        if path.read_bytes() != raw:
            raise SystemExit(f"existing proof output differs; refusing overwrite {path}")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def main() -> None:
    v10_raw, v11_raw = V10_BODY.read_bytes(), V11_BODY.read_bytes()
    v10_binding_raw, v11_binding_raw = V10_BINDING.read_bytes(), V11_BINDING.read_bytes()
    for raw, expected_bytes, expected_sha, label in (
        (v10_raw, V10_BODY_BYTES, V10_BODY_SHA256, "V10 body"),
        (v10_binding_raw, V10_BINDING_BYTES, V10_BINDING_SHA256, "V10 binding"),
        (v11_raw, V11_BODY_BYTES, V11_BODY_SHA256, "V11 body"),
        (v11_binding_raw, V11_BINDING_BYTES, V11_BINDING_SHA256, "V11 binding"),
    ):
        if len(raw) != expected_bytes or sha256(raw) != expected_sha:
            raise SystemExit(f"{label} exact pin mismatch")

    v10_source, v11_source = v10_raw.decode(), v11_raw.decode()
    inverse = replace_once(
        v11_source,
        function_source(v11_source, "file_hash"),
        function_source(v10_source, "file_hash"),
        "changed file_hash function",
    )
    inverse = replace_once(
        inverse,
        function_source(v11_source, "check_source_tree"),
        function_source(v10_source, "check_source_tree"),
        "changed check_source_tree function",
    )
    if inverse.encode() != v10_raw:
        raise SystemExit("inverse did not restore exact frozen V10 body")
    if ast.dump(ast.parse(inverse), include_attributes=False) != ast.dump(
        ast.parse(v10_raw), include_attributes=False
    ):
        raise SystemExit("inverse AST differs from V10")

    v10_binding = json.loads(v10_binding_raw)
    v11_binding = json.loads(v11_binding_raw)
    if without_binding_metadata(v10_binding) != without_binding_metadata(v11_binding):
        raise SystemExit("non-versioned binding metadata changed")
    if v10_binding["scorer_payloads"] != v11_binding["scorer_payloads"]:
        raise SystemExit("six scorer payload pins changed")
    if len(v11_binding["scorer_payloads"]) != 6:
        raise SystemExit("expected six unchanged scorer payload pins")

    module = ast.parse(v11_raw)
    file_hash_fn = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "file_hash"
    )
    file_hash_source = ast.unparse(file_hash_fn)
    if "cap - bytes_read + 1" not in file_hash_source:
        raise SystemExit("hash read size is not bounded to remaining cap plus one sentinel byte")
    if "bytes_read <= cap" not in file_hash_source:
        raise SystemExit("hash read loop does not reject over-cap bytes before hashing")
    if file_hash_source.index("bytes_read <= cap") > file_hash_source.index("hasher.update"):
        raise SystemExit("hash loop updates the digest before enforcing its cap")
    check_fn = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "check_source_tree"
    )
    check_source = ast.unparse(check_fn)
    if check_source.index("expected_total_bytes") > check_source.index("file_hash("):
        raise SystemExit("manifest sizes are not summed before source hashing")
    if "expected_total_bytes <= MAX_SOURCE_AGGREGATE_BYTES" not in check_source:
        raise SystemExit("manifest aggregate cap is missing")
    if "live_size == expected_bytes" not in check_source:
        raise SystemExit("live source size is not checked before open")
    if "min(expected_bytes, remaining_source_bytes)" not in check_source:
        raise SystemExit("file_hash does not receive expected and remaining aggregate limits")

    proof = {
        "schema": "fg19.first-cell-score-admission-v11-source-hash-cap-proof.v1",
        "status": "PASS_EXACT_V10_INVERSE_AND_INCREMENTAL_SOURCE_HASH_CAP_V11",
        "v10_body": {"bytes": len(v10_raw), "sha256": sha256(v10_raw)},
        "v11_body": {"bytes": len(v11_raw), "sha256": sha256(v11_raw)},
        "v10_binding": {"bytes": len(v10_binding_raw), "sha256": sha256(v10_binding_raw)},
        "v11_binding": {"bytes": len(v11_binding_raw), "sha256": sha256(v11_binding_raw)},
        "inverse_exact_v10_body_bytes": True,
        "inverse_ast_equal_v10": True,
        "source_manifest_size_sum_precedes_hashing": True,
        "live_file_size_checked_before_open": True,
        "per_file_cap_is_expected_size_and_remaining_aggregate": True,
        "read_loop_requests_at_most_remaining_plus_one_sentinel": True,
        "over_cap_chunk_rejected_before_hash_update": True,
        "six_scorer_payload_pins_unchanged": True,
        "scorer_payload_count": len(v11_binding["scorer_payloads"]),
        "launch_receipt_fsyncs_unchanged_from_v10": True,
        "no_worker_or_payload_execution": True,
    }
    proof_raw = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode()
    write_create_only(PROOF_PATH, proof_raw)
    print(f"proof_bytes={len(proof_raw)} proof_sha256={sha256(proof_raw)}")


if __name__ == "__main__":
    main()
