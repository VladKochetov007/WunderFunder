from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
RACED_V9 = HERE / "ACCEPT_EXPORT_CAPTURE_V9.py"
ANNOUNCED_V9 = HERE / "ACCEPT_EXPORT_CAPTURE_V9_ANNOUNCED_SNAPSHOT.py"
V10_SOURCE = HERE / "ACCEPT_EXPORT_CAPTURE_V10.py"
V9_FIXTURE = HERE / "test_accept_export_capture_v9_two_transports.py"
V10_FIXTURE = HERE / "test_accept_export_capture_v10_two_transports.py"
PROOF = HERE / "ACCEPT_EXPORT_CAPTURE_V10_EXACT_COPY_PROOF.json"

EXPECTED_V9 = (54513, "00e96230c85398f5b9d15f1c2aea6e50e9dc685766e7a507a3b36c25329bbe75")
EXPECTED_RACED_V9 = (
    54963,
    "4a10af92357d4a470aeb0cabf6033bbe21f591f78de4015b664fba8dd1bb4adf",
)
EXPECTED_V9_FIXTURE = (
    10180,
    "d814866f9d7269febfa9da3292f4332ddb98e9fde83b9bac97b5be30c9e288a0",
)


def pin(raw: bytes) -> dict[str, object]:
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def write_new(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> None:
    raced = RACED_V9.read_bytes()
    announced = ANNOUNCED_V9.read_bytes()
    v10 = V10_SOURCE.read_bytes()
    v9_fixture = V9_FIXTURE.read_bytes()
    v10_fixture = V10_FIXTURE.read_bytes()
    require(
        pin(announced) == dict(zip(("bytes", "sha256"), EXPECTED_V9, strict=True)),
        "announced V9 pin did not recover exactly",
    )
    require(
        pin(raced) == dict(zip(("bytes", "sha256"), EXPECTED_RACED_V9, strict=True)),
        "raced V9 bytes differ from the preserved race note",
    )
    require(
        v10 == announced, "V10 executable is not an exact copy of the recovered announced V9 bytes"
    )
    require(
        pin(v9_fixture) == dict(zip(("bytes", "sha256"), EXPECTED_V9_FIXTURE, strict=True)),
        "V9 focused fixture pin differs",
    )

    # The failed announcement race was one known, three-site stderr edit. Reverse
    # only those exact additions and require the previously announced SHA.
    reverted = raced
    edits = (
        (
            b'    if (\n        type(launch_transport.get("stderr_bytes")) is not int\n'
            b'        or (launch_transport.get("stderr_bytes"), launch_transport.get("stderr_sha256"))\n'
            b"        != (len(launch_stderr_raw), sha(launch_stderr_raw))\n"
            b'    ):\n        raise ValueError("launch transport does not bind its separately persisted stderr")\n',
            b"",
        ),
        (
            b'    launch_stderr_raw = stable_file(args.record_root / "STDERR.raw.txt", 4_000_000)[0]\n',
            b"",
        ),
        (b"        launch_stderr_raw=launch_stderr_raw,\n", b""),
    )
    for before, after in edits:
        require(reverted.count(before) == 1, "known race edit was not uniquely recoverable")
        reverted = reverted.replace(before, after, 1)
    require(
        reverted == announced, "inverse of the three announced-race edits did not recover V9 bytes"
    )

    changed_filename = b"ACCEPT_EXPORT_CAPTURE_V9.py"
    replacement_filename = b"ACCEPT_EXPORT_CAPTURE_V10.py"
    require(v9_fixture.count(changed_filename) == 1, "V9 fixture import path is not unique")
    require(
        v10_fixture == v9_fixture.replace(changed_filename, replacement_filename, 1),
        "V10 fixture changes beyond the single import filename",
    )
    original_ast = ast.parse(v9_fixture.decode("utf-8"))
    candidate_ast = ast.parse(v10_fixture.decode("utf-8"))
    for node in ast.walk(candidate_ast):
        if isinstance(node, ast.Constant) and node.value == "ACCEPT_EXPORT_CAPTURE_V10.py":
            node.value = "ACCEPT_EXPORT_CAPTURE_V9.py"
    require(
        ast.dump(original_ast, include_attributes=False)
        == ast.dump(candidate_ast, include_attributes=False),
        "V10 fixture AST differs beyond the import filename",
    )

    proof = {
        "schema": "fg19.first-cell-export-acceptor-v10-exact-copy-proof.v1",
        "status": "PASS_EXACT_V9_RECOVERY_V10_COPY_AND_FIXTURE_IMPORT_FORK",
        "scope": "Static byte/AST proof only; no production imports, behavior fixture replay, model/data/payload access, or B action.",
        "raced_v9": {"path": RACED_V9.name, **pin(raced)},
        "recovered_announced_v9": {"path": ANNOUNCED_V9.name, **pin(announced)},
        "v10_source": {"path": V10_SOURCE.name, **pin(v10)},
        "inverse_race_edit_count": len(edits),
        "inverse_restores_announced_v9_exact_bytes": True,
        "v10_is_exact_recovered_v9_copy": True,
        "v9_fixture": {"path": V9_FIXTURE.name, **pin(v9_fixture)},
        "v10_fixture": {"path": V10_FIXTURE.name, **pin(v10_fixture)},
        "fixture_change": "Only the imported source filename changes from ACCEPT_EXPORT_CAPTURE_V9.py to ACCEPT_EXPORT_CAPTURE_V10.py.",
        "fixture_inverse_restores_exact_v9_bytes": True,
        "fixture_ast_identical_after_filename_normalization": True,
        "behavior_fixtures_reexecuted": False,
        "source_imports_performed": False,
        "payloads_read_or_hashed": False,
        "remote_action_performed": False,
    }
    write_new(PROOF, (json.dumps(proof, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
