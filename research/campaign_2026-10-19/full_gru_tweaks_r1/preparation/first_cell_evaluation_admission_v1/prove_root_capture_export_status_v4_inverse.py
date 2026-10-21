from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
V3 = HERE / "ROOT_CAPTURE_EXPORT_STATUS_V3.py"
V4 = HERE / "ROOT_CAPTURE_EXPORT_STATUS_V4.py"
PROOF = HERE / "ROOT_CAPTURE_EXPORT_STATUS_V4_INVERSE_PROOF.json"
V3_PIN = (7998, "806e5d0a88cccd772e107f0e8dbc4f50b1f7853a0bac61891647d002ca495aef")
NEW_WRITE = '''def fsync_directory(path: Path) -> None:
    """Persist directory entries after create-only capture writes."""
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write(path: Path, raw: bytes) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError(f"capture parent is not an enumerated real directory: {path.parent}")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def write_terminal_metadata(record: Path, terminal: dict[str, object]) -> list[str]:
    """Persist only the manifest and five fixed sidecars under enumerated arm dirs."""
    expected = {"EXPORT_MANIFEST_V1.json"} | {
        f"{arm}/PREDICTION_SIDECAR.json" for arm in ("control", "ema", "sparse", "nll", "allthree")
    }
    if not isinstance(terminal, dict) or set(terminal) != expected:
        raise ValueError("terminal metadata must contain the manifest and five fixed sidecars")
    decoded = {}
    for name in sorted(expected):
        item = terminal[name]
        if not isinstance(item, dict):
            raise ValueError("terminal metadata capture row is not an object")
        raw = base64.b64decode(item["base64"], validate=True)
        if type(item.get("bytes")) is not int or (len(raw), sha(raw)) != (
            item["bytes"],
            item["sha256"],
        ):
            raise ValueError("strict terminal metadata join differs")
        decoded[name] = raw
    arms = ("control", "ema", "sparse", "nll", "allthree")
    for arm in arms:
        os.mkdir(record / arm, 0o700)
    fsync_directory(record)
    write(record / "EXPORT_MANIFEST_V1.json", decoded["EXPORT_MANIFEST_V1.json"])
    for arm in arms:
        arm_dir = record / arm
        write(arm_dir / "PREDICTION_SIDECAR.json", decoded[f"{arm}/PREDICTION_SIDECAR.json"])
        fsync_directory(arm_dir)
    fsync_directory(record)
    return sorted(expected)
'''
OLD_WRITE = """def write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
"""
NEW_CREATE = """    record = args.record / args.tag
    record.mkdir(exist_ok=False)
    fsync_directory(record.parent)
    write(record / "REMOTE.py", body)
"""
OLD_CREATE = """    record = args.record / args.tag
    record.mkdir(exist_ok=False)
    write(record / "REMOTE.py", body)
"""
NEW_CAPTURE = """    metadata_files = write_terminal_metadata(record, observation["terminal_metadata"])
    summary = {
        "schema": observation["schema"],
        "captured_utc": observation["captured_utc"],
        "unit": observation["unit"],
        "invocation_id": observation["invocation_id"],
        "systemd": observation["systemd"],
        "terminal_metadata_files": metadata_files,
        "array_payloads_read_or_hashed": False,
        "transport": transport,
    }
    write(record / "OBSERVATION.json", (json.dumps(summary, indent=2) + "\\n").encode())
    fsync_directory(record)
    print(json.dumps(summary))
"""
OLD_CAPTURE = """    for name, item in observation["terminal_metadata"].items():
        if name not in {"EXPORT_MANIFEST_V1.json"} | {
            f"{arm}/PREDICTION_SIDECAR.json"
            for arm in ("control", "ema", "sparse", "nll", "allthree")
        }:
            raise ValueError("unexpected terminal metadata filename")
        raw = base64.b64decode(item["base64"], validate=True)
        if (len(raw), sha(raw)) != (item["bytes"], item["sha256"]):
            raise ValueError("strict terminal metadata join differs")
        write(record / name, raw)
    summary = {
        "schema": observation["schema"],
        "captured_utc": observation["captured_utc"],
        "unit": observation["unit"],
        "invocation_id": observation["invocation_id"],
        "systemd": observation["systemd"],
        "terminal_metadata_files": list(observation["terminal_metadata"]),
        "array_payloads_read_or_hashed": False,
        "transport": transport,
    }
    write(record / "OBSERVATION.json", (json.dumps(summary, indent=2) + "\\n").encode())
    print(json.dumps(summary))
"""


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError("inverse patch anchor is not unique")
    return source.replace(before, after, 1)


def main() -> None:
    v3_raw, v4_raw = V3.read_bytes(), V4.read_bytes()
    if (len(v3_raw), hashlib.sha256(v3_raw).hexdigest()) != V3_PIN:
        raise ValueError("frozen V3 base pin differs")
    v4_text = v4_raw.decode("utf-8")
    inverse = replace_once(v4_text, NEW_WRITE, OLD_WRITE)
    inverse = replace_once(inverse, NEW_CREATE, OLD_CREATE)
    inverse = replace_once(inverse, NEW_CAPTURE, OLD_CAPTURE)
    if inverse.encode("utf-8") != v3_raw:
        raise ValueError("V4 inverse does not restore the exact frozen V3 bytes")
    proof = {
        "schema": "fg19.first-cell-export-status-v4-inverse-proof.v1",
        "status": "PASS_EXACT_V3_INVERSE_WITH_ENUMERATED_DIRECTORY_DURABILITY_FIX",
        "base": {
            "path": str(V3.relative_to(HERE.parents[4])),
            "bytes": len(v3_raw),
            "sha256": V3_PIN[1],
        },
        "candidate": {
            "path": str(V4.relative_to(HERE.parents[4])),
            "bytes": len(v4_raw),
            "sha256": hashlib.sha256(v4_raw).hexdigest(),
        },
        "inverse_bytes_identical": True,
        "changed_scope": [
            "remove generic parent.mkdir from create-only writer",
            "create only the exact five arm directories after exact metadata membership validation",
            "fsync the record parent, record directory, each arm directory, and final record directory",
        ],
        "remote_action": False,
    }
    if PROOF.exists():
        raise FileExistsError(PROOF)
    PROOF.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
