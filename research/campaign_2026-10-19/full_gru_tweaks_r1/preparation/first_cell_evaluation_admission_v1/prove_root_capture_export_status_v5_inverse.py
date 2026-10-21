from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
V4 = HERE / "ROOT_CAPTURE_EXPORT_STATUS_V4.py"
V5 = HERE / "ROOT_CAPTURE_EXPORT_STATUS_V5.py"
PROOF = HERE / "ROOT_CAPTURE_EXPORT_STATUS_V5_INVERSE_PROOF.json"
V4_PIN = (9456, "af914d1dec67853768569237f8500cad75db4d0fe2d0f8e1e7cc55553c77e040")
OLD_HELPER = '''def write_terminal_metadata(record: Path, terminal: dict[str, object]) -> list[str]:
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
NEW_HELPER = '''def is_terminal_success(systemd: object) -> bool:
    if not isinstance(systemd, dict):
        return False
    return (
        systemd.get("ActiveState"),
        systemd.get("SubState"),
        systemd.get("MainPID"),
        systemd.get("ExecMainCode"),
        systemd.get("ExecMainStatus"),
        systemd.get("Result"),
    ) == ("active", "exited", "0", "1", "0", "success")


def write_terminal_metadata(
    record: Path,
    terminal: dict[str, object],
    *,
    terminal_success: bool,
) -> list[str]:
    """Persist fixed terminal metadata, or accept its absence before success."""
    if type(terminal_success) is not bool:
        raise ValueError("terminal success classification must be a strict boolean")
    if not isinstance(terminal, dict):
        raise ValueError("terminal metadata must be an object")
    if not terminal_success:
        if terminal:
            raise ValueError("nonterminal observations must not contain terminal metadata")
        return []
    expected = {"EXPORT_MANIFEST_V1.json"} | {
        f"{arm}/PREDICTION_SIDECAR.json" for arm in ("control", "ema", "sparse", "nll", "allthree")
    }
    if set(terminal) != expected:
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
OLD_CALL = (
    '    metadata_files = write_terminal_metadata(record, observation["terminal_metadata"])\n'
)
NEW_CALL = """    systemd = observation["systemd"]
    terminal_success = is_terminal_success(systemd)
    metadata_files = write_terminal_metadata(
        record,
        observation["terminal_metadata"],
        terminal_success=terminal_success,
    )
"""


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError("inverse anchor is not unique")
    return source.replace(before, after, 1)


def main() -> None:
    v4_raw = V4.read_bytes()
    v5_raw = V5.read_bytes()
    if (len(v4_raw), hashlib.sha256(v4_raw).hexdigest()) != V4_PIN:
        raise ValueError("frozen V4 base pin differs")
    inverse = v5_raw.decode("utf-8")
    inverse = replace_once(inverse, NEW_HELPER, OLD_HELPER)
    inverse = replace_once(inverse, NEW_CALL, OLD_CALL)
    if inverse.encode("utf-8") != v4_raw:
        raise ValueError("V5 inverse does not restore exact frozen V4 bytes")
    proof = {
        "schema": "fg19.first-cell-export-status-v5-inverse-proof.v1",
        "status": "PASS_EXACT_V4_INVERSE_WITH_NONTERMINAL_EMPTY_METADATA_FIX",
        "base": {
            "path": str(V4.relative_to(HERE.parents[4])),
            "bytes": len(v4_raw),
            "sha256": hashlib.sha256(v4_raw).hexdigest(),
        },
        "candidate": {
            "path": str(V5.relative_to(HERE.parents[4])),
            "bytes": len(v5_raw),
            "sha256": hashlib.sha256(v5_raw).hexdigest(),
        },
        "inverse_restores_exact_base_bytes": True,
        "change_scope": [
            "allow empty terminal metadata only for nonterminal observations",
            "require exact six-file metadata set for terminal success",
            "reject partial metadata before creating arm directories",
        ],
    }
    PROOF.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proof, sort_keys=True))


if __name__ == "__main__":
    main()
