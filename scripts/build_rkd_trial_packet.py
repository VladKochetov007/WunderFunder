"""Build a write-once, externally reviewed RKD trial launch packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

RECEIPT_VERSION = 1
RECEIPT_STATUS = "approved_for_bounded_kaggle_experiment"
RECEIPT_HASH_FIELDS = {
    "actor_code_sha256",
    "protocol_sha256",
    "input_manifest_sha256",
    "profile_receipt_sha256",
    "approved_commands_sha256",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 of exact file bytes without loading the file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_file(path_arg: str | Path, project_root: Path) -> tuple[Path, str]:
    """Resolve an existing non-symlink file under the project root."""
    candidate = Path(path_arg)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = Path(os.path.abspath(candidate))

    root = project_root.resolve(strict=True)
    cursor = candidate
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValueError(f"symlink paths are not allowed: {path_arg}")
        if cursor == root:
            break
        cursor = cursor.parent

    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must be inside project root: {path_arg}") from exc
    if not resolved.is_file():
        raise ValueError(f"expected a regular file: {path_arg}")
    if resolved.stat().st_size == 0:
        raise ValueError(f"expected a nonempty file: {path_arg}")
    return resolved, relative.as_posix()


def _code_digest(code_args: list[str | Path], project_root: Path) -> tuple[str, list[str]]:
    """Hash selected Python source files with unambiguous relative-name framing."""
    if not code_args:
        raise ValueError("at least one --actor-code Python file is required")
    entries: dict[str, Path] = {}
    for path_arg in code_args:
        path, relative = _project_file(path_arg, project_root)
        if path.suffix != ".py":
            raise ValueError(f"actor code must be a .py file: {relative}")
        if relative in entries:
            raise ValueError(f"duplicate actor code path: {relative}")
        entries[relative] = path

    digest = hashlib.sha256()
    ordered = sorted(entries)
    for relative in ordered:
        path = entries[relative]
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest(), ordered


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("controller review receipt must be valid UTF-8 JSON") from exc
    if not isinstance(receipt, dict):
        raise ValueError("controller review receipt must be a JSON object")
    return receipt


def _validate_receipt(receipt: dict[str, Any], expected_hashes: dict[str, str]) -> dict[str, Any]:
    if type(receipt.get("version")) is not int or receipt["version"] != RECEIPT_VERSION:
        raise ValueError("controller review receipt version must be 1")
    if receipt.get("status") != RECEIPT_STATUS:
        raise ValueError(f"controller review receipt status must be {RECEIPT_STATUS!r}")

    for field in sorted(RECEIPT_HASH_FIELDS):
        value = receipt.get(field)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise ValueError(f"controller review receipt requires a lowercase SHA-256: {field}")
        if value != expected_hashes[field]:
            raise ValueError(f"controller review receipt hash does not match {field}")

    limits = receipt.get("resource_limits")
    if not isinstance(limits, dict):
        raise ValueError("controller review receipt requires resource_limits")
    provider_seconds = limits.get("provider_time_seconds")
    if (
        isinstance(provider_seconds, bool)
        or not isinstance(provider_seconds, (int, float))
        or not math.isfinite(provider_seconds)
        or provider_seconds <= 0
    ):
        raise ValueError("resource_limits.provider_time_seconds must be finite and positive")
    host_ram = limits.get("host_ram_bytes")
    if isinstance(host_ram, bool) or not isinstance(host_ram, int) or host_ram <= 0:
        raise ValueError("resource_limits.host_ram_bytes must be a positive integer")
    vram_fraction = limits.get("vram_headroom_fraction")
    if (
        isinstance(vram_fraction, bool)
        or not isinstance(vram_fraction, (int, float))
        or not math.isfinite(vram_fraction)
        or not 0 < vram_fraction <= 1
    ):
        raise ValueError("resource_limits.vram_headroom_fraction must be in (0, 1]")
    reviewed_at = receipt.get("controller_reviewed_at")
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise ValueError("controller_reviewed_at must be a nonempty string")
    return limits


def build_trial_packet(
    *,
    project_root: Path,
    actor_code: list[str | Path],
    protocol: str | Path,
    input_manifest: str | Path,
    profile_receipt: str | Path,
    approved_commands: str | Path,
    review_receipt: str | Path,
) -> dict[str, Any]:
    """Validate externally reviewed assets and return an immutable launch packet."""
    root = project_root.resolve(strict=True)
    actor_code_sha256, actor_code_paths = _code_digest(actor_code, root)
    protocol_path, protocol_rel = _project_file(protocol, root)
    input_path, input_rel = _project_file(input_manifest, root)
    profile_path, profile_rel = _project_file(profile_receipt, root)
    commands_path, commands_rel = _project_file(approved_commands, root)
    review_path, review_rel = _project_file(review_receipt, root)

    expected_hashes = {
        "actor_code_sha256": actor_code_sha256,
        "protocol_sha256": _sha256_file(protocol_path),
        "input_manifest_sha256": _sha256_file(input_path),
        "profile_receipt_sha256": _sha256_file(profile_path),
        "approved_commands_sha256": _sha256_file(commands_path),
    }
    receipt = _load_receipt(review_path)
    limits = _validate_receipt(receipt, expected_hashes)

    return {
        "schema_version": 1,
        "status": "ready_for_controller_gated_kaggle_launch",
        "review_receipt_path": review_rel,
        "review_receipt_sha256": _sha256_file(review_path),
        "review_receipt": receipt,
        "assets": {
            "actor_code_paths": actor_code_paths,
            **expected_hashes,
            "protocol_path": protocol_rel,
            "input_manifest_path": input_rel,
            "profile_receipt_path": profile_rel,
            "approved_commands_path": commands_rel,
        },
        "resource_limits": limits,
        "launch_performed": False,
    }


def write_packet_once(path_arg: str | Path, packet: dict[str, Any], project_root: Path) -> None:
    """Atomically create the packet without replacing an existing path."""
    root = project_root.resolve(strict=True)
    target = Path(path_arg)
    if not target.is_absolute():
        target = root / target
    target = Path(os.path.abspath(target))
    parent = target.parent.resolve(strict=True)
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise ValueError("packet output must be inside project root") from exc
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing packet: {target}")

    payload = (json.dumps(packet, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=parent, prefix=".rkd-packet-", delete=False) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, target)
        os.unlink(temporary_name)
        temporary_name = None
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            os.unlink(temporary_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--actor-code", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--profile-receipt", type=Path, required=True)
    parser.add_argument("--approved-commands", type=Path, required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        packet = build_trial_packet(
            project_root=args.project_root,
            actor_code=args.actor_code,
            protocol=args.protocol,
            input_manifest=args.input_manifest,
            profile_receipt=args.profile_receipt,
            approved_commands=args.approved_commands,
            review_receipt=args.review_receipt,
        )
        write_packet_once(args.output, packet, args.project_root)
    except (OSError, ValueError) as exc:
        print(f"refusing to create trial packet: {exc}", file=sys.stderr)
        return 2
    print(f"wrote reviewed launch packet: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
