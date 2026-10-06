"""Synthetic-only tests for the controller receipt gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_rkd_trial_packet import (
    RECEIPT_STATUS,
    _code_digest,
    _sha256_file,
    build_trial_packet,
    write_packet_once,
)


def _synthetic_inputs(root: Path) -> dict[str, Path]:
    (root / "src").mkdir(parents=True)
    (root / "local").mkdir()
    actor = root / "src" / "actor.py"
    actor.write_text("def feature():\n    return 0.0\n", encoding="utf-8")
    protocol = root / "protocol.md"
    protocol.write_text("synthetic protocol\n", encoding="utf-8")
    manifest = root / "local" / "inputs.json"
    manifest.write_text('{"synthetic": true}\n', encoding="utf-8")
    profile = root / "local" / "profile.json"
    profile.write_text('{"synthetic": true, "profiled": false}\n', encoding="utf-8")
    commands = root / "local" / "commands.txt"
    commands.write_text("synthetic command (not executable)\n", encoding="utf-8")
    receipt = root / "local" / "review.json"
    actor_sha, _ = _code_digest([actor], root)
    review = {
        "version": 1,
        "status": RECEIPT_STATUS,
        "actor_code_sha256": actor_sha,
        "protocol_sha256": _sha256_file(protocol),
        "input_manifest_sha256": _sha256_file(manifest),
        "profile_receipt_sha256": _sha256_file(profile),
        "approved_commands_sha256": _sha256_file(commands),
        "resource_limits": {
            "provider_time_seconds": 180,
            "host_ram_bytes": 2_000_000_000,
            "vram_headroom_fraction": 0.2,
        },
        "controller_reviewed_at": "synthetic-test-only",
        "synthetic": True,
    }
    receipt.write_text(json.dumps(review), encoding="utf-8")
    return {
        "root": root,
        "actor": actor,
        "protocol": protocol,
        "manifest": manifest,
        "profile": profile,
        "commands": commands,
        "review": receipt,
    }


def _build(paths: dict[str, Path]) -> dict[str, object]:
    return build_trial_packet(
        project_root=paths["root"],
        actor_code=[paths["actor"]],
        protocol=paths["protocol"],
        input_manifest=paths["manifest"],
        profile_receipt=paths["profile"],
        approved_commands=paths["commands"],
        review_receipt=paths["review"],
    )


def test_matching_synthetic_receipt_creates_write_once_packet(tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path)
    packet = _build(paths)
    assert packet["launch_performed"] is False
    assert packet["review_receipt"]["synthetic"] is True  # type: ignore[index]

    output = tmp_path / "local" / "trial-packet.json"
    write_packet_once(output, packet, tmp_path)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_packet_once(output, packet, tmp_path)
    assert output.read_bytes() == original


def test_hash_mismatch_fails_before_packet_creation(tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path)
    review = json.loads(paths["review"].read_text(encoding="utf-8"))
    review["actor_code_sha256"] = "0" * 64
    paths["review"].write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match actor_code_sha256"):
        _build(paths)
    assert not (tmp_path / "local" / "trial-packet.json").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "approved", "status must be"),
        ("version", True, "version must be 1"),
        ("controller_reviewed_at", "   ", "nonempty string"),
        ("resource_limits", {}, "provider_time_seconds"),
    ],
)
def test_malformed_receipt_fails_closed(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    paths = _synthetic_inputs(tmp_path)
    review = json.loads(paths["review"].read_text(encoding="utf-8"))
    review[field] = value
    paths["review"].write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _build(paths)


def test_external_or_symlinked_asset_is_rejected(tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path / "project")
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")
    link = paths["root"] / "src" / "linked.py"
    link.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink paths are not allowed"):
        build_trial_packet(
            project_root=paths["root"],
            actor_code=[link],
            protocol=paths["protocol"],
            input_manifest=paths["manifest"],
            profile_receipt=paths["profile"],
            approved_commands=paths["commands"],
            review_receipt=paths["review"],
        )


def test_controller_receipt_may_be_supplied_from_outside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    paths = _synthetic_inputs(project)
    external_receipt = tmp_path / "controller-review.json"
    external_receipt.write_bytes(paths["review"].read_bytes())
    paths["review"] = external_receipt

    packet = _build(paths)
    assert packet["review_receipt_path"] == str(external_receipt.resolve())
