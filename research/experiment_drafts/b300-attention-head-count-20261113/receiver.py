"""Qualify a frozen four-cell panel and run one stage through native PR75 entrypoints."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, load_campaign_split
from wnn.training.sequence_config import load_config
from wnn.training.sequence_provenance import (
    canonical_sha256,
    file_sha256,
    source_identity,
    validate_profile_receipt,
)

PACKET = Path(__file__).resolve().parent
RUNTIME = Path(os.environ["WF_RUNTIME"]).resolve()
RUNS = Path(os.environ["WF_RUNS"]).resolve()
DATA = Path(os.environ["WF_DATA"]).resolve()
SPLIT = Path(os.environ["WF_SPLIT"]).resolve()
OWNER = os.environ["WF_OWNER"].strip()
PIN = "8626000a4b965763a70011a0291a891aa6535d7e"
SPLIT_HASH = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
BASE = Path("research/experiment_drafts/transformer-qk-normalization-20261109")
CELLS = {f"heads{h}_s{s}": (h, s) for h in (12, 6) for s in (207301, 207302)}
# Inputs, two-output head, final norm and 24 blocks with two norms, QKV, output and FFN.
WIDTH = 768
PARAMETERS = (112 + 1 + 2 + 2) * WIDTH + 2 + 24 * (12 * WIDTH**2 + 13 * WIDTH)


def read(path: Path) -> dict[str, Any]:
    """Read producer metadata from the receiver's artifact tree."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular metadata file: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def publish(path: Path, value: dict[str, Any], *, approve: bool = False) -> None:
    """Validate and publish a complete receipt, preserving every previous output."""
    with NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        if approve:
            validate_profile_receipt(Path(stream.name), value["identity"])
        os.link(stream.name, path)


def git(directory: Path, *args: str) -> str:
    """Read a checkout identity or its working-tree status."""
    return subprocess.check_output(["git", "-C", directory, *args], text=True).strip()


def qualify() -> dict[str, Any]:
    """Match the packet, pinned native code, original split and all four configs."""
    packet_git = git(PACKET, "rev-parse", "HEAD")
    if git(PACKET, "status", "--porcelain", "--untracked-files=all", "--", "."):
        raise ValueError("packet must be clean and published")
    if git(RUNTIME, "rev-parse", "HEAD") != PIN or git(
        RUNTIME,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "src",
        "scripts",
        "utils.py",
        str(BASE / "configs/plain_s203301.json"),
    ):
        raise ValueError("native runtime or control recipe differs from PR75")
    split = load_campaign_split(SPLIT, DATA)
    if (
        split.schema,
        split.sha256,
        len(split.train_groups),
        len(split.valid_groups),
        len(split.evaluation_groups),
    ) != (CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, SPLIT_HASH, 10_325, 0, 1_873):
        raise ValueError("wrong chronological train/valid membership")
    recipes = {}
    for cell, (heads, seed) in CELLS.items():
        expected = read(RUNTIME / BASE / "configs/plain_s203301.json")
        expected["seed"] = seed
        expected["model"]["heads"] = heads
        filename = PACKET / "configs" / f"{cell}.json"
        if read(filename) != expected:
            raise ValueError(f"{cell}: deviation from PR75 outside head count and seed")
        recipes[cell] = canonical_sha256(load_config(filename))
    return {
        "schema": "wnn.head-count.qualification.v1",
        "packet_commit": packet_git,
        "runtime_commit": PIN,
        "source_sha256": source_identity(RUNTIME, RUNTIME / "scripts/train_sequence.py")["sha256"],
        "data_root": str(DATA),
        "split_sha256": split.sha256,
        "fit_train": 10_325,
        "fit_valid": 0,
        "evaluation_valid": 1_873,
        "configs": recipes,
    }


def qualified_profile(cell: str, qualification: dict[str, Any]) -> dict[str, Any]:
    """Bind the original 50-update profile and strict reload to its configuration."""
    profile = read(RUNS / cell / "profile/profile_receipt.json")
    identity = profile["identity"]
    reload = profile["engineering_reload"]
    if (
        profile.get("approved") is not False
        or profile.get("updates") != 50
        or profile.get("status") != "profile_complete"
        or profile.get("errors") != []
        or identity["config_sha256"] != qualification["configs"][cell]
        or identity["source_sha256"] != qualification["source_sha256"]
        or identity["data"]["split_sha256"] != SPLIT_HASH
        or identity["world_size"] != 2
        or identity["model_parameters"] != {"trainable": PARAMETERS, "total": PARAMETERS}
        or reload["status"] != "passed"
        or [r["rank"] for r in reload["by_rank"]] != [0, 1]
        or any(r["model_optimizer_scheduler_rng_equal"] is not True for r in reload["by_rank"])
    ):
        raise ValueError(f"{cell}: failed profile, reload, membership or parameter contract")
    return profile


def budget(name: str) -> float:
    """Read the receiver's positive finite memory ceiling in GiB."""
    value = float(os.environ[name])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def approve_profile(cell: str, profile: dict[str, Any]) -> None:
    """Record the review and enforce actual host and per-rank GPU measurements."""
    review = os.environ["WF_REVIEW_NOTE"].strip()
    host, vram = budget("WF_MAX_HOST_GIB"), budget("WF_MAX_VRAM_GIB")
    ram = sum(rank["host_peak_rss_bytes"] for rank in profile["memory_by_rank"])
    if not review or ram > host * 2**30 or profile["peak_reserved_bytes"] > vram * 2**30:
        raise ValueError("resource limits exceeded or the review note is empty")
    approved = dict(
        profile,
        approved=True,
        reviewer=OWNER,
        approval_note=review,
        reviewed_at=dt.datetime.now(dt.UTC).isoformat(),
        max_host_gib=host,
        max_vram_gib=vram,
        profile_sha256=file_sha256(RUNS / cell / "profile/profile_receipt.json"),
    )
    publish(RUNS / cell / "approved_profile.json", approved, approve=True)


def approval(cell: str, profile: dict[str, Any]) -> Path:
    """Require the saved reviewer decision to match the original qualified receipt."""
    run = RUNS / cell
    path = run / "approved_profile.json"
    approved = read(path)
    expected = dict(profile, approved=True, approval_note=approved["approval_note"])
    if (
        not approved["reviewer"].strip()
        or not approved["approval_note"].strip()
        or approved["profile_sha256"] != file_sha256(run / "profile/profile_receipt.json")
        or any(approved.get(k) != v for k, v in expected.items())
    ):
        raise ValueError("approval does not match the original profile")
    validate_profile_receipt(path, profile["identity"])
    return path


def main(stage: str, cell: str) -> None:
    """Execute one source-bound stage, preserving incomplete outputs for diagnosis."""
    q = qualify()
    receipt = RUNS / "qualification.json"
    if stage == "check":
        publish(receipt, q)
        return
    if read(receipt) != q:
        raise ValueError("receiver qualification missing or changed")
    from geometry_probe import run_probes, validate_probe

    probe_path = RUNS / "probe.json"
    if stage == "probe":
        publish(probe_path, run_probes(PACKET / "configs", q, OWNER))
        return
    if cell not in CELLS:
        raise ValueError(f"unknown cell: {cell}")
    validate_probe(read(probe_path), q)
    run = RUNS / cell
    run.mkdir(exist_ok=True)
    data = ["--data-root", str(DATA), "--split-manifest", str(SPLIT)]
    train = ["scripts/train_sequence.py", "--config", str(PACKET / "configs" / f"{cell}.json")]
    launch = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]
    if stage == "profile":
        args = launch + train + data + ["--output", str(run / "profile"), "--profile-updates", "50"]
    else:
        profile = qualified_profile(cell, q)
        if stage == "approve":
            approve_profile(cell, profile)
            return
        approved = approval(cell, profile)
        if stage == "fit":
            args = (
                launch
                + train
                + data
                + ["--output", str(run / "fit"), "--profile-receipt", str(approved)]
            )
        else:
            status = read(run / "fit/terminal_status.json")
            checkpoint = run / "fit/model.pt"
            if (
                status["status"] != "completed"
                or status["completed_epochs"] != 4
                or status["global_updates"] != 3_280
                or status["identity"] != profile["identity"]
                or status["model_sha256"] != file_sha256(checkpoint)
            ):
                raise ValueError("requires this cell's completed and hashed fixed E4 endpoint")
            if stage == "export":
                args = launch + [
                    "scripts/predict_sequence.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--membership",
                    "evaluation",
                    "--output",
                    str(run / "predictions"),
                ]
            elif stage == "score":
                manifest = read(run / "predictions/manifest.json")
                if manifest["producer"]["checkpoint_sha256"] != status["model_sha256"]:
                    raise ValueError("prediction producer differs from the admitted endpoint")
                args = [
                    sys.executable,
                    "scripts/score_sequence.py",
                    "--predictions",
                    str(run / "predictions"),
                    "--output",
                    str(run / "score.json"),
                ]
            else:
                raise ValueError(f"unknown receiver stage: {stage}")
            args += data
    os.execv(sys.executable, args)


if __name__ == "__main__":
    main(*sys.argv[1:])
