"""Admit each B03 run and dispatch the pinned native fit/export/scoring programs."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from runtime_source import BASE_COMMIT, PACKET, git, read, sha256
from wnn.training.sequence_ensemble_config import requires_teacher_cache

from wnn.training.sequence_config import load_config
from wnn.training.sequence_data import build_data
from wnn.training.sequence_provenance import canonical_sha256, validate_profile_receipt

RUNS = Path(os.environ["WF_RUNS"]).resolve()
RUNTIME = RUNS / "runtime"
DATA = Path(os.environ["WF_DATA"]).resolve()
SPLIT = Path(os.environ["WF_SPLIT"]).resolve()
CACHE = Path(os.environ["WF_CACHE"]).resolve()
OWNER = os.environ["WF_OWNER"].strip()
SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
CACHE_SHA = "138ef43c6ef03755a3b6195ca83f2c346f7dca4e3fe365f9e8e151ad2c71792d"
TEACHER_SHA = "759165147d99b33034b74f4af63a30e8a202dafc78fd5e5699d05b6cfe9bef90"
ARMS = {"labels": (0.0, 0.0), "since32": (32.0, 0.0), "kd": (0.0, 0.5), "both": (32.0, 0.5)}
CELLS = {
    f"{arm}_s{seed}": (seed, *values) for seed in (203301, 203302) for arm, values in ARMS.items()
}


def publish(path: Path, value: dict[str, Any], *, approval: bool = False) -> None:
    """Write one immutable receipt, validating raw native admission before publication."""
    with NamedTemporaryFile("w", dir=path.parent, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        if approval:
            validate_profile_receipt(Path(stream.name), value["identity"])
        os.link(stream.name, path)


def qualification() -> dict[str, Any]:
    """Bind packet, native postimage, eight unchanged-factor recipes, split and teacher."""
    manifest = read(PACKET / "runtime_manifest.json")
    if sha256(SPLIT) != SPLIT_SHA or sha256(CACHE / "manifest.json") != CACHE_SHA:
        raise ValueError("Use the exact official split and fitting-only PR49 teacher cache")
    # Bind the entire common recipe to the PR96 production template, allowing
    # exactly the named mechanism and the two registered seeds to move.
    template_path = PACKET / "configs/labels_s203301.json"
    original = dict(read(template_path), objective_mode="standalone_labels_kd_wmse_v1", seed=203501)
    if (
        canonical_sha256(original)
        != "dffcc35ea409222b9857c10d15c2b3d134940a215ebb53bb61bdcfeaab02dce2"
    ):
        raise ValueError("Non-factor settings differ from the pinned PR96 zero-KD reference")
    template = load_config(template_path)
    hashes = {}
    for cell, (seed, tau, alpha) in CELLS.items():
        cfg = load_config(PACKET / "configs" / f"{cell}.json")
        expected = dict(template, seed=seed, since_weight=tau, kd_weight=alpha)
        if cfg != expected or not requires_teacher_cache(cfg, False):
            raise ValueError(f"{cell}: inconsistent factorial configuration or cache semantics")
        hashes[cell] = canonical_sha256(cfg)
    return {
        "schema": "wnn.b03.qualification.v1",
        "packet_commit": git(PACKET, "rev-parse", "HEAD"),
        "base_commit": BASE_COMMIT,
        "source_sha256": manifest["source_sha256"],
        "patch_sha256": manifest["patch_sha256"],
        "data_root": str(DATA),
        "split_sha256": SPLIT_SHA,
        "cache_root": str(CACHE),
        "cache_manifest_sha256": CACHE_SHA,
        "teacher_sha256": TEACHER_SHA,
        "configs": hashes,
        "fit_train": 10_325,
        "fit_valid": 0,
        "evaluation_valid": 1_873,
    }


def check_data() -> None:
    """Use the actual native loader to verify teacher ancestry, every shard hash and the split."""
    cfg = load_config(PACKET / "configs/both_s203301.json")
    data = build_data(cfg, DATA, SPLIT, CACHE)
    fitting, evaluation = data.fit.members, data.evaluation.members
    teacher = data.provenance["teacher"]
    if (
        len(fitting) != len(set(fitting))
        or len(fitting) != 10_325
        or any(s != "train" for s, _ in fitting)
        or len(evaluation) != len(set(evaluation))
        or len(evaluation) != 1_873
        or any(s != "valid" for s, _ in evaluation)
        or data.split.sha256 != SPLIT_SHA
        or data.split.valid_groups
        or data.split.fold != 0
        or teacher["checkpoint_sha256"] != TEACHER_SHA
        or teacher["manifest_sha256"] != CACHE_SHA
    ):
        raise ValueError("Native fit-only teacher lineage or source qualification failed")


def qualified_profile(cell: str, q: dict[str, Any]) -> dict[str, Any]:
    """Match the original 30-update profile to this source, cache, cell and single L4."""
    profile = read(RUNS / cell / "profile/profile_receipt.json")
    identity = profile["identity"]
    d = identity["data"]
    if (
        profile.get("approved") is not False
        or profile["updates"] != 30
        or profile["status"] != "profile_complete"
        or profile["errors"] != []
        or identity["world_size"] != 1
        or identity["profile_reload_required"] is not True
        or identity["source_sha256"] != q["source_sha256"]
        or identity["config_sha256"] != q["configs"][cell]
        or d["split_sha256"] != SPLIT_SHA
        or d["teacher"]["checkpoint_sha256"] != TEACHER_SHA
        or d["teacher"]["manifest_sha256"] != CACHE_SHA
        or len(d["fit_members"]) != 10_325
        or any(m[0] != "train" for m in d["fit_members"])
        or len(d["evaluation_members"]) != 1_873
        or any(m[0] != "valid" for m in d["evaluation_members"])
        or "L4" not in identity["runtime"]["devices"][0]["name"]
    ):
        raise ValueError(f"{cell}: engineering profile is incomplete or belongs to another arm")
    return profile


def limit(name: str) -> float:
    """Require a populated finite positive allocation in GiB."""
    value = float(os.environ[name])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid memory budget {name}")
    return value


def review(cell: str, profile: dict[str, Any]) -> None:
    """Bind measured memory and reload results to the receiver's explicit decision."""
    note = os.environ["WF_REVIEW_NOTE"].strip()
    host, vram = limit("WF_MAX_HOST_GIB"), limit("WF_MAX_VRAM_GIB")
    host_peak = sum(rank["host_peak_rss_bytes"] for rank in profile["memory_by_rank"])
    if not note or host_peak > host * 2**30 or profile["peak_reserved_bytes"] > vram * 2**30:
        raise ValueError("Profile exceeds reserved memory or review note missing")
    approved = dict(
        profile,
        approved=True,
        reviewer=OWNER,
        approval_note=note,
        max_host_gib=host,
        max_vram_gib=vram,
        profile_sha256=sha256(RUNS / cell / "profile/profile_receipt.json"),
        objective_probe_sha256=sha256(RUNS / "probe.json"),
    )
    publish(RUNS / cell / "approved_profile.json", approved, approval=True)


def admitted(cell: str, profile: dict[str, Any]) -> Path:
    """Reconcile the unchanged production profile with its reviewed copy."""
    path = RUNS / cell / "approved_profile.json"
    approval = read(path)
    for key, value in profile.items():
        if key != "approved" and approval.get(key) != value:
            raise ValueError(f"Frozen profile field differs: {key}")
    if (
        approval["profile_sha256"] != sha256(RUNS / cell / "profile/profile_receipt.json")
        or approval["objective_probe_sha256"] != sha256(RUNS / "probe.json")
        or not approval["reviewer"].strip()
        or not approval["approval_note"].strip()
    ):
        raise ValueError("Profile does not match the recorded review")
    validate_profile_receipt(path, profile["identity"])
    return path


def execute(script: Path, arguments: list[str]) -> None:
    """Replace this one owned worker with the native stage under the launcher's deadline."""
    os.execv(sys.executable, [sys.executable, str(script), *arguments])


def main(stage: str, cell: str) -> None:
    """Run a bounded receiver stage with immutable evidence."""
    q = qualification()
    receipt = RUNS / "qualification.json"
    if stage == "check":
        check_data()
        publish(receipt, q)
        return
    if read(receipt) != q:
        raise ValueError("Source, split, teacher or frozen configuration has changed")
    probe_path = RUNS / "probe.json"
    if stage == "probe":
        execute(PACKET / "objective_probe.py", ["--output", str(probe_path)])
    probe = read(probe_path)
    if (
        probe.get("schema"),
        probe.get("status"),
        probe.get("source_sha256"),
        probe.get("qualification_sha256"),
    ) != ("wnn.b03.objective-probe.v1", "passed", q["source_sha256"], sha256(receipt)):
        raise ValueError("Missing source-qualified passed objective probe")
    if cell not in CELLS:
        raise ValueError("Unknown B03 cell")
    if stage in ("profile", "fit", "predict"):
        import torch

        if torch.cuda.device_count() != 1 or "L4" not in torch.cuda.get_device_name(0):
            raise ValueError("This cell requires one allocated L4 GPU")
    run = RUNS / cell
    run.mkdir(exist_ok=True)
    data = ["--data-root", str(DATA), "--split-manifest", str(SPLIT)]
    train = [
        "--config",
        str(PACKET / "configs" / f"{cell}.json"),
        *data,
        "--teacher-cache",
        str(CACHE),
    ]
    if stage == "profile":
        execute(
            RUNTIME / "scripts/train_sequence.py",
            [*train, "--output", str(run / "profile"), "--profile-updates", "30"],
        )
    profile = qualified_profile(cell, q)
    if stage == "approve":
        review(cell, profile)
        return
    approval = admitted(cell, profile)
    if stage == "fit":
        execute(
            RUNTIME / "scripts/train_sequence.py",
            [*train, "--output", str(run / "fit"), "--profile-receipt", str(approval)],
        )
    terminal = read(run / "fit/terminal_status.json")
    checkpoint = run / "fit/model.pt"
    if (
        terminal["status"],
        terminal["completed_epochs"],
        terminal["global_updates"],
        terminal["model_sha256"],
        terminal["identity"],
    ) != ("completed", 6, 38_760, sha256(checkpoint), profile["identity"]):
        raise ValueError("Expected this admitted cell's completed fixed E6 weights")
    if stage == "predict":
        execute(
            RUNTIME / "scripts/predict_sequence.py",
            [
                *data,
                "--checkpoint",
                str(checkpoint),
                "--membership",
                "evaluation",
                "--output",
                str(run / "predictions"),
            ],
        )
    if stage == "score":
        manifest = read(run / "predictions/manifest.json")
        if manifest["producer"]["checkpoint_sha256"] != terminal["model_sha256"]:
            raise ValueError("Export belongs to a different checkpoint")
        execute(
            RUNTIME / "scripts/score_sequence.py",
            [*data, "--predictions", str(run / "predictions"), "--output", str(run / "score.json")],
        )
    raise ValueError(f"Unknown stage {stage}")


if __name__ == "__main__":
    main(*sys.argv[1:])
