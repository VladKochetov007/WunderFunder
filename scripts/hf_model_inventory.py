from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO_ID = "hft-kagglers/wnn-challenge2-model-archive"
NOTE_ROOT = Path("research/unified/models/huggingface")
PAYLOAD_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".joblib", ".cbm", ".ubj"}
MODEL_NAME = re.compile(
    r"model|checkpoint|weights|readout|ridge|classifier|regressor|coef|unary|bilinear|output_only|binary",
    re.I,
)
PRIORITY_RUNS = {
    "vt_all_h390_ds_homogeneous_s3_r3": "All-data H390 student, seed 3; quarter of public-best C (user-reported 0.6671).",
    "vt_all_h390_ds_homogeneous_s4_r3": "All-data H390 student, seed 4; quarter of public-best C (user-reported 0.6671).",
    "bd26_h230_faithful_student_seed99_muonparent_v1_core_identity_v2": "Own Z4 student trunk, seed 99; merged with its HFT head for half of public-best C.",
    "z4_seed99_student_hft_alldata_v2": "Own all-data Z4 fine-tuned head; paired with the Z4 trunk in public-best C.",
    "clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1": "Clean-T3 F0 student CP5, seed 3; exposed local WP 0.6811553603.",
    "clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1": "Clean-T3 F0 student CP5, seed 4; exposed local WP 0.6812584217; shared teacher cohort.",
    "clean_t3_f0_muonall_h384_e3_s3_r5": "Own clean-F0 H384 Muon teacher CP2, seed 3; exposed WP 0.67898918.",
    "clean_t3_f0_muonall_h384_e3_s4_r5": "Own clean-F0 H384 Muon teacher CP2, seed 4; exposed WP 0.67836691.",
    "clean_t3_f0_muonall_h384_e3_s5_r5": "Own clean-F0 H384 Muon teacher CP2, seed 5; exposed WP 0.67969849.",
    "vt_f1_h391_dsq_h384_t3_a9_e6_s3_r2": "F1 H384-teacher distilled student; exposed WP 0.67103 in the archived comparison.",
    "h382_muonall256prop0_f3_s3_e3": "H382 F3 epoch-3 treatment; exposed WP about 0.70865, a local fold result.",
}


def read_json(path: Path) -> object:
    """Read small metadata without loading checkpoint tensors."""
    if path.stat().st_size > 8_000_000:
        raise ValueError(f"Metadata exceeds the admitted bound: {path}")
    return json.loads(path.read_text())


def digest(path: Path) -> tuple[str, str]:
    """Hash a payload in fixed-size chunks."""
    sha = hashlib.sha256()
    git_sha = hashlib.sha1(f"blob {path.stat().st_size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            sha.update(chunk)
            git_sha.update(chunk)
    return sha.hexdigest(), git_sha.hexdigest()


def discover(root: Path) -> list[Path]:
    """Find model candidates while excluding unrelated packages and runtime trees."""
    command = ["rg", "--files", "--hidden", "--no-ignore"]
    for pattern in [
        "!**/.git/**",
        "!**/.venv*/**",
        "!**/node_modules/**",
        "!**/site-packages/**",
        "!wunder/**",
        "!.claude/**",
        "!.runtime/**",
        "!native/target/**",
        "!cache/**",
        "!datasets/**",
        "!baseline/**",
        "!research/merge_2026-09-29/**",
        "!artifacts/hf_model_upload_20261012/**",
    ]:
        command.extend(["-g", pattern])
    for extension in sorted(PAYLOAD_SUFFIXES | {".npz", ".zip", ".pkl", ".npy"}):
        command.extend(["-g", f"*{extension}"])
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    return [root / line for line in result.stdout.splitlines()]


def classify(path: Path, relative: str) -> str:
    """Select checkpoints and served packages; keep data and diagnostic captures out."""
    if path.is_symlink() or not path.is_file():
        return "symlink_or_missing"
    if any(
        part
        in {
            "vendor",
            "baseline_snapshot",
            "default_baselines",
            "skip_baselines",
            "python311-site",
            "uv-cache",
            "target",
            "wheels",
        }
        for part in path.parts
    ):
        return "runtime_or_diagnostic_copy"
    if path.suffix in PAYLOAD_SUFFIXES:
        if (
            relative.startswith("artifacts/runs/")
            or path.suffix == ".safetensors"
            or MODEL_NAME.search(path.name)
            or (
                path.name == "training_state.pt"
                and (
                    (path.parent / "provenance.json").is_file()
                    or (path.parent / "model.pt").is_file()
                )
            )
        ):
            return "checkpoint"
        return "noncheckpoint_tensor_or_unqualified_model"
    if path.suffix == ".pkl":
        return "model_parameters" if MODEL_NAME.search(path.stem) else "data_or_unclassified_array"
    if path.suffix == ".npy":
        return (
            "model_parameters"
            if re.search(
                r"(^|_)(weights?|coefficients?|beta|intercept|readout)(_|$)", path.stem, re.I
            )
            else "data_or_unclassified_array"
        )
    if path.suffix == ".npz" and MODEL_NAME.search(path.stem):
        if path.stat().st_size > 128_000_000:
            return "oversized_unclassified_array"
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        if any(
            re.search(
                r"(^|/)(y|targets?|labels?|predictions?|preds?|features?|x|input_summary|shuffled_gain)\.npy$",
                name,
                re.I,
            )
            for name in names
        ):
            return "data_or_prediction_array"
        return "model_parameters"
    if path.suffix == ".zip" and relative.startswith(("submissions/", "submission/", "delivery/")):
        if path.stat().st_size > 128_000_000:
            return "oversized_nonserving_archive"
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        if "solution.py" in names and any(
            MODEL_NAME.search(name) and name.endswith((".npz", ".pt", ".onnx")) for name in names
        ):
            return "submission_package"
    return "nonmodel_artifact"


def run_metadata(path: Path, root: Path) -> tuple[str, dict[str, object], list[Path]]:
    """Identify each run from its local provenance, without unpickling weights."""
    parent = path.parent
    if path.relative_to(root).parts[:2] == ("artifacts", "runs"):
        parent = root / "artifacts/runs" / path.relative_to(root).parts[2]
    metadata: dict[str, object] = {}
    companions = []
    for name in [
        "provenance.json",
        "history.json",
        "config.json",
        "metrics.json",
        "args.json",
        "members.json",
        "precision.txt",
    ]:
        candidate = parent / name
        if candidate.is_file() and candidate.stat().st_size <= 8_000_000:
            companions.append(candidate)
    provenance = parent / "provenance.json"
    if provenance in companions:
        value = read_json(provenance)
        if isinstance(value, dict):
            metadata = value
    args = metadata.get("args", {})
    name = args.get("name") if isinstance(args, dict) else None
    return str(name or parent.relative_to(root)), metadata, companions


def describe(run_id: str, metadata: dict[str, object]) -> str:
    """Describe recorded architecture and training treatment without invented scores."""
    if run_id in PRIORITY_RUNS:
        return PRIORITY_RUNS[run_id]
    args = metadata.get("args", {})
    if not isinstance(args, dict):
        args = {}
    fields = [
        f"{key}={args[key]}"
        for key in [
            "arch",
            "hidden",
            "layers",
            "optim",
            "seed",
            "distill_alpha",
            "checkpoint_epoch",
            "train_source",
        ]
        if key in args
    ]
    return (
        "Archived own run: "
        + (", ".join(fields) or "see source path and companion metadata")
        + ". Scores and ancestry require the linked original evidence."
    )


def main() -> None:
    """Inventory, hash and hardlink explicitly selected models for private upload."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    root, stage = args.root.resolve(), args.stage.resolve()
    if stage == root or not stage.is_relative_to(root / "artifacts"):
        raise ValueError("Use a dedicated physical staging folder under artifacts")
    note = root / NOTE_ROOT
    old_inventory = note / "MODEL_INVENTORY_20261012.json"
    cached_entries = (
        json.loads(old_inventory.read_text())["files"] if old_inventory.is_file() else []
    )
    if stage.exists():
        if stage != root / "artifacts/hf_model_upload_20261012/stage":
            raise ValueError("Only the task's exact regenerable staging directory can be rebuilt")
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "--target", str(stage)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if filesystem in {"tmpfs", "ramfs"}:
        raise ValueError("RAM-backed staging is forbidden")
    entries: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    runs: dict[str, dict[str, object]] = {}
    inode_hashes: dict[tuple[int, int, int, int], tuple[str, str]] = {}
    for entry in cached_entries:
        path = root / entry["local_path"]
        if path.is_file():
            stat = path.stat()
            if stat.st_size == entry["bytes"] and stat.st_mtime_ns == entry["mtime_ns"]:
                identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                inode_hashes[identity] = (entry["sha256"], entry["git_blob_sha1"])
    canonical: dict[str, dict[str, object]] = {}
    selected: dict[Path, tuple[str, str]] = {}
    for path in discover(root):
        relative = path.relative_to(root).as_posix()
        kind = classify(path, relative)
        if kind not in {"checkpoint", "model_parameters", "submission_package"}:
            excluded.append({"local_path": relative, "reason": kind})
            continue
        run_id, metadata, companions = run_metadata(path, root)
        priority = (
            run_id in PRIORITY_RUNS
            or relative.startswith("submissions/10-12-best-r1/")
            or any(x in run_id for x in ["h384", "h512", "h382", "oldz4", "z4", "z5", "gru768"])
        )
        public_best = run_id in list(PRIORITY_RUNS)[:4] or relative.startswith(
            "submissions/10-12-best-r1/"
        )
        if run_id not in runs:
            runs[run_id] = {
                "run_id": run_id,
                "description": describe(run_id, metadata),
                "priority": 0 if public_best else (1 if priority else 2),
                "source_directory": path.parent.relative_to(root).as_posix(),
                "metadata_paths": [p.relative_to(root).as_posix() for p in companions],
                "files": [],
            }
        selected[path] = (run_id, kind)
        for companion in companions:
            selected.setdefault(companion, (run_id, "run_metadata"))
    for index, (path, (run_id, kind)) in enumerate(
        sorted(selected.items(), key=lambda item: (runs[item[1][0]]["priority"], str(item[0])))
    ):
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        hashes = inode_hashes.get(identity)
        if hashes is None:
            hashes = digest(path)
            inode_hashes[identity] = hashes
        sha, git_sha = hashes
        if kind == "run_metadata" and re.search(
            r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}", path.read_text()
        ):
            raise ValueError(f"Credential-like content in upload metadata: {path}")
        relative = path.relative_to(root).as_posix()
        hub_path = f"files/{relative}"
        entry = {
            "local_path": relative,
            "run_id": run_id,
            "kind": kind,
            "bytes": stat.st_size,
            "sha256": sha,
            "git_blob_sha1": git_sha,
            "mtime_ns": stat.st_mtime_ns,
            "priority": runs[run_id]["priority"],
            "hub_path": hub_path,
        }
        previous = canonical.get(sha)
        if previous is not None:
            entry["hub_path"] = previous["hub_path"]
            entry["duplicate_of"] = previous["local_path"]
        else:
            canonical[sha] = entry
            lane = ["best", "priority", "archive"][int(entry["priority"])]
            target = stage / lane / hub_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(path, target)
        entries.append(entry)
        runs[run_id]["files"].append(relative)
        if index % 500 == 0:
            logging.info("Inventoried %s/%s files", index, len(selected))
    names = sorted((name for name in runs if "/" not in name), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(name) for name in names))
    for ledger in ["research/hypotheses.jsonl", "research/experiments.jsonl"]:
        with (root / ledger).open() as handle:
            for number, line in enumerate(handle, 1):
                hits = set(pattern.findall(line))
                if not hits:
                    continue
                record = json.loads(line)
                for name in hits:
                    refs = runs[name].setdefault("research_records", [])
                    refs.append(
                        {
                            "path": ledger,
                            "line": number,
                            "id": record.get(
                                "id", record.get("hypothesis_id", record.get("experiment_id"))
                            ),
                        }
                    )
    inventory = {
        "schema": "wnn.hf-private-model-inventory.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "repo_id": REPO_ID,
        "repo_type": "model",
        "required_private": True,
        "status": "inventoried_not_uploaded",
        "stage": str(stage),
        "root": str(root),
        "files": entries,
        "runs": list(runs.values()),
        "excluded": excluded,
        "summary": {
            "files": len(entries),
            "unique_files": len(canonical),
            "runs": len(runs),
            "unique_bytes": sum(int(e["bytes"]) for e in canonical.values()),
            "kinds": dict(Counter(str(e["kind"]) for e in entries)),
        },
    }
    note.mkdir(parents=True, exist_ok=True)
    (note / "MODEL_INVENTORY_20261012.json").write_text(json.dumps(inventory, indent=2) + "\n")
    for run_id, run in runs.items():
        key = hashlib.sha256(run_id.encode()).hexdigest()[:12]
        card = stage / "best" / "run-cards" / f"{key}.md"
        files = [e for e in entries if e["run_id"] == run_id]
        lines = [
            f"# {run_id}",
            "",
            str(run["description"]),
            "",
            f"Local run: `{run['source_directory']}`.",
            "",
            "## Checkpoints and metadata",
            "",
        ]
        lines += [
            f"- `{e['local_path']}` → [{e['hub_path']}](/"
            + REPO_ID
            + f"/blob/main/{e['hub_path']})"
            for e in files
        ]
        lines += ["", "## Research records", ""]
        lines += [
            f"- `{ref['id']}`: `{ref['path']}:{ref['line']}`"
            for ref in run.get("research_records", [])
        ]
        if not run.get("research_records"):
            lines.append(
                "Search the local path/run ID in the repository; no direct ledger mention was found."
            )
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text("\n".join(lines) + "\n")
        run["hub_card_path"] = f"run-cards/{key}.md"
    (note / "MODEL_INVENTORY_20261012.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (stage / "best" / "MODEL_INVENTORY_20261012.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    logging.info("Inventory complete: %s", inventory["summary"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
