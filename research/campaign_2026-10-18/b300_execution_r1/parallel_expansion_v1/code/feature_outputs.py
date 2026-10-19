"""Verify exported native-feature banks and build unchanged official paired grades."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path

from feature_common import GRADER_SHA256, read, sha, write


def validate_export(root: Path, work: Path, job: dict, checkpoint_sha256: str) -> dict:
    """Require source-bound FP32 predictions over the exact held-out row-group order."""
    import numpy as np

    split_path = root / "splits" / f"fold{job['fold']}.json"
    split = read(split_path)
    row_groups = split["evaluation"]["valid"]
    if not isinstance(row_groups, list) or any(type(group) is not int for group in row_groups):
        raise ValueError("Fold evaluation row groups must be integer sequence IDs")
    membership_sha256 = hashlib.sha256(np.asarray(row_groups, dtype="<i8").tobytes()).hexdigest()
    prediction = work / "cache" / f"pred_{job['id']}_valid.npy"
    sidecar = prediction.with_suffix(".json")
    metadata = read(sidecar)
    actual_groups = metadata.get("row_groups")
    if (
        metadata.get("run") != job["id"]
        or type(metadata.get("epoch")) is not int
        or metadata.get("epoch") != 5
        or metadata.get("split") != "valid"
        or metadata.get("split_manifest_sha256") != sha(split_path)
        or metadata.get("campaign_export_scope") != "evaluation"
        or metadata.get("feature_only") is not True
        or metadata.get("inference_precision") != "fp32"
        or metadata.get("campaign_split_manifest_sha256") != sha(split_path)
        or not isinstance(actual_groups, list)
        or any(type(group) is not int for group in actual_groups)
        or actual_groups != row_groups
        or metadata.get("evaluation_membership_sha256") != membership_sha256
        or metadata.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("Prediction metadata differs from the frozen fit/fold endpoint")
    data = np.load(prediction, mmap_mode="r", allow_pickle=False)
    if data.shape != (len(row_groups), 20000, 2) or data.dtype != np.float32:
        raise ValueError("Prediction shape or dtype differs")
    if metadata.get("prediction_sha256") != sha(prediction):
        raise ValueError("Prediction bank bytes differ from sidecar")
    for i in range(len(data)):
        if not np.isfinite(data[i]).all():
            raise ValueError("Prediction bank contains nonfinite values")
    return dict(
        path=str(prediction),
        sha256=sha(prediction),
        metadata_sha256=sha(sidecar),
        checkpoint_sha256=checkpoint_sha256,
        shape=list(data.shape),
        dtype=str(data.dtype),
        split_sha256=sha(split_path),
    )


def grade_pair(root: Path, state: Path, candidate: dict, control: dict) -> dict:
    """Run the original official grader on two verified same-cell fixed endpoints."""
    c, r = candidate["job"], control["job"]
    if (c["fold"], c["seed"], c["family"]) != (r["fold"], r["seed"], r["family"]):
        raise ValueError("Grade endpoints do not share one scientific cell")
    for endpoint in (candidate, control):
        job = endpoint["job"]
        work = state / "work" / job["id"]
        if sha(work / "FIT.json") != endpoint["fit_receipt_sha256"]:
            raise ValueError("Fit receipt changed before grading")
        validate_export(root, work, job, endpoint["fit"]["checkpoint_sha256"])
    grade_root = state / "grades" / f"{c['family']}_f{c['fold']}_s{c['seed']}"
    grade_root.mkdir(parents=True, exist_ok=False)
    view = grade_root / "view"
    view.mkdir()
    (view / "datasets").symlink_to(root / "data/datasets", target_is_directory=True)
    cache = view / "cache"
    cache.mkdir()
    for name in ("chains.npz", "chain_folds.npz", "train_targets.npz", "valid_targets.npz"):
        (cache / name).symlink_to(root / "data/cache" / name)
    runs = view / "artifacts/runs"
    runs.mkdir(parents=True)
    for job in (c, r):
        work = state / "work" / job["id"]
        (runs / job["id"]).symlink_to(work / "artifacts/runs" / job["id"], target_is_directory=True)
        for suffix in ("npy", "json"):
            name = f"pred_{job['id']}_valid.{suffix}"
            (cache / name).symlink_to(work / "cache" / name)
    grader = root / "runtime/b300-r01-v1/grader/grade_endpoint.py"
    if sha(grader) != GRADER_SHA256:
        raise ValueError("Official scorer changed")
    scratch = grade_root / "scratch"
    scratch.mkdir()
    output = grade_root / "grade.json"
    argv = [
        sys.executable,
        str(grader),
        "--root",
        str(view),
        "--run",
        c["id"],
        "--reference",
        r["id"] + "@5",
        "--split",
        str(root / "splits" / f"fold{c['fold']}.json"),
        "--epoch",
        "5",
        "--output",
        str(output),
        "--bootstrap",
        "2000",
        "--scratch-dir",
        str(scratch),
    ]
    write(
        grade_root / "command.json",
        dict(
            argv=argv,
            grader_sha256=GRADER_SHA256,
            candidate_export_sha256=sha(state / "work" / c["id"] / "EXPORTED.json"),
            control_export_sha256=sha(state / "work" / r["id"] / "EXPORTED.json"),
        ),
        fresh=True,
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(scratch),
        PYTHONPATH=str(root / "runtime/source-a266-lf-subset-v1/src"),
    )
    with (grade_root / "stdout.txt").open("x") as out, (grade_root / "stderr.txt").open("x") as err:
        process = subprocess.run(argv, env=env, stdout=out, stderr=err, timeout=3600, check=False)
    if process.returncode != 0:
        raise RuntimeError("Official grader failed; original logs are retained")
    grade = read(output)
    if (
        grade.get("metric_id") != "global_wp_clip2_v1"
        or grade.get("fold") != c["fold"]
        or grade.get("endpoint", {}).get("run") != c["id"]
        or grade.get("reference", {}).get("run") != r["id"]
        or grade.get("split_manifest_sha256") != candidate["prediction"]["split_sha256"]
    ):
        raise ValueError("Official grade identities differ from requested endpoints")
    for key in ("endpoint", "reference"):
        if not math.isfinite(grade[key]["score"]["wp"]):
            raise ValueError("Official grade WP is nonfinite")
    receipt = dict(
        schema="tabdata-native-feature-paired-grade.v1",
        status="graded",
        family=c["family"],
        fold=c["fold"],
        seed=c["seed"],
        candidate=c["id"],
        control=r["id"],
        grade_path=str(output),
        grade_sha256=sha(output),
        grader_sha256=GRADER_SHA256,
        command_sha256=sha(grade_root / "command.json"),
        stdout_sha256=sha(grade_root / "stdout.txt"),
        stderr_sha256=sha(grade_root / "stderr.txt"),
        candidate_wp=grade["endpoint"]["score"]["wp"],
        control_wp=grade["reference"]["score"]["wp"],
    )
    write(grade_root / "RECEIPT.json", receipt, fresh=True)
    write(state / "scored" / f"{c['family']}_f{c['fold']}_s{c['seed']}.json", receipt, fresh=True)
    return receipt
