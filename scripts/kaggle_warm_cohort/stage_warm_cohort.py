"""Stage the warm-cohort source dataset, queue specs and a local fake Kaggle input tree.

Modes:
  --stage DIR        write the upload folder for the new private source dataset: wnn_src.tar.gz
                     (scripts/, src/, warm_specs/), assets-manifest.json and the small fold-0 parent
                     checkpoint files, plus specs/ for the receipt.
  --fake-input DIR   symlink tree shaped like /kaggle/input (raw data, fold-0 assets, source dataset)
                     so kernel_driver.py can be dry-run locally with WARM_DRY_RUN=1.

Code is read from the main checkout (scripts/, src/) and from this worktree (kaggle_warm_cohort/).
Nothing here touches Kaggle or Vast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
HERE = Path(__file__).resolve().parent
FOLD0_MANIFEST = Path("/tmp/kc2/manifest/assets-manifest.json")
PARENT_RUN = "c20261003_combo_s3_f0"
PARENT_FILES = ("model_ep3.pt", "provenance.json")
FOLD0_SPLIT = "research/campaign_2026-09-29/artifacts/splits/fold0.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def build_fit_spec(run_id: str, *, gpu: int, seed: int, max_batches: int, priority: int = 50) -> dict:
    """Fold-0 continuation of the C1 parent (E511 argv minus the teacher-prediction distill)."""
    repo = "{repo}"
    fit = [
        "{python}", f"{repo}/scripts/train_rnn.py",
        "--name", run_id, "--device", "cuda", "--arch", "gru", "--cell", "gru",
        "--hidden", "256", "--layers", "2", "--dropout", "0", "--seed", str(seed),
        "--split-manifest", f"{repo}/{FOLD0_SPLIT}", "--valid-fold", "0",
        "--train-source", "train+valid", "--train-seqs", "10607",
        "--init", f"{PARENT_RUN}@ep3", "--epochs", "2", "--checkpoint-policy", "fixed",
        "--checkpoint-epoch", "1", "--no-eval-during-fit", "--batch", "16", "--chunk", "2000",
        "--loss", "wmse", "--lr", "0.0001", "--wd", "0.01", "--optim", "adamw", "--sched", "onecycle",
        "--propensity", "c20260929_prop_f0", "--prop-power", "2", "--prop-eps", "0.03",
        "--valid-true-mask", "0.03", "--aux-struct", "0.3", "--aux-next", "0.3",
        "--distill-boot-h", "0", "--fast", "--save-training-state",
        "--max-batches", str(max_batches),
    ]
    env = {
        "WNN_THREADS": "2", "WNN_LOADER_WORKERS": "0", "ARROW_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
    }
    return {
        "run_id": run_id,
        "priority": priority,
        "gpu": gpu,
        "arm": "smoke-fold0-continuation",
        "hypothesis": "pipeline smoke: fold-0 parent continuation, capped batches, predict, official grade",
        "steps": [
            {
                "name": "fit",
                "argv": fit,
                "timeout_seconds": 3000,
                "env": env,
                "cwd": repo,
                "resume_state_path": f"{repo}/artifacts/runs/{run_id}/training_state.pt",
            },
            {"name": "predict", "argv": ["{python}", f"{repo}/scripts/predict_run.py", run_id, "valid", "--epoch", "1"],
             "timeout_seconds": 3600, "env": env, "cwd": repo},
            {"name": "grade", "argv": ["{python}", f"{repo}/scripts/kaggle_warm_cohort/grade_fold0_run.py",
                                        "--root", repo, "--run", run_id, "--epoch", "1", "--out-dir", "{out}/grades"],
             "timeout_seconds": 900, "env": {}, "cwd": repo},
        ],
        "expected_outputs": [
            f"{repo}/artifacts/runs/{run_id}/model_ep1.pt",
            f"{repo}/cache/pred_{run_id}_valid.npy",
            f"{{out}}/grades/{run_id}.grade.json",
        ],
    }


def smoke_specs(suffix: str = "v1") -> list[dict]:
    return [
        build_fit_spec(f"smoke-f0-s3-gpu0-{suffix}", gpu=0, seed=3, max_batches=30),
        build_fit_spec(f"smoke-f0-s13-gpu1-{suffix}", gpu=1, seed=13, max_batches=30),
    ]


def code_files() -> list[tuple[Path, str]]:
    files_by_name: dict[str, Path] = {}
    for top in ("scripts", "src"):
        for path in sorted((MAIN / top).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            files_by_name[path.relative_to(MAIN).as_posix()] = path
    for path in sorted(HERE.glob("*.py")):
        files_by_name[f"scripts/kaggle_warm_cohort/{path.name}"] = path
    worker_override = HERE.parent / "warm_queue_worker.py"
    if worker_override.is_file():
        files_by_name["scripts/warm_queue_worker.py"] = worker_override
    return [(files_by_name[name], name) for name in sorted(files_by_name)]


def write_archive(dest: Path, specs: list[dict]) -> None:
    staging = dest.parent / "_spec_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for spec in specs:
        (staging / f"{spec['run_id']}.json").write_text(json.dumps(spec, indent=1) + "\n")
    with tarfile.open(dest, "w:gz", compresslevel=6) as tar:
        for path, arcname in code_files():
            tar.add(path, arcname=arcname, recursive=False)
        for spec_path in sorted(staging.glob("*.json")):
            tar.add(spec_path, arcname=f"warm_specs/{spec_path.name}")
    shutil.rmtree(staging)


def stage(dest: Path, specs: list[dict]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    write_archive(dest / "wnn_src.tar.gz", specs)
    parent_dir = MAIN / "artifacts/runs" / PARENT_RUN
    entries = []
    for name in PARENT_FILES:
        source = parent_dir / name
        target = dest / name
        shutil.copy2(source, target)
        entries.append({
            "filename": name,
            "path": f"artifacts/runs/{PARENT_RUN}/{name}",
            "sha256": sha256(target),
            "size": target.stat().st_size,
        })
    manifest = {"dataset": "vladkochetov07/wnn-warm-cohort-src-20261008", "private": True,
                "version": 1, "files": entries}
    (dest / "assets-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    specs_dir = dest.parent / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        (specs_dir / f"{spec['run_id']}.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({"stage": str(dest), "files": sorted(os.listdir(dest))}, indent=1))


def stage_hash_bound_source(dest: Path, specs: list[dict]) -> None:
    """Build a small manifest-bound source dataset with opaque ``.wnn`` names."""
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty source upload: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []

    def add_asset(source: Path, logical_path: str) -> None:
        digest = sha256(source)
        filename = f"{digest}.wnn"
        target = dest / filename
        if target.exists():
            raise FileExistsError(f"duplicate content-addressed source asset: {filename}")
        shutil.copy2(source, target)
        entries.append(
            {
                "filename": filename,
                "path": logical_path,
                "sha256": digest,
                "size": target.stat().st_size,
            }
        )

    archive = dest / "_source_bundle.tmp"
    write_archive(archive, specs)
    add_asset(archive, "wnn_src.tar.gz")
    archive.unlink()

    parent_dir = MAIN / "artifacts/runs" / PARENT_RUN
    for name in PARENT_FILES:
        add_asset(parent_dir / name, f"artifacts/runs/{PARENT_RUN}/{name}")

    slug = "vladkochetov07/wnn-warm-cohort-src-20261008-v2"
    manifest = {"dataset": slug, "private": True, "version": 1, "files": entries}
    (dest / "assets-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metadata = {
        "id": slug,
        "title": "wnn warm cohort source 20261008 v2",
        "isPrivate": True,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (dest / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    specs_dir = dest.parent / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        (specs_dir / f"{spec['run_id']}.json").write_text(json.dumps(spec, indent=1) + "\n")
    print(
        json.dumps(
            {
                "stage": str(dest),
                "file_count": len(entries) + 2,
                "files": [
                    {
                        "filename": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for path in sorted(dest.iterdir())
                    if path.is_file()
                ],
            },
            indent=1,
        )
    )


def fake_input(root: Path) -> None:
    """Mirror the Kaggle mounts with symlinks into the main checkout (no data copies)."""
    raw = root / "wunder-funder"
    raw.mkdir(parents=True, exist_ok=True)
    for name in ("train.parquet", "valid.parquet", "valid_mask.parquet"):
        link = raw / name
        if not link.exists():
            os.symlink(MAIN / "datasets" / name, link)
    fold = root / "fold0-assets"
    fold.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(FOLD0_MANIFEST.read_text())
    for entry in manifest["files"]:
        link = fold / entry["filename"]
        if not link.exists():
            os.symlink(MAIN / entry["path"], link)
    shutil.copy2(FOLD0_MANIFEST, fold / "assets-manifest.json")
    src = root / "warm-cohort-src"
    src.mkdir(parents=True, exist_ok=True)
    stage_dir = Path("/tmp/kc2/stage/src")
    for name in ("wnn_src.tar.gz", "assets-manifest.json", *PARENT_FILES):
        link = src / name
        if not link.exists():
            os.symlink(stage_dir / name, link)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--stage-hash-bound-source", type=Path)
    parser.add_argument("--fake-input", type=Path)
    args = parser.parse_args()
    specs = smoke_specs()
    if args.stage:
        stage(args.stage, specs)
    if args.stage_hash_bound_source:
        stage_hash_bound_source(args.stage_hash_bound_source, smoke_specs("v3"))
    if args.fake_input:
        fake_input(args.fake_input)


if __name__ == "__main__":
    main()
