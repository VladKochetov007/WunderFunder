"""Scale-then-distill chain on one host: teacher predictions -> distillation cache -> matched student -> validation export.

Runs entirely from checkpoints already on the host.  For the teachers under test (one or two scratch runs) it exports the
train-split predictions (once, under a file lock), builds a four-family heterogeneous cache in which the three families the
test does not vary are filled with aliases of the same teachers (symlinked checkpoint and prediction files with a rewritten
sidecar run name, so each source run appears once), selects the ``homogeneous`` main target (strong-family mean scaled by
the total coefficient 0.8125) and fits an H230-recipe student warm-started from a parent checkpoint.  Students of different
arms differ only in the teachers.  The cache (13 GB) is deleted afterwards unless ``--keep-cache`` is given.

    distill_pipeline.py --arm t384 --fold 1 --teachers vt_f1_gru384_s4,vt_f1_gru384_s5 --parent c20261003_combo_s3_f1@ep3
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("WNN_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
RUNNER = ("import runpy,sys; from pathlib import Path; run_script=sys.argv[1]; trainer=sys.argv[2]; "
          "sys.path.insert(0,str(Path(trainer).resolve().parent)); sys.argv=[run_script,*sys.argv[3:]]; "
          "runpy.run_path(run_script,run_name='__main__')")


def run(argv: list[str], log: str) -> None:
    print("RUN", " ".join(argv)[:300], flush=True)
    with open(log, "ab") as handle:
        subprocess.run(argv, check=True, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)


def predict(run_name: str, split: str, epoch: int, output: str | None, log: str) -> None:
    argv = [sys.executable, "-c", RUNNER, str(ROOT / "scripts/predict_run.py"), str(ROOT / "scripts/train_rnn.py"),
            run_name, split, "--epoch", str(epoch)]
    if output:
        argv += ["--output-name", output]
    run(argv, log)


def ensure_train_predictions(name: str, epoch: int, log: str) -> None:
    lock_dir = Path("/root/run/locks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"export_{name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for split in ("valid", "train"):
            if not (ROOT / f"cache/pred_{name}_{split}.json").exists():
                predict(name, split, epoch, None, log)


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(source.resolve())


def build_alias_roots(arm: str, teachers: list[str], fold: int, epoch: int) -> tuple[Path, Path, list[str]]:
    base = Path(f"/root/run/alias_{arm}")
    shutil.rmtree(base, ignore_errors=True)
    ckpt, cache = base / "ckpt", base / "cache"
    ckpt.mkdir(parents=True)
    cache.mkdir(parents=True)
    link(ROOT / f"artifacts/runs/c20260929_prop_f{fold}", ckpt / f"c20260929_prop_f{fold}")
    for name in teachers:
        link(ROOT / f"artifacts/runs/{name}", ckpt / name)
        for split in ("train", "valid"):
            for suffix in ("npy", "json"):
                link(ROOT / f"cache/pred_{name}_{split}.{suffix}", cache / f"pred_{name}_{split}.{suffix}")

    def alias(source: str, tag: str) -> str:
        name = f"{source}_{tag}"
        directory = ckpt / name
        directory.mkdir()
        for filename in (f"model_ep{epoch}.pt", "provenance.json"):
            link(ROOT / f"artifacts/runs/{source}/{filename}", directory / filename)
        for split in ("train", "valid"):
            link(ROOT / f"cache/pred_{source}_{split}.npy", cache / f"pred_{name}_{split}.npy")
            sidecar = json.loads((ROOT / f"cache/pred_{source}_{split}.json").read_text())
            sidecar["run"] = name
            (cache / f"pred_{name}_{split}.json").write_text(json.dumps(sidecar))
        return name

    first, second = teachers[0], teachers[-1]
    spec = [f"strong={name}@ep{epoch}*{0.3 / len(teachers):.6f}" for name in teachers]
    spec.append(f"full_finetuned={alias(first, 'a1')}@ep{epoch}*0.2")
    spec.append(f"raw_wmse={alias(second, 'a2')}@ep{epoch}*0.25")
    spec.append(f"samefold_baseline={alias(first, 'a3')}@ep{epoch}*0.0625")
    return ckpt, cache, spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--teachers", required=True, help="one or two scratch run names")
    parser.add_argument("--teacher-epoch", type=int, default=3)
    parser.add_argument("--parent", required=True, help="warm-start checkpoint RUN@epN")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--variant", default="homogeneous", choices=["source", "normalized", "homogeneous"])
    parser.add_argument("--epochs", type=int, default=2, help="student epochs (final checkpoint = epochs-1)")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=256, help="student width (parent must match)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sched", default="onecycle")
    parser.add_argument("--no-teacher", action="store_true", help="control: same continuation without any distillation")
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    log = f"/root/run/logs/pipeline_{args.arm}.log"
    teachers = [t for t in args.teachers.split(",") if t]
    split_path = ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{args.fold}.json"
    student = f"vt_f{args.fold}_dsq_{args.arm}"
    import hashlib
    cache_key = hashlib.sha1((",".join(teachers) + f"@{args.teacher_epoch}").encode()).hexdigest()[:8]
    cache_dir = ROOT / f"cache/dc_{cache_key}"  # shared by every arm with the same teacher set/epoch (13 GB each otherwise)
    distill_flags: list[str] = []
    if not args.no_teacher:
        for name in teachers:
            ensure_train_predictions(name, args.teacher_epoch, log)
        ckpt_root, cache_root, spec = build_alias_roots(args.arm, teachers, args.fold, args.teacher_epoch)
        lock_dir = Path("/root/run/locks"); lock_dir.mkdir(parents=True, exist_ok=True)
        lock_handle = (lock_dir / f"dc_{cache_key}.lock").open("w")
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if (cache_dir / "metadata.json").exists():
            print("REUSE shared distillation cache", cache_dir, flush=True)
            reuse = True
        else:
            reuse = False
            shutil.rmtree(cache_dir, ignore_errors=True)
        build = [sys.executable, str(ROOT / "scripts/build_distill_cache.py"), "--checkpoint-root", str(ckpt_root),
                 "--teacher-cache-root", str(cache_root), "--split-manifest", str(split_path), "--output", str(cache_dir),
                 "--homogeneous-family", "strong", "--chunk-sequences", "4", "--output-budget-gb", "16",
                 "--min-free-gb", "20", "--expected-coefficient-sum", "0.8125"]
        if not reuse:
            for entry in spec:
                build += ["--teacher", entry]
            run(build, log)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        distill_flags = ["--distill-cache-root", str(cache_dir), "--distill-cache-variant", args.variant,
                         "--distill-alpha", str(args.alpha), "--distill-boot-h", "0"]
    fit = [sys.executable, str(ROOT / "scripts/train_rnn.py"), "--name", student, "--device", "cuda", "--arch", "gru",
           "--cell", "gru", "--hidden", str(args.hidden), "--layers", "2", "--dropout", "0", "--seed", str(args.seed),
           "--split-manifest", str(split_path), "--valid-fold", str(args.fold), "--eval-source", "vfold",
           "--train-source", "train+valid", "--train-seqs", "10607", "--init", args.parent, "--epochs", str(args.epochs),
           "--checkpoint-policy", "fixed", "--checkpoint-epoch", str(args.epochs - 1), "--no-eval-during-fit", "--batch", str(args.batch),
           "--chunk", "2000", "--loss", "wmse", "--loss-norm", "chunk", "--lr", str(args.lr), "--wd", str(args.wd),
           "--optim", "adamw", "--sched", args.sched, "--precision", "fp16", "--propensity", f"c20260929_prop_f{args.fold}",
           "--prop-power", "2", "--prop-eps", "0.03", "--valid-true-mask", "0.03", "--aux-struct", "0.3",
           "--aux-next", "0.3", *distill_flags, "--fast", "--save-training-state"]
    run(fit, log)
    predict(student, "valid", args.epochs - 1, None, log)
    if False:  # shared cache is kept (reused by sibling arms); delete manually when the teacher set is retired
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree(f"/root/run/alias_{args.arm}", ignore_errors=True)
    print("PIPELINE_DONE", student, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
