"""Scale-then-distill chain with out-of-sample path distillation (H-350), from checkpoints already on the host.

Same arguments and student recipe as ``scripts/distill_pipeline.py`` (warm start from --parent, AdamW lr 1e-4, wMSE with the
cached homogeneous teacher target on real chunks) plus ``--ood-*`` flags: a share of the batches is replaced by synthetic
fit-split paths whose target is the frozen teachers' mean output on that path (computed online).

    ood_student_pipeline.py --arm ood_t256 --fold 1 --teachers vt_f1_gru256_s4,vt_f1_gru256_s5 \
        --parent c20261003_combo_s3_f1@ep3 --ood-aug splice,warp,noise --ood-frac 0.5

Reuses ``distill_pipeline`` (same directory as the project's scripts) for teacher exports, the alias roots and the cache build.
``--real-distill none`` skips the 13 GB cache: real chunks then use plain labels (the OOD term is the only distillation).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ.get("WNN_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import distill_pipeline as dp  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--teachers", required=True, help="one or two scratch run names")
    parser.add_argument("--teacher-epoch", type=int, default=3)
    parser.add_argument("--parent", required=True, help="warm-start checkpoint RUN@epN")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.7, help="real-chunk distillation alpha (as distill_pipeline)")
    parser.add_argument("--variant", default="homogeneous", choices=["source", "normalized", "homogeneous"])
    parser.add_argument("--real-distill", choices=["cache", "none"], default="cache")
    parser.add_argument("--ood-aug", default="splice", help="splice,warp[:P/LO/HI],noise[:SIGMA],mixup_paths[:P]")
    parser.add_argument("--ood-frac", type=float, default=0.5)
    parser.add_argument("--ood-mode", choices=["mix", "only", "alternate"], default="mix")
    parser.add_argument("--ood-scale", type=float, default=0.8125, help="teacher-mean scale; 0.8125 = dsq cache coefficient sum")
    parser.add_argument("--ood-weight", choices=["teacher", "uniform"], default="teacher")
    parser.add_argument("--ood-sources", type=int, default=24)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    log = f"/root/run/logs/pipeline_{args.arm}.log"
    teachers = [t for t in args.teachers.split(",") if t]
    split_path = ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{args.fold}.json"
    student = f"vt_f{args.fold}_ood_{args.arm}"
    cache_dir = ROOT / f"cache/dc_{args.arm}"
    distill_flags: list[str] = []
    if args.real_distill == "cache":
        for name in teachers:
            dp.ensure_train_predictions(name, args.teacher_epoch, log)
        ckpt_root, cache_root, spec = dp.build_alias_roots(args.arm, teachers, args.fold, args.teacher_epoch)
        shutil.rmtree(cache_dir, ignore_errors=True)
        build = [sys.executable, str(ROOT / "scripts/build_distill_cache.py"), "--checkpoint-root", str(ckpt_root),
                 "--teacher-cache-root", str(cache_root), "--split-manifest", str(split_path), "--output", str(cache_dir),
                 "--homogeneous-family", "strong", "--chunk-sequences", "4", "--output-budget-gb", "16",
                 "--min-free-gb", "20", "--expected-coefficient-sum", "0.8125"]
        for entry in spec:
            build += ["--teacher", entry]
        dp.run(build, log)
        distill_flags = ["--distill-cache-root", str(cache_dir), "--distill-cache-variant", args.variant,
                         "--distill-alpha", str(args.alpha), "--distill-boot-h", "0"]
    ood_flags = ["--ood-teachers", ",".join(f"{name}@ep{args.teacher_epoch}" for name in teachers),
                 "--ood-aug", args.ood_aug, "--ood-frac", str(args.ood_frac), "--ood-mode", args.ood_mode,
                 "--ood-scale", str(args.ood_scale), "--ood-weight", args.ood_weight,
                 "--ood-sources", str(args.ood_sources)]
    trainer = Path(__file__).resolve().parent / "train_rnn.py"  # this snapshot's trainer (carries the --ood-* flags)
    fit = [sys.executable, str(trainer), "--name", student, "--device", "cuda", "--arch", "gru",
           "--cell", "gru", "--hidden", "256", "--layers", "2", "--dropout", "0", "--seed", str(args.seed),
           "--split-manifest", str(split_path), "--valid-fold", str(args.fold), "--eval-source", "vfold",
           "--train-source", "train+valid", "--train-seqs", "10607", "--init", args.parent, "--epochs", "2",
           "--checkpoint-policy", "fixed", "--checkpoint-epoch", "1", "--no-eval-during-fit", "--batch", "16",
           "--chunk", "2000", "--loss", "wmse", "--loss-norm", "chunk", "--lr", "0.0001", "--wd", "0.01",
           "--optim", "adamw", "--sched", "onecycle", "--precision", "fp16", "--propensity", f"c20260929_prop_f{args.fold}",
           "--prop-power", "2", "--prop-eps", "0.03", "--valid-true-mask", "0.03", "--aux-struct", "0.3",
           "--aux-next", "0.3", *distill_flags, *ood_flags, "--fast", "--save-training-state"]
    dp.run(fit, log)
    dp.predict(student, "valid", 1, None, log)
    if not args.keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree(f"/root/run/alias_{args.arm}", ignore_errors=True)
    print("PIPELINE_DONE", student, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
