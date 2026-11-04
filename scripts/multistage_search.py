from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.multistage_search_adopt import adopt_external_worker  # noqa: E402
from wnn.multistage_search_adopt_fit import adopt_completed_fit  # noqa: E402
from wnn.multistage_search_approval import approve_profile  # noqa: E402
from wnn.multistage_search_controller import (  # noqa: E402
    launch_worker,
    status_summary,
    stop_worker,
)
from wnn.multistage_search_ensemble import run_stage4_ensemble  # noqa: E402
from wnn.multistage_search_prepare import prepare_run  # noqa: E402
from wnn.multistage_search_successor_approval import (  # noqa: E402
    approve_successor_profile,
)
from wnn.multistage_search_successor_prepare import (  # noqa: E402
    prepare_stage2_trial,
    prepare_stage3_pair,
)
from wnn.multistage_search_successor_roll import (  # noqa: E402
    prepare_and_supervise_stage2,
    prepare_and_supervise_stage3,
)
from wnn.multistage_search_supervisor import supervise_stage1  # noqa: E402
from wnn.multistage_search_teachers import build_teacher_pool  # noqa: E402
from wnn.multistage_search_worker import run_worker  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the bounded four-stage Optuna controller and one-shot worker CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="freeze reviewed contract/admission and create the study"
    )
    prepare.add_argument("--contract", required=True, type=Path)
    prepare.add_argument("--admission", required=True, type=Path)
    prepare.add_argument("--run-dir", required=True, type=Path)
    prepare.add_argument("--training-source", type=Path)
    prepare.add_argument("--training-source-revision")
    prepare.add_argument("--evaluation-source", type=Path)
    prepare.add_argument("--evaluation-source-revision")

    launch = commands.add_parser("launch", help="detach one bounded profile or fit worker")
    launch.add_argument("--run-dir", required=True, type=Path)
    launch.add_argument("--mode", required=True, choices=("profile", "fit", "teacher-export"))
    launch.add_argument("--worker-id", required=True)
    launch.add_argument("--gpu-key", required=True)
    launch.add_argument("--trial-number", type=int)
    launch.add_argument("--result-path", type=Path)
    launch.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)

    worker = commands.add_parser("worker", help="run one profile or fit worker in the foreground")
    worker.add_argument("--run-dir", required=True, type=Path)
    worker.add_argument("--mode", required=True, choices=("profile", "fit", "teacher-export"))
    worker.add_argument("--worker-id", required=True)
    worker.add_argument("--gpu-key", required=True)
    worker.add_argument("--trial-number", type=int)
    worker.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    worker.add_argument("--result-path", type=Path)

    approve = commands.add_parser(
        "approve-profile", help="consume a separate deterministic reviewer receipt for one profile"
    )
    approve.add_argument("--run-dir", required=True, type=Path)
    approve.add_argument("--trial-number", required=True, type=int)
    approve.add_argument("--review-receipt", required=True, type=Path)

    successor_approve = commands.add_parser(
        "approve-successor-profile", help="consume a separate Stage2/3 reviewer receipt"
    )
    successor_approve.add_argument("--run-dir", required=True, type=Path)
    successor_approve.add_argument("--stage", required=True, type=int, choices=(2, 3))
    successor_approve.add_argument("--trial-number", required=True, type=int)
    successor_approve.add_argument("--review-receipt", required=True, type=Path)

    stage2 = commands.add_parser(
        "stage2-propose", help="propose one fresh-optimizer lower-LR continuation"
    )
    stage2.add_argument("--run-dir", required=True, type=Path)
    stage2.add_argument("--parent-result", required=True, type=Path)
    stage2.add_argument("--lr-multiplier", type=float, choices=(0.1, 0.25, 0.5))

    stage3 = commands.add_parser(
        "stage3-pair-propose", help="propose a matched label-only/KD student pair"
    )
    stage3.add_argument("--run-dir", required=True, type=Path)
    stage3.add_argument("--family", required=True)
    stage3.add_argument("--base-config", required=True, type=Path)
    stage3.add_argument("--teacher-cache", required=True, type=Path)

    pool = commands.add_parser(
        "stage3-teacher-pool", help="merge pinned fit-only teacher exports using frozen weights"
    )
    pool.add_argument("--run-dir", required=True, type=Path)
    pool.add_argument("--pool-id", required=True)
    pool.add_argument("--teacher-export", action="append", required=True, type=Path)
    pool.add_argument("--weight", action="append", required=True, type=float)

    status = commands.add_parser(
        "status", help="report durable budget, trial and owned-worker state"
    )
    status.add_argument("--run-dir", required=True, type=Path)

    stop = commands.add_parser(
        "stop-worker", help="stop only one PID/birth/PGID receipt owned by this run"
    )
    stop.add_argument("--run-dir", required=True, type=Path)
    stop.add_argument("--worker-id", required=True)

    adopt = commands.add_parser(
        "adopt-worker", help="register one already-running root-owned fit from its identity receipt"
    )
    adopt.add_argument("--run-dir", required=True, type=Path)
    adopt.add_argument("--receipt", required=True, type=Path)

    adopt_fit = commands.add_parser(
        "adopt-completed-fit",
        help="validate and score one completed external fit without rerunning training",
    )
    adopt_fit.add_argument("--run-dir", required=True, type=Path)
    adopt_fit.add_argument("--attempt-id", required=True)
    adopt_fit.add_argument("--config", required=True, type=Path)
    adopt_fit.add_argument("--approved-profile", required=True, type=Path)
    adopt_fit.add_argument("--owner-receipt", required=True, type=Path)
    adopt_fit.add_argument("--terminal-receipt", required=True, type=Path)
    adopt_fit.add_argument("--gpu-key", required=True)

    supervisor = commands.add_parser(
        "supervise", help="continuously fill free admitted stage-one worker slots"
    )
    supervisor.add_argument("--run-dir", required=True, type=Path)
    supervisor.add_argument("--worker-prefix", default="stage1")
    supervisor.add_argument("--max-workers", type=int, default=6)
    supervisor.add_argument("--poll-seconds", type=float, default=5.0)
    supervisor.add_argument("--max-trials", type=int)

    roll2 = commands.add_parser(
        "roll-stage2", help="prepare parent continuations and roll reviewed profiles into fits"
    )
    roll2.add_argument("--run-dir", required=True, type=Path)
    roll2.add_argument("--parent-result", action="append", required=True, type=Path)
    roll2.add_argument("--worker-prefix", default="stage2")
    roll2.add_argument("--max-workers", type=int, default=6)
    roll2.add_argument("--poll-seconds", type=float, default=5.0)
    roll2.add_argument("--max-runtime-seconds", type=int, default=21600)
    roll2.add_argument("--lr-multiplier", type=float, choices=(0.1, 0.25, 0.5))

    roll3 = commands.add_parser(
        "roll-stage3", help="prepare matched KD pairs and roll reviewed profiles into fits"
    )
    roll3.add_argument("--run-dir", required=True, type=Path)
    roll3.add_argument("--family", required=True)
    roll3.add_argument("--base-config", required=True, type=Path)
    roll3.add_argument("--teacher-cache", required=True, type=Path)
    roll3.add_argument("--pairs", required=True, type=int)
    roll3.add_argument("--worker-prefix", default="stage3")
    roll3.add_argument("--max-workers", type=int, default=6)
    roll3.add_argument("--poll-seconds", type=float, default=5.0)
    roll3.add_argument("--max-runtime-seconds", type=int, default=21600)

    ensemble = commands.add_parser(
        "stage4-ensemble", help="fit CPU simplex weights from qualified inner predictions"
    )
    ensemble.add_argument("--run-dir", required=True, type=Path)
    ensemble.add_argument("--data-root", required=True, type=Path)
    ensemble.add_argument("--auxiliary-split", required=True, type=Path)
    ensemble.add_argument(
        "--candidate", action="append", required=True, help="qualified NAME=trial_result.json"
    )

    return parser


def main() -> None:
    """Execute one controller operation and print its machine-readable receipt."""
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = prepare_run(
            args.contract,
            args.admission,
            args.run_dir,
            args.training_source,
            args.training_source_revision,
            args.evaluation_source,
            args.evaluation_source_revision,
        )
    elif args.command == "launch":
        result = launch_worker(
            args.run_dir,
            args.mode,
            args.worker_id,
            args.gpu_key,
            args.trial_number,
            stage=args.stage,
            result_path=args.result_path,
        )
    elif args.command == "worker":
        result = run_worker(
            args.run_dir,
            mode=args.mode,
            worker_id=args.worker_id,
            gpu_key=args.gpu_key,
            trial_number=args.trial_number,
            stage=args.stage,
            result_path=args.result_path,
        )
    elif args.command == "approve-profile":
        result = approve_profile(args.run_dir, args.trial_number, args.review_receipt)
    elif args.command == "approve-successor-profile":
        result = approve_successor_profile(
            args.run_dir, args.stage, args.trial_number, args.review_receipt
        )
    elif args.command == "stage2-propose":
        result = prepare_stage2_trial(
            args.run_dir, args.parent_result, lr_multiplier=args.lr_multiplier
        )
    elif args.command == "stage3-pair-propose":
        result = prepare_stage3_pair(
            args.run_dir, args.base_config, args.teacher_cache, family=args.family
        )
    elif args.command == "stage3-teacher-pool":
        result = build_teacher_pool(
            args.run_dir, args.teacher_export, args.weight, pool_id=args.pool_id
        )
    elif args.command == "roll-stage2":
        result = prepare_and_supervise_stage2(
            args.run_dir,
            args.parent_result,
            worker_prefix=args.worker_prefix,
            max_workers=args.max_workers,
            poll_seconds=args.poll_seconds,
            max_runtime_seconds=args.max_runtime_seconds,
            lr_multiplier=args.lr_multiplier,
        )
    elif args.command == "roll-stage3":
        result = prepare_and_supervise_stage3(
            args.run_dir,
            args.base_config,
            args.teacher_cache,
            family=args.family,
            pair_count=args.pairs,
            worker_prefix=args.worker_prefix,
            max_workers=args.max_workers,
            poll_seconds=args.poll_seconds,
            max_runtime_seconds=args.max_runtime_seconds,
        )
    elif args.command == "status":
        result = status_summary(args.run_dir)
    elif args.command == "stop-worker":
        result = stop_worker(args.run_dir, args.worker_id)
    elif args.command == "adopt-worker":
        result = adopt_external_worker(args.run_dir, args.receipt)
    elif args.command == "adopt-completed-fit":
        result = adopt_completed_fit(
            args.run_dir,
            args.attempt_id,
            config_path=args.config,
            approved_profile_path=args.approved_profile,
            owner_receipt_path=args.owner_receipt,
            terminal_receipt_path=args.terminal_receipt,
            gpu_key=args.gpu_key,
        )
    elif args.command == "supervise":
        result = supervise_stage1(
            args.run_dir,
            worker_prefix=args.worker_prefix,
            max_workers=args.max_workers,
            poll_seconds=args.poll_seconds,
            max_trials=args.max_trials,
        )
    elif args.command == "stage4-ensemble":
        result = run_stage4_ensemble(
            args.run_dir,
            args.data_root,
            args.auxiliary_split,
            args.candidate,
        )
    else:
        raise AssertionError(f"unsupported command {args.command}")
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
