from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from wnn.multistage_search import atomic_write_json, read_json
from wnn.multistage_search_reviewed_successor_validation import (
    _is_live_process,
    ensure_gpu_slot_released,
)
from wnn.multistage_search_runtime import launch_detached_worker
from wnn.multistage_search_study import open_study
from wnn.multistage_search_successor_prepare import prepare_stage3_pair

ROOT = Path("/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1")
KD_ANCHORS = (0.05, 0.1, 0.25, 0.5, 1.0, 0.1)


def make_r6_driver(root: Path) -> Path:
    """Freeze a distinct driver that reviews actual profiles with the independent R6 checker."""
    old = (root / "requests/profile_review_launch_successor.py").read_text()
    old = old.replace(
        'review_root = operator / "research/experiment_drafts/multistage-optuna-b300-20261031"',
        'review_root = root / "review-r6"',
    )
    old = old.replace("review_multistage_profile_r4.py", "review_multistage_profile_r6.py")
    old = old.replace(
        "PROFILE_REVIEW_POLICY_AUTOMATED_R4.json", "PROFILE_REVIEW_POLICY_AUTOMATED_R6.json"
    )
    old = old.replace("profile_review_receipt_r4.json", "profile_review_receipt_r6.json")
    old = old.replace(
        'str(review_root / "launch_reviewed_successor_fit.py")',
        'str(operator / "research/experiment_drafts/multistage-optuna-b300-20261031/launch_reviewed_successor_fit.py")',
    )
    path = root / "requests/profile_review_launch_successor_r6.py"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(old)
    return path


def main() -> None:
    """Prepare six frozen matched pairs, then fill only released owned GPU slots."""
    root, campaign, operator = ROOT, ROOT / "campaign-r2", ROOT / "operator-r9"
    deadline = read_json(campaign / "campaign_deadline.json")
    reserve_start = float(deadline["reserve_start_epoch"])
    pool_record = read_json(campaign / "stage3/pools/gru-lstm-transformer-uniform-r1.pool.json")
    if pool_record.get("status") != "complete" or pool_record.get("fit_members_count") != 8260:
        raise ValueError("the declared train-only three-teacher pool is not complete")
    base = (
        operator
        / "research/experiment_drafts/multistage-optuna-b300-20261031/recipes/r2/gru256x2-stage3.json"
    )
    controller = campaign / "stage3/controller-r1"
    controller.mkdir(exist_ok=False)
    atomic_write_json(
        controller / "anchors.json",
        {
            "kd_weights": list(KD_ANCHORS),
            "meaning": "cover five admitted coefficients; repeat0.1 on a different matched seed",
            "primary_valid_used": False,
        },
        replace=False,
    )
    driver = make_r6_driver(root)
    study = open_study(campaign, "stage3-anchor-preparer", stage=3)
    pending: list[int] = []
    for weight in KD_ANCHORS:
        study.enqueue_trial({"kd_weight": weight})
        pair = prepare_stage3_pair(
            campaign, base, Path(str(pool_record["cache_root"])), family="gru"
        )
        numbers = [int(item["trial_number"]) for item in pair["variants"]]
        pending.extend(numbers)
        print("STAGE3_PAIR_PREPARED", pair["pair_id"], weight, numbers, flush=True)
    atomic_write_json(controller / "prepared.json", {"trial_numbers": pending}, replace=False)
    lanes: dict[str, dict[str, object]] = {}
    # A successor's profile is bound to its GPU. Preserve the queued Stage2 order on these lanes.
    stage2_reservations = {"0": 4, "4": 5, "7": 6}
    while pending:
        if time.time() >= reserve_start - 900:
            raise RuntimeError("new profile admission closed before collection reserve")
        for gpu in map(str, range(8)):
            if not pending:
                break
            if gpu in lanes and _is_live_process(lanes[gpu]):
                continue
            reserved_trial = stage2_reservations.get(gpu)
            if (
                reserved_trial is not None
                and not (
                    root / "receipts" / f"stage2-trial-{reserved_trial:06d}.child.json"
                ).exists()
            ):
                continue
            try:
                ensure_gpu_slot_released(campaign, gpu)
            except RuntimeError:
                continue
            number = pending.pop(0)
            command = [
                "timeout",
                "--kill-after=5",
                "900",
                sys.executable,
                str(driver),
                "--runtime-root",
                str(root),
                "--operator-root",
                str(operator),
                "--stage",
                "3",
                "--trial-number",
                str(number),
                "--gpu-key",
                gpu,
            ]
            record = launch_detached_worker(
                command,
                cwd=operator,
                env=os.environ.copy(),
                receipt_path=controller / f"trial-{number:06d}.driver.json",
                metadata={"stage": 3, "trial_number": number, "assigned_gpu_key": gpu},
            )
            lanes[gpu] = record
            print("STAGE3_PROFILE_AND_FIT_DRIVER", number, gpu, record["pid"], flush=True)
        time.sleep(3)
    atomic_write_json(
        controller / "all_drivers_dispatched.json",
        {"count": 12, "at_epoch": time.time()},
        replace=False,
    )
    print("STAGE3_ALL_TWELVE_DRIVERS_DISPATCHED", flush=True)


if __name__ == "__main__":
    main()
