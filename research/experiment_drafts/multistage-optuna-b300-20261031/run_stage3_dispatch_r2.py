from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from wnn.multistage_search import atomic_write_json, file_sha256, read_json
from wnn.multistage_search_reviewed_successor_validation import (
    _is_live_process,
    ensure_gpu_slot_released,
)
from wnn.multistage_search_runtime import launch_detached_worker

ROOT = Path("/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1")
ANCHORS = (0.05, 0.1, 0.25, 0.5, 1.0, 0.1)
BASE_SHA = "31424878cc230cb2b5ca3b46f25b305495a83b6975f8814927cebcff7b0e75f2"


def main() -> None:
    """Dispatch the existing twelve prepared trials after the queued-anchor count repair."""
    root, campaign, operator = ROOT, ROOT / "campaign-r2", ROOT / "operator-r12"
    controller = campaign / "stage3/controller-r2"
    controller.mkdir(exist_ok=False)
    pending: list[int] = []
    inputs: list[dict[str, object]] = []
    for pair_index, weight in enumerate(ANCHORS):
        kd_number, control_number = 2 * pair_index, 2 * pair_index + 1
        pair_path = (
            campaign / "stage3/pairs" / f"pair-{kd_number:06d}-{control_number:06d}" / "pair.json"
        )
        pair = read_json(pair_path)
        if pair.get("base_config_sha256") != BASE_SHA:
            raise ValueError("prepared pair changed the frozen GRU256x2 base")
        for variant in pair["variants"]:
            number = int(variant["trial_number"])
            config_path = Path(str(variant["config_path"]))
            config = read_json(config_path)
            if (
                number not in {kd_number, control_number}
                or file_sha256(config_path) != variant["config_sha256"]
                or config["epochs"] != 2
                or config["seed"] != 1301 + kd_number
                or config["kd_weight"] != (weight if number == kd_number else 0)
            ):
                raise ValueError("prepared pair differs from the declared matched treatment")
            pending.append(number)
        inputs.append({"path": str(pair_path), "sha256": file_sha256(pair_path)})
    if sorted(pending) != list(range(12)):
        raise ValueError("the wave must contain exactly the twelve existing paired trials")
    driver = root / "requests/profile_review_launch_successor_r6.py"
    text = driver.read_text()
    if (
        "review_multistage_profile_r6.py" not in text
        or 'review_root = root / "review-r6"' not in text
    ):
        raise ValueError("profile driver does not use the frozen independent R6 checker")
    atomic_write_json(
        controller / "prepared.json",
        {"pairs": inputs, "driver_sha256": file_sha256(driver), "trial_numbers": pending},
        replace=False,
    )
    reserve_start = float(read_json(campaign / "campaign_deadline.json")["reserve_start_epoch"])
    reservations = {"0": 4, "4": 5, "7": 6}
    lanes: dict[str, dict[str, object]] = {}
    while pending:
        if time.time() >= reserve_start - 900:
            raise RuntimeError("profile admission closed before collection reserve")
        for gpu in map(str, range(8)):
            if not pending:
                break
            if gpu in lanes and _is_live_process(lanes[gpu]):
                continue
            reserved = reservations.get(gpu)
            if (
                reserved is not None
                and not (root / "receipts" / f"stage2-trial-{reserved:06d}.child.json").exists()
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
            print("STAGE3_PROFILE_FIT_DISPATCHED", number, gpu, record["pid"], flush=True)
        time.sleep(3)
    atomic_write_json(
        controller / "all_drivers_dispatched.json",
        {"count": 12, "at_epoch": time.time()},
        replace=False,
    )
    print("STAGE3_ALL_TWELVE_DRIVERS_DISPATCHED", flush=True)


if __name__ == "__main__":
    main()
