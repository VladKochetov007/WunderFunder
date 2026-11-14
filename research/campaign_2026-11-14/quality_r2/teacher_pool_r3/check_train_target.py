"""Verify actual canonical dataset targets before admitting pool-distilled students."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SOURCE = Path("/home/vlad/development/wunder/research_tmp/q14_strict_source")
DATA = Path("/home/vlad/development/wunder/research_tmp/chrono_root")
SPLIT = ROOT / "research/campaign_2026-10-29/teacher_track_r1/chronological_primary.json"
TEACHERS = [("q14t_fbc_s5", 1.02), ("q14t_tlead_s5", 0.99), ("q14t_tlead_s6", 0.99)]
sys.path.insert(0, str(SOURCE / "src"))
from wnn.splits import load_campaign_split  # noqa: E402
from wnn.training.cache_identity import bind_fold_teachers  # noqa: E402
from wnn.training.datasets import Sequences  # noqa: E402
from wnn.training.environment import TrainingEnvironment  # noqa: E402


def main() -> None:
    """Bind complete caches and test producer arithmetic on three real fitting sequences."""
    for name, _ in TEACHERS:
        assert (DATA / "cache" / f"pred_{name}_train.json").is_file(), "train export incomplete"
    split = load_campaign_split(SPLIT, DATA)
    environment = TrainingEnvironment(root=DATA, source_root=SOURCE)
    bindings = bind_fold_teachers(TEACHERS, split, ("train",), environment=environment)
    with np.load(DATA / "cache/train_targets.npz") as targets:
        order = list(map(int, targets["row_group"]))
    positions = {group: index for index, group in enumerate(order)}
    selected = [split.train_groups[index] for index in (0, len(split.train_groups) // 2, -1)]
    dataset = Sequences("train", selected, teachers=TEACHERS, environment=environment)
    arrays = [
        np.load(DATA / "cache" / f"pred_{name}_train.npy", mmap_mode="r") for name, _ in TEACHERS
    ]
    maximum_error = 0.0
    for index, group in enumerate(selected):
        _, targets, _ = dataset[index]
        actual = targets[:, 2:4].numpy()
        expected = sum(
            np.asarray(array[positions[group]], dtype=np.float32) * np.float32(weight)
            for array, weight in zip(arrays, (0.34, 0.33, 0.33), strict=True)
        )
        assert np.isfinite(actual).all()
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=3e-7)
    loss_path = SOURCE / "src/wnn/training/losses.py"
    result = {
        "status": "verified_actual_canonical_train_targets",
        "teacher_scales": TEACHERS,
        "effective_pool_weights": [0.34, 0.33, 0.33],
        "canonical_divisor": 3,
        "teacher_bindings": bindings,
        "actual_dataset_groups_checked": selected,
        "max_absolute_fp32_error": maximum_error,
        "loss_source_sha256": hashlib.sha256(loss_path.read_bytes()).hexdigest(),
        "student_target": "0.1*clip(label,-2,2) +0.9*clip(pool,-2,2); alpha applied once by unchanged losses.py",
        "required_student_args": {
            "distill_alpha": 0.9,
            "distill_alpha_far": None,
            "distill_boot_h": 0,
            "distill_reanchor": 0.0,
            "init": "",
        },
        "fits_launched": 0,
    }
    (OUT / "TRAIN_TARGET_CHECK_R1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "max_error": maximum_error}))


if __name__ == "__main__":
    main()
