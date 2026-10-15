"""Score fixed F0 position bins against the frozen H256x2 zero-state control."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from score_fold0_panel import _member  # noqa: E402

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402
from wnn.gmetric import METRIC_ID, global_wp  # noqa: E402
from wnn.splits import file_sha256, load_campaign_split  # noqa: E402

BUCKETS = (
    (99, 200),
    (200, 1000),
    (1000, 5000),
    (5000, 10_000),
    (10_000, 20_000),
)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 271828


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="initializer-only run name")
    parser.add_argument("--reference", default="student_frontier_f0s3_h256_l2_r2@ep0")
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("WNN_PROJECT_ROOT", ROOT)).resolve(),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    split_path = args.split or (
        args.data_root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    )
    split = load_campaign_split(split_path, args.data_root)
    if split.fold != 0 or len(split.evaluation_groups) != 468:
        raise ValueError("expected the fixed, repeatedly exposed F0 468-group panel")
    target_path = args.data_root / "cache/valid_targets.npz"
    with np.load(target_path, allow_pickle=False) as cache:
        all_groups = list(map(int, cache["row_group"]))
        position = {group: index for index, group in enumerate(all_groups)}
        indices = np.asarray(
            [position[int(group)] for group in split.evaluation_groups], dtype=np.int64
        )
        targets = np.asarray(cache["y"][indices], dtype=np.float32)
        scored = np.asarray(cache["mask"][indices], dtype=bool)
    candidate, candidate_binding = _member(f"{args.run}@ep0", split, all_groups, args.data_root)
    reference, reference_binding = _member(args.reference, split, all_groups, args.data_root)
    candidate_eval = np.asarray(candidate[indices], dtype=np.float32)
    reference_eval = np.asarray(reference[indices], dtype=np.float32)
    official_mask = scored & (np.arange(20_000)[None, :] >= 99)
    component_of = {
        int(group): int(component)
        for source, group, component in split.components
        if source == "valid"
    }
    components = np.asarray(
        [component_of[int(group)] for group in split.evaluation_groups], dtype=np.int64
    )
    bins = []
    for start, stop in BUCKETS:
        mask = official_mask.copy()
        mask[:, :start] = False
        mask[:, stop:] = False
        if not mask.any():
            raise ValueError(
                f"fixed position bucket contains no official scored rows: {start}:{stop}"
            )
        candidate_wp = global_wp(targets, candidate_eval, mask)
        reference_wp = global_wp(targets, reference_eval, mask)
        candidate_moments, reference_moments = paired_component_moments(
            targets, candidate_eval, reference_eval, mask
        )
        paired = paired_component_bootstrap(
            candidate_moments,
            reference_moments,
            components,
            n_boot=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
            centered=True,
        )
        bins.append(
            {
                "rows_inclusive_start_exclusive_stop": [start, stop],
                "scored_rows": int(mask.sum()),
                "candidate_global_wp_clip2_v1": candidate_wp,
                "reference_global_wp_clip2_v1": reference_wp,
                "paired_component_bootstrap_candidate_minus_zero": paired,
            }
        )
    result = {
        "format": "wnn.initializer-position-buckets",
        "version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "metric_id": METRIC_ID,
        "candidate": args.run,
        "candidate_binding": candidate_binding,
        "reference": args.reference,
        "reference_binding": reference_binding,
        "split_manifest_sha256": split.sha256,
        "valid_targets_sha256": file_sha256(target_path),
        "evaluation_groups": len(indices),
        "evaluation_components": int(len(np.unique(components))),
        "buckets": bins,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "selection_rule": "fixed bins are descriptive diagnostics; no threshold or winner selection",
        "exposure": "repeatedly inspected F0 evaluation panel; not a public forecast or fresh holdout",
        "quality_claim": None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"preserving existing position-bucket grade: {args.out}")
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"run": args.run, "metric_id": METRIC_ID, "buckets": len(bins)}))


if __name__ == "__main__":
    main()
