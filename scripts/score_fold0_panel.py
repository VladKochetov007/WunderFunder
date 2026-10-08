"""Score one run's fixed-epoch valid predictions on the frozen fold-0 evaluation panel.

    scripts/limited .venv/bin/python scripts/score_fold0_panel.py RUN --epoch 1 \
        [--reference RUN@EPOCH ...] [--blend RUN@EPOCH*W,RUN@EPOCH*W] --out grade.json

Reuses the official metric (``wnn.gmetric.global_wp``), the frozen split, the
``cache/valid_targets.npz`` labels/mask and the bandit grader's prediction-sidecar
validation, then reports paired component-bootstrap contrasts against references.
The evaluation groups are a repeatedly exposed development panel: results are
screening evidence, never a public forecast.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_bandit_continuation import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    SEQUENCE_ROWS,
    _official_mask,
    _read_json,
    _validate_prediction_artifact,
    compare_arrays,
)

from wnn.gmetric import METRIC_ID, global_wp  # noqa: E402
from wnn.splits import file_sha256, load_campaign_split  # noqa: E402

DEFAULT_SPLIT = "research/campaign_2026-09-29/artifacts/splits/fold0.json"


def _member(run_at: str, split, row_groups: list[int], root: Path) -> tuple[np.memmap, dict]:
    """Load one fold-0-clean RUN@EPOCH prediction after membership and sidecar checks."""
    run, _, epoch_text = run_at.partition("@ep")
    epoch = int(epoch_text)
    run_root = root / "artifacts" / "runs" / run
    provenance = _read_json(run_root / "provenance.json")
    fit = provenance.get("fit_row_groups")
    if not isinstance(fit, dict):
        raise ValueError(f"{run}: provenance lacks fit-row membership")
    if set(map(int, fit.get("valid", []))) & set(split.evaluation_groups):
        raise ValueError(f"{run}: evaluation groups occur in direct fit membership")
    split.assert_fit_groups(
        set(map(int, fit.get("train", []))), set(map(int, fit.get("valid", [])))
    )
    prediction_path = root / "cache" / f"pred_{run}_valid.npy"
    prediction, binding = _validate_prediction_artifact(
        prediction_path,
        _read_json(prediction_path.with_suffix(".json")),
        run=run,
        epoch=epoch,
        split_sha256=split.sha256,
        checkpoint_sha256=file_sha256(run_root / f"model_ep{epoch}.pt"),
        row_groups=row_groups,
        shape=(len(row_groups), SEQUENCE_ROWS, 2),
    )
    return prediction, binding


def _blend(spec: str, split, row_groups, root, indices) -> tuple[np.ndarray, list[dict]]:
    """Fixed-weight raw blend of members; the official scorer clips the result."""
    total = np.zeros((len(indices), SEQUENCE_ROWS, 2), dtype=np.float64)
    bindings = []
    for part in spec.split(","):
        run_at, _, weight = part.partition("*")
        prediction, binding = _member(run_at, split, row_groups, root)
        total += float(weight or 1.0) * np.asarray(prediction[indices], dtype=np.float64)
        bindings.append(binding | {"weight": float(weight or 1.0)})
    return total.astype(np.float32), bindings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="RUN@EPOCH, or a blend RUN@EPOCH*W,RUN@EPOCH*W")
    parser.add_argument("--reference", action="append", default=[], help="RUN@EPOCH or blend spec")
    parser.add_argument("--split", type=Path, default=ROOT / DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()

    split = load_campaign_split(args.split, ROOT)
    if split.fold != 0 or len(split.evaluation_groups) != 468:
        raise ValueError("expected the frozen 468-group fold-0 evaluation panel")
    target_path = ROOT / "cache" / "valid_targets.npz"
    with np.load(target_path, allow_pickle=False) as cache:
        row_groups = list(map(int, cache["row_group"]))
        position = {group: index for index, group in enumerate(row_groups)}
        indices = np.asarray([position[int(g)] for g in split.evaluation_groups], dtype=np.int64)
        targets = np.asarray(cache["y"][indices], dtype=np.float32)
        is_scored = np.asarray(cache["mask"][indices], dtype=bool)
    mask = _official_mask(is_scored)
    component_of = {int(g): int(c) for source, g, c in split.components if source == "valid"}
    components = np.asarray([component_of[int(g)] for g in split.evaluation_groups], dtype=np.int64)

    candidate, candidate_bindings = _blend(
        args.run if "@ep" in args.run else f"{args.run}@ep1", split, row_groups, ROOT, indices
    )
    report: dict[str, object] = {
        "metric_id": METRIC_ID,
        "created_utc": datetime.now(UTC).isoformat(),
        "candidate": args.run,
        "candidate_members": candidate_bindings,
        "split_manifest_sha256": split.sha256,
        "valid_targets_sha256": file_sha256(target_path),
        "evaluation_groups": len(indices),
        "components": int(len(set(components.tolist()))),
        "official": global_wp(targets, candidate, mask),
        "exposure": "fold0 excluded-468 panel; repeatedly inspected development evidence, not a public forecast",
        "contrasts": [],
    }
    for reference_spec in args.reference:
        reference, reference_bindings = _blend(reference_spec, split, row_groups, ROOT, indices)
        contrast = compare_arrays(
            targets,
            candidate,
            reference,
            is_scored,
            components,
            n_boot=args.n_boot,
            seed=BOOTSTRAP_SEED,
        )
        report["contrasts"].append(
            {
                "reference": reference_spec,
                "reference_members": reference_bindings,
                "reference_official": global_wp(targets, reference, mask),
                "paired": contrast,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({"candidate": args.run, "wp": report["official"]["wp"]}))


if __name__ == "__main__":
    main()
