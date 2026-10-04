"""Score the frozen E427 arms on declared historical panels, without selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-09-30/audit"))

import campaign_score as cs  # noqa: E402
from e427_contracts import validate_screen_design  # noqa: E402
from wnn.splits import file_sha256, load_campaign_split  # noqa: E402

CAMPAIGN = ROOT / "research/campaign_2026-10-04"
STATS_PATH = CAMPAIGN / "checks/e427_context_stats_s3_v1.json"
MANIFEST = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
DESIGN = CAMPAIGN / "staging/e427_pipeline_v1/specs/e427_screen_v1.json"
ARMS = ("aligned", "stale", "noop")
RUNS = {arm: f"c20261004_e427_{arm}_s3_1ep_v2" for arm in ARMS}
EXPECTED_GROUPS = list(range(1873))


def digest(path: Path) -> str:
    return file_sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    validate_screen_design(design, cs.METRIC_ID)
    target_path = ROOT / "cache/valid_targets.npz"
    targets = cs.mmap_stored_npz(target_path)
    groups = [int(group) for group in targets["row_group"]]
    if groups != EXPECTED_GROUPS:
        raise ValueError("VALID target order/domain changed")
    positions = {group: index for index, group in enumerate(groups)}
    split = load_campaign_split(MANIFEST, ROOT)
    if split.fold != 0:
        raise ValueError("E427 panel requires the registered fold-0 manifest")
    fold0 = list(map(int, split.evaluation_groups))
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    fit_valid = set(map(int, stats["fit_row_groups"]["valid"]))
    held = list(map(int, stats["evaluation_row_groups_excluded"]))
    if (len(fold0) != 468 or len(held) != 368 or len(fit_valid) != 1505
            or len(set(held) & fit_valid) != 0):
        raise ValueError("fixed E427 panel membership/counts changed")
    nonfit = [group for group in fold0 if group not in fit_valid]
    if len(nonfit) != 94 or set(nonfit) != set(held) & set(fold0):
        raise ValueError("E427 fold0 nonfit subset is not the preregistered 94 groups")
    panels = {
        "fold0_primary_historically_exposed": fold0,
        "stats_excluded_368_historically_exposed": held,
        "fold0_nonfit_94_diagnostic": nonfit,
    }
    predictions: dict[str, np.ndarray] = {}
    bindings = {}
    for arm, run in RUNS.items():
        path = ROOT / "cache" / f"pred_{run}_valid.npy"
        side_path = path.with_suffix(".json")
        side = json.loads(side_path.read_text(encoding="utf-8"))
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if (arr.shape != (1873, 20_000, 2) or arr.dtype != np.float32
                or side.get("schema") != "wnn.e427-fixed-epoch0-fp32-prediction-v1"
                or side.get("run") != run or side.get("arm") != arm
                or side.get("epoch") != 0 or side.get("inference_precision") != "fp32"
                or side.get("row_groups") != groups
                or side.get("prediction_sha256") != digest(path)):
            raise ValueError(f"{arm} prediction shape/domain/sidecar/hash is invalid")
        predictions[arm] = arr
        bindings[arm] = {
            "run": run,
            "prediction_sha256": digest(path),
            "sidecar_sha256": digest(side_path),
            "checkpoint_sha256": side.get("checkpoint_sha256"),
            "provenance_sha256": side.get("provenance_sha256"),
        }
    component_map = {group: component for split_name, group, component in split.components
                     if split_name == "valid"}
    if not set(fold0 + held) <= set(component_map):
        raise ValueError("fold0 component map lacks a scored row")
    targets_by_group = {group: positions[group] for group in groups}
    design_summary = {"comparisons": [["aligned", "stale"], ["aligned", "noop"]],
                      "n_boot": design["bootstrap_replicates"],
                      "bootstrap_seed": design["bootstrap_seed"]}
    reports = {}
    for panel_name, panel_groups in panels.items():
        moment_rows = {arm: [] for arm in ARMS}
        soft_rows = {str(lam): {arm: [] for arm in ARMS} for lam in cs.SOFTENED}
        block_rows = {arm: [] for arm in ARMS}
        class_rows, components = [], []
        selected_rows = 0
        for group in panel_groups:
            index = targets_by_group[group]
            current = {arm: predictions[arm][index] for arm in ARMS}
            row = cs.score_sequence(targets["y"][index], targets["mask"][index], current)
            for arm in ARMS:
                moment_rows[arm].append(row["moments"][arm])
                block_rows[arm].append(row["blocks"][arm])
                for lam in soft_rows:
                    soft_rows[lam][arm].append(row["soft"][lam][arm])
            selected_rows += row["selected_rows"]
            class_rows.append(int(targets["market_class"][index]))
            components.append(component_map[group])
        moment_arrays = {name: np.stack(rows) for name, rows in moment_rows.items()}
        soft_arrays = {lam: {name: np.stack(rows) for name, rows in values.items()}
                       for lam, values in soft_rows.items()}
        reports[panel_name] = {
            "n_sequences": len(panel_groups),
            "selected_rows": selected_rows,
            "fit_overlap_count": len(set(panel_groups) & fit_valid),
            "not_fit_count": len(set(panel_groups) - fit_valid),
            **cs.summary(moment_arrays, soft_arrays, block_rows, np.asarray(class_rows),
                         np.asarray(components), design_summary),
        }
    result = {
        "schema": "wnn.e427-component-panel-report-v1",
        "metric_id": cs.METRIC_ID,
        "preregistration_sha256": digest(DESIGN),
        "screen_sha256": digest(Path(__file__)),
        "targets_sha256": digest(target_path),
        "fold0_manifest_sha256": digest(MANIFEST),
        "context_stats_sha256": digest(STATS_PATH),
        "n_boot": design["bootstrap_replicates"],
        "bootstrap_seed": design["bootstrap_seed"],
        "comparisons": ["aligned_minus_stale", "aligned_minus_noop"],
        "qualification": (
            "historically exposed panels; descriptive causal-arm contrasts only; "
            "no independent holdout or generalization claim; no model/epoch selection"
        ),
        "panel_membership": {
            name: {"groups_sha256": hashlib.sha256(
                ",".join(map(str, values)).encode()
            ).hexdigest(), "n": len(values)} for name, values in panels.items()
        },
        "overlap": {
            "fold0_n": len(fold0), "fold0_fit_overlap": len(set(fold0) & fit_valid),
            "fold0_nonfit": len(nonfit), "stats_excluded_n": len(held),
            "fit_valid_n": len(fit_valid),
        },
        "bindings": bindings,
        "panels": reports,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "panels": list(reports)}, sort_keys=True))


if __name__ == "__main__":
    main()
