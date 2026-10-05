"""Post-score descriptive class breakdown for every recurrent recipe/seed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent / "activation_optimizer"))

import score_activation as common  # noqa: E402

from wnn.gmetric import global_wp  # noqa: E402


def main() -> None:
    """Use diagnostic labels solely for descriptive evaluation, never routing."""
    output = HERE / "CLASS_DIAGNOSTICS.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite post-score diagnostics")
    verified = json.loads((HERE / "THREE_SEED_ACTIVATION_RESULT.json").read_text())
    results = {}
    for seed in verified["seeds"]:
        run = HERE / "runs" / ("six_v1" if seed == 271828 else f"seed{seed}_v1")
        report = json.loads((run / "result.json").read_text())
        nodes, _ = common.load_nodes(run, report)
        arms = {}
        for item in report["arms"]:
            arm = item["arm"]
            path = run / f"{arm}_selected_predictions.npz"
            expected = verified["attempts_and_scores"][str(seed)]["artifact_sha256"][path.name]
            if common.sha256_file(path) != expected:
                raise ValueError("selected export changed after official grading")
            with np.load(path, allow_pickle=False) as archive:
                y = common.finite_float32(archive["targets"], "raw targets")
                p = common.finite_float32(archive["predictions"], "raw predictions")
            labels = np.repeat(
                [node["label_class_diagnostic_only"] for node in nodes],
                [sequence["scored_rows"] for sequence in item["sequences"]],
            )
            classes = {}
            for class_id in range(4):
                selected = labels == class_id
                cy, cp = y[selected], p[selected]
                score = global_wp(cy[None], cp[None], np.ones((1, len(cy)), dtype=bool))
                classes[str(class_id)] = {
                    **score,
                    "components": sum(
                        node["label_class_diagnostic_only"] == class_id for node in nodes
                    ),
                    "target_weight_mass": np.abs(cy.astype(np.float64)).sum(axis=0).tolist(),
                }
            arms[arm] = classes
        results[str(seed)] = arms
    result = {
        "scope": "Post-score exploratory diagnostic of all72classscores; four exposed components per class. Labels are target-derived evaluation diagnostics, never inputs or deployed routing. No class CI or public-mixture forecast.",
        "scores": results,
        "source_sha256": common.sha256_file(Path(__file__)),
        "verified_result_sha256": common.sha256_file(HERE / "THREE_SEED_ACTIVATION_RESULT.json"),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    for seed, arms in results.items():
        baseline = arms["zero__muon_adamw"]
        candidate = arms["p10__muon_adamw"]
        print(seed, {c: candidate[c]["wp"] - baseline[c]["wp"] for c in baseline})


if __name__ == "__main__":
    main()
