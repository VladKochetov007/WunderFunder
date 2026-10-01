"""Critic checks (CPU only) for scripts/screen_score.py against the frozen campaign scorer.

    WNN_CPU_PCT=7 WNN_MEM_MAX=4G CUDA_VISIBLE_DEVICES='' scripts/limited .venv/bin/python \
        research/campaign_2026-10-01/critic/check_screen.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import screen_score  # noqa: E402
from wnn.splits import load_campaign_split  # noqa: E402

out = {}
frozen = json.loads((ROOT / "research/campaign_2026-09-30/audit/reference_fourfold_report.json").read_text())
screen = json.loads((ROOT / "research/campaign_2026-10-01/screens/validate_reference.json").read_text())
report = screen_score.run(screen)
ref = report["pooled"]["exact_global_wp"]["reference"]["wp"]
out["pooled_reference_wp"] = ref
out["pooled_equals_frozen_exactly"] = ref == frozen["pooled"]["exact_global_wp"]["reference"]["wp"] == 0.6895232393809754
out["per_fold_equal"] = all(report["folds"][k]["exact_global_wp"]["reference"] == frozen["folds"][k]["exact_global_wp"]["reference"]
                            for k in frozen["folds"])
out["by_class_equal"] = all(report["pooled"]["by_class"][c]["reference"] == frozen["pooled"]["by_class"][c]["reference"]
                            for c in frozen["pooled"]["by_class"])
out["n_components"] = report["pooled"]["paired_component_bootstrap"]["reference_minus_s3_only"]["n_components"]

# fold guard: a fold-1 model scored on fold 0 must be refused
try:
    screen_score.run({"folds": [0], "arms": {"a": {"c20260929_ft_since_s3_f1": 1.0}, "b": {"c20260929_ft_since_s3_f0": 1.0}},
                      "comparisons": [["a", "b"]], "n_boot": 10})
    out["wrong_fold_member_refused"] = False
except ValueError as error:
    out["wrong_fold_member_refused"] = str(error)

# duplicated fold in the screen: the frozen scorer refuses duplicated evaluation rows; does the screen?
try:
    dup = screen_score.run({"folds": [0, 0], "arms": {"a": {"c20260929_ft_since_s3_f{fold}": 1.0},
                                                      "b": {"c20260929_ft_since_s4_f{fold}": 1.0}},
                            "comparisons": [["a", "b"]], "n_boot": 50})
    out["duplicate_fold_refused"] = False
    out["duplicate_fold_n_sequences"] = dup["n_sequences"]
except ValueError as error:
    out["duplicate_fold_refused"] = str(error)

# components are global ids: evaluation components of different folds never coincide
seen = {}
for fold in range(4):
    split = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json", ROOT)
    comp = {c for s, g, c in split.components if s == "valid" and g in set(split.evaluation_groups)}
    seen[fold] = comp
out["eval_components_disjoint_across_folds"] = all(not (seen[a] & seen[b]) for a in seen for b in seen if a < b)
print(json.dumps(out, indent=1))
