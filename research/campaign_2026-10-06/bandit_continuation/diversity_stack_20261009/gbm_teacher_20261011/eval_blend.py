"""Score GBM predictions on fold-1 eval and test blends with the h230_f1_s3_ep3 student (component-disjoint halves)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from crossfold_blend import load_fold  # noqa
from grade_prediction_bank import score  # noqa
from wnn.evaluation import paired_component_moments, paired_component_bootstrap  # noqa

G = Path(__file__).resolve().parent
pred = np.load(sys.argv[1])
bank = ROOT / "research/campaign_2026-10-06/bandit_continuation/prediction_bank/BANK_FOLD1.jsonl"
y, mask, comp, models = load_fold(bank, 1, ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json", ROOT / "cache/valid_targets.npz")
name, arr, order = models["h230_s3_ep3"]
st = np.asarray(arr)[np.argsort(np.argsort(order))] if False else np.stack([np.asarray(arr[i]) for i in order]).astype(np.float32)
assert st.shape == pred.shape == y.shape, (st.shape, pred.shape, y.shape)
out = {"student": name}
out["student_score"] = score(y, mask, st); out["gbm_score"] = score(y, mask, pred)
print(out, flush=True)
std_ratio = [float(st[mask][:, k].std() / pred[mask][:, k].std()) for k in range(2)]
out["std_ratio"] = std_ratio
pred_s = pred * np.array(std_ratio, dtype=np.float32)
out["gbm_scaled_score"] = score(y, mask, pred_s)
grid = [0, .05, .1, .15, .2, .25, .3]
rng = np.random.default_rng(0)
uc = np.unique(comp); half = set(rng.choice(uc, len(uc) // 2, replace=False).tolist())
isA = np.array([c in half for c in comp])
res = {}
for tag, g in (("raw", pred), ("scaled", pred_s)):
    table = {}
    for w in grid:
        b = (1 - w) * st + w * g
        table[w] = {"all": score(y, mask, b)["wp"], "A": score(y[isA], mask[isA], b[isA])["wp"], "B": score(y[~isA], mask[~isA], b[~isA])["wp"]}
    wA = max(grid, key=lambda w: table[w]["A"]); wB = max(grid, key=lambda w: table[w]["B"])
    # cross-fit: weight chosen on A applied to B and vice versa
    sel = np.where(isA[:, None, None], (1 - wB) * st + wB * g, (1 - wA) * st + wA * g)
    ca, cb = paired_component_moments(y, sel, st, mask)
    boot = paired_component_bootstrap(ca, cb, comp, n_boot=1000, seed=0, centered=True)
    res[tag] = {"table": table, "w_from_A": wA, "w_from_B": wB, "crossfit_blend_wp": boot["candidate_wp"], "student_wp": boot["reference_wp"],
                "delta": boot["mean_diff"], "ci95": boot["ci95"], "p_le_0": boot["p_le_0"]}
    for w in (0.1, 0.2):
        b = (1 - w) * st + w * g
        ca, cb = paired_component_moments(y, b, st, mask)
        bt = paired_component_bootstrap(ca, cb, comp, n_boot=1000, seed=0, centered=True)
        res[tag][f"fixed_w{w}"] = {"delta": bt["mean_diff"], "ci95": bt["ci95"]}
    print(tag, {k: v for k, v in res[tag].items() if k != "table"}, flush=True)
out["blend"] = res
(G / "eval_blend.json").write_text(json.dumps(out, indent=1, default=float))
