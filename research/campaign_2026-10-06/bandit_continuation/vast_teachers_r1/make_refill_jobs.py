"""Write waves 3-4 as refill job files (one mechanism each, fold1 from scratch, combo recipe otherwise)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_bundle as B  # noqa: E402

PARENT = "c20261003_combo_s3_f1@ep3"
WAVES = {
    "w11": {
        # scale-then-distill: matched H230-recipe students that differ only in the scratch teachers feeding the cache
        "vt_f1_dsq_t256": {"pipeline": ["--arm", "t256", "--fold", "1", "--teachers", "vt_f1_seed4,vt_f1_seed5", "--parent", PARENT]},
        "vt_f1_dsq_t384": {"pipeline": ["--arm", "t384", "--fold", "1", "--teachers", "vt_f1_gru384_s4,vt_f1_gru384_s5", "--parent", PARENT]},
        "vt_f1_dsq_mix": {"pipeline": ["--arm", "mix", "--fold", "1", "--teachers", "vt_f1_gru384,vt_f1_gru448", "--parent", PARENT]},
        "vt_f1_dsq_none": {"pipeline": ["--arm", "none", "--fold", "1", "--teachers", "", "--parent", PARENT, "--no-teacher"]},
    },
    "w10": {
        # fold0 transfer of the recipe-fidelity arms (lane A holds the fold0 assets)
        "vt_f0_mu3": {"fold": 0, "+": ["--muon-lr-mult", "3"]},
        "vt_f0_mu6": {"fold": 0, "+": ["--muon-lr-mult", "6"]},
    },
    "w10b": {
        # recipe fidelity from the teammate evidence ledger: Muon LR multiplier and EMA-evaluated weights, paired controls
        "vt_f1_ctrl_s3": {},
        "vt_f1_mu3": {"+": ["--muon-lr-mult", "3"]},
        "vt_f1_mu3_s4": {"--seed": "4", "+": ["--muon-lr-mult", "3"]},
        "vt_f1_mu6": {"+": ["--muon-lr-mult", "6"]},
        "vt_f1_mu6_s4": {"--seed": "4", "+": ["--muon-lr-mult", "6"]},
        "vt_f1_ema": {"+": ["--ema-teacher", "1.0", "--eval-ema"]},
        "vt_f1_mu3_ema": {"+": ["--muon-lr-mult", "3", "--ema-teacher", "1.0", "--eval-ema"]},
        "vt_f1_gru384_mu3": {"--hidden": "384", "+": ["--muon-lr-mult", "3"]},
    },
    "w9": {
        # data-weighting recipe knobs never swept for teachers
        "vt_f1_since128": {"--since-weight": "128"},
    },
    "w8": {
        # teacher-side target weighting (tail power / zero-label floor / class 3), regularization before scale,
        # untried architectures and input views, alternative losses
        "vt_f1_lw15": {"+": ["--label-weight-power", "1.5"]},
        "vt_f1_lwfloor": {"+": ["--label-weight-floor", "0.1"]},
        "vt_f1_cls3w2": {"+": ["--class-weight", "1,1,1,2"]},
        "vt_f1_gru384_drop2": {"--hidden": "384", "--dropout": "0.2"},
        "vt_f1_gru512_drop2": {"--hidden": "512", "--dropout": "0.2"},
        "vt_f1_wd05": {"--wd": "0.05"},
        "vt_f1_tcn": {"--arch": "tcn", "--layers": "10", "--optim": "adamw"},
        "vt_f1_structured": {"--arch": "structured", "--optim": "adamw"},
        "vt_f1_eventcontent": {"--arch": "event-content", "--optim": "adamw"},
        "vt_f1_rawonly": {"+": ["--raw-only"]},
        "vt_f1_corr": {"--loss": "corr"},
        "vt_f1_huber": {"--loss": "huber"},
    },
    "w1": {
        # four-fold evidence for width diversity (also feeds nested-CV stacking work) + new mechanisms on fold1
        "vt_f2_gru384": {"--hidden": "384", "fold": 2},
        "vt_f3_gru384": {"--hidden": "384", "fold": 3},
        "vt_f2_gru256": {"fold": 2},
        "vt_f3_gru256": {"fold": 3},
        # vt_f1_cls3 (class-3-only specialist) retired: teammate evidence says specialization loses 0.14 on class 3
        "vt_f1_gated384": {"--arch": "gated", "--hidden": "384"},
        "vt_f1_gated_trap": {"--arch": "gated", "--sched": "wsd", "+": ["--wsd-decay", "0.2", "--wsd-final", "0.0", "--wsd-shape", "linear"]},
    },
    "w0": {
        # priority: does a capacity-diverse scratch teacher add to the best student?  (fold1 finding: +0.0029)
        "vt_f0_gru384": {"--hidden": "384", "fold": 0},
        "vt_f0_gru512": {"--hidden": "512", "fold": 0},
        "vt_f1_gru384_s4": {"--hidden": "384", "--seed": "4"},
        "vt_f1_gru384_s5": {"--hidden": "384", "--seed": "5"},
        "vt_f1_gru320": {"--hidden": "320"},
        "vt_f1_gru448": {"--hidden": "448"},
        # fold0 diagnostics after the fold1 gain failed to transfer (student-teacher gap 0.012 vs 0.002)
        "vt_f0_gru256": {"fold": 0},
        "vt_f0_gru384_ep6": {"--hidden": "384", "--epochs": "6", "--checkpoint-epoch": "5", "fold": 0},
    },
    "w3": {
        "vt_f1_adamw": {"--optim": "adamw"},
        "vt_f1_wmsegcorr": {"--loss": "wmse_gcorr"},
        "vt_f1_encmlp": {"--enc": "mlp"},
        "vt_f1_dual": {"--arch": "dual"},
    },
    "w4": {
        "vt_f1_twoclock": {"--arch": "twoclock"},
        "vt_f1_gated": {"--arch": "gated"},
        "vt_f1_gru384_fixmix": {"--hidden": "384", "+": ["--readout-activation", "fixed_mix"]},
        "vt_f1_seed5": {"--seed": "5"},
    },
    "w5": {
        # training-dynamics telemetry arms: trace.jsonl + intra-epoch evaluations/snapshots on every run
        "vt_f1_tr_base": {},
        "vt_f1_tr_wsd": {"--sched": "wsd"},
        "vt_f1_tr_ep6": {"--epochs": "6", "--checkpoint-epoch": "5"},
        "vt_f1_tr_lr2": {"--lr": "0.002"},
        "vt_f1_tr_lrh": {"--lr": "0.0005"},
        "vt_f1_tr_swa": {"+": ["--swa-frac", "0.25"]},
        "vt_f1_tr_b16": {"--batch": "16"},
    },
    "w6": {
        # large-small-large-small: SGDR cosine cycles; every cycle minimum is a snapshot-ensemble member
        "vt_f1_tr_sgdr4": {"--sched": "sgdr", "+": ["--restarts", "3"], "snapshots": [1, 2]},
        "vt_f1_tr_sgdr2": {"--sched": "sgdr", "+": ["--restarts", "1"], "snapshots": [1]},
        "vt_f1_tr_sgdr8": {"--sched": "sgdr", "+": ["--restarts", "7"], "snapshots": [0, 1, 2]},
    },
    "w7": {
        # optimizer comparison with LR response: AdamW at 3 LRs against the Muon trio of w5 (1e-3, 2e-3, 5e-4)
        "vt_f1_tr_adamw_lr1": {"--optim": "adamw"},
        "vt_f1_tr_adamw_lr2": {"--optim": "adamw", "--lr": "0.002"},
        "vt_f1_tr_adamw_lrh": {"--optim": "adamw", "--lr": "0.0005"},
        "vt_f1_tr_muonhidden": {"--muon-scope": "hidden"},
    },
}

LANE = os.environ.get("VT_LANE", "")
LANE_WAVES = {"b": ("w10b", "w8", "w9")}

TRACE = ["--trace-every", "10", "--trace-state-every", "200", "--intra-epoch-points", "3", "--intra-snapshots"]


def with_trace(change: dict) -> dict:
    merged = dict(change)
    merged["+"] = [*change.get("+", []), *TRACE]
    merged.pop("snapshots", None)
    return merged


def build_job(name: str, change: dict, wave: str = "bl") -> dict:
    """One refill job from a variant spec; ``wave`` in w5/w6/w7 adds telemetry and the snapshot-curve step."""
    if "pipeline" in change:
        return {"name": name, "steps": [{"label": "pipeline", "timeout": 4 * 3600,
                                         "argv": [B.PY, f"{B.W}/scripts/distill_pipeline.py", *change["pipeline"]]}]}
    snapshots = change.get("snapshots", [])
    fold = int(change.get("fold", 1))
    change = {k: v for k, v in change.items() if k != "fold"}
    change = with_trace({k: v for k, v in change.items() if k != "snapshots"}) if wave in ("w5", "w6", "w7") else change
    epoch = change.get("--checkpoint-epoch", "3")
    steps = [
        {"label": "fit", "timeout": 6 * 3600, "argv": B.argv_for(name, change, fold)},
        {"label": "export", "timeout": 1800, "argv": B.export_argv(name, epoch)},
    ]
    if wave in ("w5", "w6", "w7"):
        steps.insert(1, {"label": "curve", "timeout": 5400, "argv": [
            B.PY, f"{B.W}/scripts/snapshot_curve.py", "--run", name, "--split",
            f"{B.W}/research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"]})
    steps += [{"label": f"export_e{k}", "timeout": 1800, "argv": B.export_argv(name, str(k), f"{name}__ep{k}")}
              for k in snapshots]
    return {"name": name, "steps": steps}


def main() -> None:
    out = Path(__file__).resolve().parent / (f"refill_jobs_{LANE}" if LANE else "refill_jobs")
    out.mkdir(exist_ok=True)
    for wave, variants in WAVES.items():
        if LANE and wave not in LANE_WAVES.get(LANE, ()):
            continue
        if not LANE and wave in sum(LANE_WAVES.values(), ()):
            continue
        for index, (name, change) in enumerate(variants.items()):
            (out / f"{wave}_{index}_{name}.json").write_text(json.dumps(build_job(name, change, wave), indent=1))
    print(sorted(p.name for p in out.glob("*.json")))


if __name__ == "__main__":
    main()
