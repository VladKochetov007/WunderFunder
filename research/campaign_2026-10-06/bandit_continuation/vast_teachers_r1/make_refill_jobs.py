"""Write waves 3-4 as refill job files (one mechanism each, fold1 from scratch, combo recipe otherwise)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_bundle as B  # noqa: E402

WAVES = {
    "w0": {
        # priority: does a capacity-diverse scratch teacher add to the best student?  (fold1 finding: +0.0029)
        "vt_f0_gru384": {"--hidden": "384", "fold": 0},
        "vt_f0_gru512": {"--hidden": "512", "fold": 0},
        "vt_f1_gru384_s4": {"--hidden": "384", "--seed": "4"},
        "vt_f1_gru384_s5": {"--hidden": "384", "--seed": "5"},
        "vt_f1_gru320": {"--hidden": "320"},
        "vt_f1_gru448": {"--hidden": "448"},
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

TRACE = ["--trace-every", "10", "--trace-state-every", "200", "--intra-epoch-points", "3", "--intra-snapshots"]


def with_trace(change: dict) -> dict:
    merged = dict(change)
    merged["+"] = [*change.get("+", []), *TRACE]
    merged["-"] = ["--no-eval-during-fit"]
    merged.pop("snapshots", None)
    return merged


def main() -> None:
    out = Path(__file__).resolve().parent / "refill_jobs"
    out.mkdir(exist_ok=True)
    for wave, variants in WAVES.items():
        for index, (name, change) in enumerate(variants.items()):
            snapshots = change.get("snapshots", [])
            fold = int(change.get("fold", 1))
            change = {k: v for k, v in change.items() if k != "fold"}
            change = with_trace({k: v for k, v in change.items() if k != "snapshots"}) if wave in ("w5", "w6", "w7") else change
            steps = [
                {"label": "fit", "timeout": 4 * 3600, "argv": B.argv_for(name, change, fold)},
                {"label": "export", "timeout": 1800, "argv": B.export_argv(name, change.get("--checkpoint-epoch", "3"))},
            ]
            steps += [{"label": f"export_e{k}", "timeout": 1800, "argv": B.export_argv(name, str(k), f"{name}__ep{k}")}
                      for k in snapshots]
            job = {"name": name, "steps": steps}
            (out / f"{wave}_{index}_{name}.json").write_text(json.dumps(job, indent=1))
    print(sorted(p.name for p in out.glob("*.json")))


if __name__ == "__main__":
    main()
