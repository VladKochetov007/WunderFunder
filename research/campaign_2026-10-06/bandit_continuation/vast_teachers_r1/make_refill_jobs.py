"""Write waves 3-4 as refill job files (one mechanism each, fold1 from scratch, combo recipe otherwise)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_bundle as B  # noqa: E402

WAVES = {
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
}

TRACE = ["--trace-every", "10", "--trace-state-every", "200", "--intra-epoch-points", "3", "--intra-snapshots"]


def with_trace(change: dict) -> dict:
    merged = dict(change)
    merged["+"] = [*change.get("+", []), *TRACE]
    merged["-"] = ["--no-eval-during-fit"]
    return merged


def main() -> None:
    out = Path(__file__).resolve().parent / "refill_jobs"
    out.mkdir(exist_ok=True)
    for wave, variants in WAVES.items():
        for index, (name, change) in enumerate(variants.items()):
            change = with_trace(change) if wave == "w5" else change
            job = {"name": name, "steps": [
                {"label": "fit", "timeout": 4 * 3600, "argv": B.argv_for(name, change)},
                {"label": "export", "timeout": 1800, "argv": B.export_argv(name, change.get("--checkpoint-epoch", "3"))},
            ]}
            (out / f"{wave}_{index}_{name}.json").write_text(json.dumps(job, indent=1))
    print(sorted(p.name for p in out.glob("*.json")))


if __name__ == "__main__":
    main()
