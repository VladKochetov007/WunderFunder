"""Write proposed launch argv; root must freeze recipes and own GPU launches."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main() -> None:
    clean = ["--split-manifest", "research/campaign_2026-09-29/artifacts/splits/fold0.json", "--checkpoint-policy", "fixed", "--checkpoint-epoch", "0", "--no-eval-during-fit", "--valid-fold", "0", "--n-valid-folds", "4", "--eval-source", "vfold", "--full-valid"]
    train = [".venv/bin/python", "scripts/train_rnn.py"]
    continuation = ["--init", "c20260929_since_s3_f0@ep3", "--inherit-init-preprocessing", "--train-source", "train+valid", "--epochs", "1", "--seed", "3", "--batch", "32", "--chunk", "2000", "--lr", "0.001", "--wd", "0.01", "--loss", "wmse", "--aux-struct", "0.3", "--aux-next", "0.3", "--ema-teacher", "1", "--ema-decay", "0.999", "--eval-ema", "--propensity", "c20260929_prop_f0", "--prop-power", "2", "--prop-eps", "0.03", "--valid-true-mask", "0.03", "--since-weight", "32", "--since-floor", "0.25", "--loss-norm", "chunk", "--fast", "--precision", "fp16", "--spread-cache-root", "research/campaign_2026-09-30/spread_aux"]
    ft = ["--inherit-init-preprocessing", "--train-source", "valid", "--epochs", "1", "--seed", "3", "--batch", "16", "--chunk", "2000", "--wd", "0.01", "--loss", "gcorr", "--mask-weight", "--aux-struct", "0.3", "--aux-next", "0.3", "--spread-aux-mode", "original", "--spread-cache-root", "", "--ema-teacher", "0", "--ema-decay", "0.999", "--propensity", "", "--prop-power", "1", "--prop-eps", "0", "--valid-true-mask", "-1", "--since-weight", "0", "--since-floor", "0.25", "--loss-norm", "chunk", "--fast", "--precision", "fp16"]
    jobs = []
    members = {}
    for mode in ["proxy_known", "current_known"]:
        short = mode.removesuffix("_known")
        name = f"c20260930_spread_{short}_f0"
        final = f"c20260930_ft_spread_{short}_f0"
        jobs.append({"stage": "continuation", "mode": mode, "argv": train + ["--name", name] + continuation + ["--spread-aux-mode", mode] + clean})
        jobs.append({"stage": "FT_head", "mode": mode, "argv": train + ["--name", final + "h", "--init", name + "@ep0"] + ft + clean + ["--freeze", "inp,gru", "--lr", "0.0005"]})
        jobs.append({"stage": "FT_full", "mode": mode, "argv": train + ["--name", final, "--init", final + "h@ep0"] + ft + clean + ["--freeze", "", "--lr", "0.0002"]})
        members[mode] = [["c20260929_ft_since_s4_f0@ep0", 0.5], [final + "@ep0", 0.5]]
    # Complete both equal fitting paths before predictions or scoring.
    jobs.sort(key=lambda job: ["continuation", "FT_head", "FT_full"].index(job["stage"]))
    predictions = [{"mode": mode, "argv": [".venv/bin/python", "scripts/predict_run.py", f"c20260930_ft_spread_{mode.removesuffix('_known')}_f0", "valid", "--epoch", "0"]} for mode in ["proxy_known", "current_known"]]
    proposal = {"status": "proposed, root and critic freeze required before fitting", "launch_authority": "root only, after all fourfold clean parents/FT/prediction jobs finish", "budget": "root must freeze equal per-stage caps before fit; failure of either arm invalidates mechanism comparison", "initial_checkpoint": "c20260929_since_s3_f0@ep3", "output_width": 23, "spread_bins": 12, "training_jobs_in_order": jobs, "prediction_jobs_after_all_training": predictions, "replacement_pairs": members, "unchanged_pair": [["c20260929_ft_since_s4_f0@ep0", 0.5], ["c20260929_ft_since_s3_f0@ep0", 0.5]], "screen": {"candidate_vs_unchanged_Global_WP": ">=+0.0005", "candidate_vs_proxy_control": "positive", "same_exact_rows": True, "weights_frozen": True}, "FT_semantics": "Both stages in BOTH arms retain inherited aux_struct=0.3 and aux_next=0.3, n_out23, original future-proxy spread labels without known-code masks. FT uses original gcorr/true-mask recipe, seed3; no EMA teacher, eval_ema false, no propensity/since weighting. The default false eval_ema is intentional and matches historical FT."}
    (HERE / "launch_recipes_proposed.json").write_text(json.dumps(proposal, indent=2) + "\n")
    lines = ["# Proposed commands only; do not execute until root freezes recipes and releases controls gate."]
    for job in jobs + predictions:
        lines.extend(["", "# " + job.get("stage", "prediction") + " " + job["mode"], shlex.join(job["argv"])])
    (HERE / "launch_commands_proposed.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
