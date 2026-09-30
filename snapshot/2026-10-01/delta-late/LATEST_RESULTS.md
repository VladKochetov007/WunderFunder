# Latest completed Transformer fold1 results

The captured pipeline completed at **2026-09-30 21:23:53 UTC** (2026-10-01 00:23:53 Moscow). The supplementary copy was taken at 21:25:51-21:25:53 UTC, after completion; no train/predict jobs remained at that cutoff. This status is supported by the archived `DONE` log marker.

| Run | Ordinary full-fold WP | Checkpoint |
|---|---:|---|
| tfm_f1 parent | 0.6753309203 | model_ep3.pt |
| fttfm_f1h head-only FT | 0.6757656017 | model_ep0.pt |
| fttfm_f1 full FT | 0.6748358573 | model_ep0.pt |

These are historical training-script readouts over all fold rows, not new pure-chain paired evaluations. They do not establish an improvement over C/V1. All three new prediction cache files are inventoried only, not transferred. No training or score evaluation was launched by the export task.
