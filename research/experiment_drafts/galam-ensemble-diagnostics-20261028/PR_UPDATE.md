Full official-valid fixed-blend diagnostic completed; seven original GALAM fits continue unchanged.

| Frozen output | t0 | t1 | WP | Delta vs GRU |
|---|---:|---:|---:|---:|
| GRU256x2 epoch8 | .672775003 | .682937244 | .677856123 | — |
| 80%GRU +20%GALAM-LSTM128 epoch3 | .676426908 | .686012449 | **.681219679** | **+.003363555** |
| 80%GRU +20%GALAM-GRU72 epoch3 | .676246162 | .685740803 | **.680993482** | **+.003137359** |

All1873official-valid sequences/3528273scored rows with aligned original masks;
standalone scores reproduced. Independent rescore of hashed LSTM128/GRU arrays
with a separate float64 formula agrees within2e-13. Fixed10/30%sensitivities are
reported; primary20%was frozen before outcomes. No coefficient fitting or
calibration on valid.

The reference is the clean GRU benchmark, not the full team ensemble. Valid
selected the checkpoints, so these are exposed single-seed development results,
not hidden-transfer or deployment evidence. Zero new neural fits. The diagnostic
worker exited and freedGPU0; seven fits retain their original startticks/configs.

[Full diagnostic report and evidence](https://github.com/VladKochetov007/WunderFunder/blob/experiment/galam-l4-quality8-20261028/research/experiment_drafts/galam-ensemble-diagnostics-20261028/RESULTS.md)
contains the complete fixed panel, source/weights/array/trace identities,
independent review, real inference CUDA trace and costs, live-fit continuity,
BN/context/objective audit and interpretation limits.
