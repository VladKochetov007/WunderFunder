# GALAM contributes to the clean GRU benchmark

Full official-valid frozen-checkpoint inference completed on the freed GPU0.
All three standalone scores reproduce their monitored values. No new neural
fit, calibration, coefficient fitting or change to the seven active fits occurred.

| Frozen output | t0 | t1 | WP | Delta vs GRU |
|---|---:|---:|---:|---:|
| GRU256x2 epoch8 | 0.672775003 | 0.682937244 | 0.677856123 | — |
| 80%GRU +20%GALAM-LSTM128 epoch3 | 0.676426908 | 0.686012449 | **0.681219679** | **+0.003363555** |
| 80%GRU +20%GALAM-GRU72 epoch3 | 0.676246162 | 0.685740803 | **0.680993482** | **+0.003137359** |

Both primary mixtures exceed the predeclared useful delta+.0005 and improve
both targets. GALAM-LSTM128 alone scores0.672369413 and GALAM-GRU72 alone
0.671357523. Their mixture contribution is the relevant result. A separate
review session independently rescored the actual hashed LSTM128/GRU arrays
with chunked two-pass float64 weighted Pearson, agreeing within2e-13.
This verifies fixed-prediction computation, not training-seed robustness.

## Complete predeclared sensitivity panel

| GALAM candidate | 10% weight | **20% primary** | 30% weight |
|---|---:|---:|---:|
| LSTM128 | 0.679809721 | **0.681219679** | 0.682070749 |
| GRU72 | 0.679681451 | **0.680993482** | 0.681770943 |

The primary coefficient remains20%, fixed before outcomes. The higher70/30
sensitivity is not used to claim an optimized coefficient. Prediction
correlations are0.956–0.962 and residual correlations0.965–0.970 under the
official target weights. These models are highly correlated; the measured
positive blend supports an incremental contribution, not independent regime signals.

## Training and architecture diagnosis

At15:43UTC all seven GALAM fits remain live with their original PID start ticks,
source, split and PANEL. Four LSTM/history arms completed4epochs; GRU72,
TBPTT256 and Muon completed5. Most last scores are modestly below their epoch3
best; TBPTT256 continues rising, reaching0.665725589 at epoch5. There is no
measured basis for forecasting a large remaining score gain merely from20epochs.
The original early-stop policy remains unchanged.

The [independent source/log audit](../galam-l4-quality8-20261028/launch/receiving/INDEPENDENT_TRACE_DIAGNOSIS_20261028.md)
finds finite, nonzero hidden/cell/attention/relative-bias gradients in sampled
windows. LSTM128 fusion saturation61% merits tracking, not a broken-model claim.
History256 actually uses broader lag context. Chunk256 changes gradient horizon
and update frequency together; it is not a pure context-only contrast.

Our clearest normalization hypothesis is per-timestamp BatchNorm: default
momentum.1 updates20,000times per sequence batch, so roughly95% of running-stat
weight lies in its last29timestamp calls. Training uses current cross-sequence
batch moments; inference uses tail-weighted running moments. This is an unknown
detail of the original reconstruction, not target/future leakage. Harm to WP
has not been demonstrated. [PyTorch documents these running-stat semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html).
The next cheap falsifier is an isolated frozen-weight normalization diagnostic
using only predeclared fitting members, never official valid.

GALAM is a plausible causal sequence regressor: recurrent state, causal local
convolution, adaptive attention and gated fusion. It now has measured value in
the fixed benchmark mixture. The original dimensions/equations remain partly
unknown; this is a schematic reconstruction, not an exact Denis Tomin replication.
Raw112/AdamW/LR1e-4/B384/C64 versus event211/Muon/LR1e-3/B32/C1000 cannot rank
architecture quality. LSTM128 has1,423,978parameters versus GRU975,618; increasing
capacity alone is not a demonstrated remedy.

## Actual inference trace and execution cost

One real LSTM128 CPU/CUDA trace atB384 over64steps after64warmup contains
**7683CUDA kernel launches**,1280addmm calls and64separate cuDNN LSTM calls.
Profiled scope296ms; kernel union92.8ms,31.4% of that instrumented interval.
Profiler overhead is included. This is neither whole-training utilization nor a
promised3x speedup; no backward/optimizer trace was captured. It supports testing
per-time dispatch/cache-copy overhead. See [PyTorch's launch-overhead guidance](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html).
Per-endpoint full-valid loading/forward/elapsed times are in the result receipt.

Preserve the frozen80/20 candidate and finish the original fits. Separate next
checks should target BN statistics and then matched causal211inputs. Profile and
verify any per-chunk vectorization before changing execution. A useful mixture
may later be distilled into a smaller GRU after measuring actual CPU serving
constraints; no current deployment package or public-score claim is made.

## Evidence and limits

- [Frozen plan](PLAN.json), [candidate recipe](ENSEMBLE_RECIPE.json).
- [Complete result](evidence/RESULT.json) and [independent rescore](evidence/INDEPENDENT_GALAM_BLEND_RESCORING_R1.json).
- [Owner](evidence/INFERENCE_OWNER.json), [terminal/artifact hashes](evidence/TERMINAL_AND_ARTIFACTS.json), [progress](evidence/INFERENCE_PROGRESS.jsonl).
- [Original fit continuity](evidence/ORIGINAL_FIT_CONTINUITY.json).
- [LSTM128 trace analysis](evidence/LSTM128_INFERENCE_TRACE_ANALYSIS.json), [LSTM128 operators](evidence/LSTM128_INFERENCE_OPERATORS.json), [GRU72 operators](evidence/GRU72_INFERENCE_OPERATORS.json).

All1873official-valid sequences/3,528,273scored rows are included and aligned.
Valid already selected the retained epochs: results are exposed, single-seed
development evidence. The reference is a clean one-model benchmark, not the full
deployed team ensemble. Predictions and raw traces remain on the receiving host;
only bounded receipts/hashes enter Git. The diagnostic worker exited and freed
GPU0; seven original fits continue unchanged.

[PR update body](PR_UPDATE.md) records the same result and links this evidence.
