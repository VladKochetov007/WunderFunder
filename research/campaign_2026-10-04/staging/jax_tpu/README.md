# Staged JAX TPU recurrent core

This tree is isolated from root trainer/model sources. It implements the
PyTorch `RNNModel` linear-encoder / stacked-GRU / residual-readout computation
using JAX primitives and `lax.scan`, with PyTorch-named tensor layout for NPZ
import/export. `src/wnn_jax/bridge.py` adds a PyArrow/NumPy fold-0 data bridge,
fixed SBD targets, and the original 23-output combo loss; it does not launch a
TPU job.
The weighted loss accepts fixed two-channel targets and supervises only the
first two model outputs, preserving 23-output checkpoints with auxiliary heads.

The initial core is for `cell='gru'`, `enc='linear'`, and dropout 0. It keeps
Torch checkpoint names and shapes, including both GRU biases and PyTorch's
reset/update/new gate order. Its AdamW is a transparent custom optimizer and
is not claimed to reproduce Muon or the root trainer's full optimizer schedule.
`train_step` is JIT-compiled with `chunk_rows` and `precision` static so TPU
compilation can specialize the recurrent scan and TBPTT layout.

The real-data bridge reads the fold-0 JSON groups and raw TRAIN/VALID parquet
row groups, builds the 211 columns `raw112 || event_features(raw112)`, verifies
prediction sidecars before addressing caches by row-group id, and uses the
actual combo teachers `c20260929_since_s3_f0` and `c20260929_since_s4_f0` plus
`c20260929_prop_f0`. Teacher prediction row order must exactly match the
trainer target-cache row-group list. Propensity arrays use absolute Parquet
group IDs and preserve the corrupt TRAIN hole at 528; the preflight chooses a
fit group on each side of that hole. It implements the stopped bootstrap at horizon 8 and
alpha 0.5 while retaining the original label-based |y| weights,
propensity/since row weights, incoming EMA consistency, structural
spread/branch/class losses, and next-change distance/sign losses. The combo
chunk update is in the bridge so the frozen GRU equations in `rnn.py` remain
unchanged. Student and EMA recurrent carries are separate and both detach at
chunk boundaries.

`experiment_pair.json` describes a staged FP32/BF16 precision pair with shared
data order, dropout 0, initial NPZ, AdamW schedule, and evaluation mask. AdamW
parameters and moments stay FP32 in both arms; BF16 applies only to recurrent
compute. Target validity weights are floating point values, while the official
evaluation mask is boolean `need_prediction AND is_scored AND row >= 99`. This
explicit official mask can differ from the current trainer's checkpoint-selection
mask, which uses `is_scored` and warm-up only; root should decide which metric
drives paired checkpoint selection before sealing the run.

The host-only exporter is `scripts/export_combo_checkpoint_npz.py`. It imports
Torch and the root `RNNModel`, but no JAX, validates strict tensor names/shapes,
FP32 finiteness and positive normalization scales, refuses to overwrite files,
and rehashes the source checkpoint after load. Root can run this small
conversion in its bounded CPU scope while the GPU is occupied. Then the bounded
real-data harness is `scripts/realdata_preflight.py`. Export
`artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt` to the exact
Torch-named NPZ at `artifacts/runs/c20261003_combo_s3_f0/model_ep3.jax.npz` and
confirm its tensor hash/layout. The harness runs one 1000-row update on two
TRAIN and two fit-VALID sequences under each precision, exports and reloads
both states, then scores the first chunk of four held-out VALID sequences with
the same official mask. Invoke it with an explicit `--backend cpu` or
`--backend tpu`; it fails if JAX selects a different backend. It records hashes
for the raw data, teacher arrays, propensity arrays, initial checkpoint, and
NPZ artifacts. It writes `metrics.json` and two NPZ checkpoints. This
is a systems preflight only; its four-sequence score is not predictive evidence.

The staged tests compare a deterministic two-layer Torch model against JAX,
check whole-sequence and carried-chunk parity, verify stop-gradient boundaries,
exercise masks/weighted fixed-target loss, and check FP32/bfloat16 execution.
Bridge formula tests are drafted but have not been run. No imports, compilation,
device probing, or jobs have been run. JAX/TPU support and numerical tolerances
remain unverified locally. The required combo TRAIN and VALID prediction and
propensity caches are present and sidecar-addressable; the exact starting
checkpoint is `c20261003_combo_s3_f0/model_ep3.pt` and its JAX NPZ export is
still required. Run the bounded four-stream preflight before throughput
estimates or any full TPU fit.

For chunk-by-chunk optimization, `train_chunk_step` accepts explicit recurrent
state, applies one weighted update, and returns stop-gradient carry. `train_step`
is a convenience update over a full sequence split into detached TBPTT chunks;
it applies one update to the aggregate full-sequence loss.
