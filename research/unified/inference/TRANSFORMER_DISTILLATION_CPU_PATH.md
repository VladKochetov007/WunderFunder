# CPU inference path for Transformer distillation

**Status: proposal grounded in source inspection.** No model was trained, imported, profiled, packaged, or run through the
evaluator for this note. The available CPU evidence below is historical and host-specific; the acceptance steps are proposed.

## Recommendation

Use the fixed L207M causal Transformer as an offline teacher and compare FP32 students: the canonical two-layer GRU and compact
local-causal Transformers. Inference for the experiment executes only the student. A GRU student is the lowest-risk release
candidate, but the proposed raw-112 student does not match the 211-input default native callback and needs a thin rowwise adapter.
Treat direct Transformer serving as a separate engineering track until its own bounded-state CPU callback, exporter, and complete
one-core package gate exist.

The fixed teacher is L207M (16 layers, width 1024, 16 heads, SwiGLU 2816), trained on complete normalized raw112 rows with
anchored-prefix causal attention (1024 new rows plus up to 3072 prefix rows). Teacher predictions are offline training targets;
the teacher is never in the submitted inference ensemble.

The compact student proposal is raw112 input, fitting-only per-column mean/std (std floor `1e-6`), linear `112→128`, two
pre-LayerNorm blocks, four 32-wide heads, RoPE, GELU FFN `128→512→128`, and a direct linear `128→2` head. Compare local windows
256 and 1024 (including the current row). Train and serve with the same per-layer local mask and cache semantics. L207M is a
separate, much larger offline teacher with anchored-prefix attention; its cache cannot be truncated and called equivalent to the
student's streaming state.

## Evaluator contract

The baseline task supplies 112 float32 features per row. There are 20,000 ordered rows per `seq_ix`; rows 0–98 are warm-up and
rows 99–19,999 require a length-two prediction. The callback must consume every row, including warm-up and unscored rows, return
`None` during warm-up, and reset sequence-local state when `seq_ix` changes. Sequence ids are shuffled and are not a time feature.
No prediction may use a future row, `is_scored`, or either target.

The official metric is Global Weighted Pearson, separately pooled over all selected rows for each output and then averaged. It
clips predictions and targets to `[-2, 2]`; it does not average per-sequence or per-block correlations. A training cache or local
acceptance score must preserve this pooling and row membership.

Primary contract sources are `docs/data_overview.md`, `docs/submission_guide.md`, and `utils.py` (`ScorerStepByStep`,
`GlobalAccumulator`). `ScorerStepByStep` validates full ordered sequence callbacks and requires `None` on warm-up after the model
has advanced its state.

## What the repository currently supports

| Area | Source evidence | Status |
|---|---|---|
| Current package callback | `submission/solution.py`, `native/src/python.rs`, `native/src/rnn_fast.rs` | `build_submission.sh` packages `wnn_native.Predictor`, a stateful one-row Rust callback with reset. It requires 211 inputs and defaults to FP32. |
| Raw-feature GRU stepper | `native/src/python.rs` (`GruStepper`), `native/src/rnn.rs` | Already accepts an arbitrary input width and exposes reset/step, but it has no `PredictionModel` wrapper or package-builder route for the raw112 student. |
| Streaming feature path | `native/src/stream.rs`, `native/src/python.rs` (`Predictor`) | `Predictor` builds 99 causal event features from 112 raw inputs and steps a 211-input GRU. This is incompatible with the proposed no-extra-feature raw112 students unless the wrapper selects a different stepper. |
| Recurrent export and package | `scripts/build_submission.sh`, `scripts/export_native_weights.py`, `submission/Dockerfile.build` | The package script writes an FP32 `.npz` from the recurrent checkpoint and copies `submission/solution.py`; `export_native_weights.py` is a separate recurrent binary exporter. Neither exports a Transformer. |
| Optional native precision | `submission/solution_native.py`, `native/src/python.rs`, `native/src/rnn_fast.rs` | The separate ensemble wrapper supports selected recurrent precision modes; absent a precision file it defaults to `i16`. `f16`/`i16` are reduced precision, not INT8. The baseline package script does not copy this wrapper. |
| Canonical trainer | `src/wnn/training/README.md`, `src/wnn/training/models.py`, `src/wnn/training/forward.py`, `src/wnn/nn.py` | Reusable model factory and runner cover recurrent models; `RNNModel.forward(x, h)` returns predictions and recurrent state. `make_model` has no causal-Transformer branch in the baseline. |
| Distillation targets | `src/wnn/training/cli_targets.py`, `src/wnn/training/losses.py`, `src/wnn/training/data_caches.py` | Existing `--distill`, `--distill-cache-root`, and `--distill-alpha` provide a student-side seam. |
| Teacher cache builder | `scripts/build_distill_cache.py` | Current builder is bound to four declared teacher families and a fixed 0.8125 coefficient sum. It is not a generic one-Transformer cache builder. |
| Teacher prediction export | `scripts/predict_run.py` | Exports ordered full-sequence predictions and provenance for supported training models; it is chunked batch prediction, not a scorer callback. |
| Transformer state | Legacy branch `codex/transformer-mamba-gpu-wave`, `src/wnn/causal_transformer.py` | Research-only chunked causal Transformer with explicit per-layer K/V state and absolute position offset. Not part of baseline `origin/research-team` at `87e89de`. |
| Submission budget gate | Legacy branch `codex/transformer-mamba-gpu-wave`, `scripts/check_submission_budget.py` | A source-reviewed full-package one-core Docker gate exists on that branch. It is absent from the baseline source inspected here and was not run for this note. |

The current native build recipe uses Python 3.11 slim Bookworm, Rust 1.94.0, `maturin>=1.9,<2`, and emits x86-64-v2/v3 wheels.
Repository `requirements.txt` pins NumPy 2.2.6 and ONNX Runtime 1.23.2 for local verification; it does not prove that ORT is
installed in the evaluator. These are the inspected build pins, not a verified hidden-scorer runtime manifest.

The legacy research Transformer is not this packet's L207M teacher. Its layer concatenates prior and current K/V tensors with
`torch.cat` on each chunk call, applies a local causal mask, and retains the last `window - 1` rows. The API is useful for teacher
training and chunk parity. A single-row callback would repeatedly allocate and copy nearly the entire cache for every layer and
row. A serving implementation needs preallocated circular buffers or an equivalent runtime-owned cache; changing the cache update
rule can change outputs and must be checked against the reference.

## Proposed compact model and arithmetic

The following counts assume raw input width `N=112`, width `D=128`, layers `L=2`, four heads, FFN width `4D=512`, and the direct
two-output head described above. The head count changes tensor layout but not these dense parameter totals.

For a pre-norm layer with Q/K/V projections, attention output, two layer norms, and a two-layer FFN, trainable parameters are:

```text
per layer = 12D² + 9D linear biases + 4D LayerNorm parameters
          = 198,272 parameters at D=128
input projection = 112D + D                         = 14,464
two Transformer layers = 2 × 198,272                = 396,544
direct prediction head = 2D + 2                     = 258
total trainable parameters                         = 411,266
mean/std buffers                                    = 2 × 112 = 224 floats
FP32 tensor payload                                 = 411,490 × 4 = 1,645,960 bytes
```

Any training-only hint projection is stripped when producing the two-output student endpoint; it is not part of the served head or
this payload count. These are parameter and payload counts, not a ZIP-size measurement. Conservatively treat the ambiguous “20 MB”
rule as a 20,000,000-byte package ceiling until the scorer’s exact unit is verified. Reserving 4,000,000 bytes for code, native
runtime, metadata, and archive overhead leaves 16,000,000 bytes, or 4,000,000 FP32 parameters, as a proposed combined-model
payload ceiling. This is an engineering budget, not an official model-size or parameter-count rule. Count all ensemble members
together and still measure the final ZIP bytes.

For a batch-one FP32 cache with `W-1` previous rows retained, the K/V memory is:

```text
cache_bytes = 2 (key,value) × L × D × (W - 1) × 4 bytes
L=2, D=128, W=256:   522,240 bytes per model
L=2, D=128, W=1024: 2,095,104 bytes per model
```

This excludes weights, runtime, Python arrays, activations, scratch buffers, normalizer/input state, and temporary export tensors.
For illustration, a four-member fold ensemble uses 2,088,960 bytes at W=256 or 8,380,416 bytes at W=1024 for K/V alone. Every
member needs an independent reset and absolute position; four members are not a claim that the selected packet deploys an
ensemble. A smaller student does not automatically imply smaller inference memory if caches or runtime copies are duplicated.

Approximate dense work per row, counting multiply-accumulates (MACs) and omitting biases, normalization, activation, RoPE,
softmax, feature generation, and state movement, is:

```text
static linears = 112D + 12LD² + 2D = 407,808 MAC/row
attention      = 2LD × min(W, rows_seen) MAC/row
W=256 (steady) = 538,880 MAC/row
W=1024 (steady)= 932,096 MAC/row
```

The packet's H=256, two-layer GRU student with raw112 input and a 256-wide encoder has about 946,688 dense MAC/row by the same
convention (encoder `112×256`, two 256-wide GRU layers, and readout `512→256→2`). Its recurrent state is 512 FP32 values, or 2,048
bytes, before scratch. The compact Transformer K/V cache is substantially larger but still small in absolute terms for one member.
These counts do not predict which backend is faster.

The validation-shaped panel has 1,873 × 20,000 = 37,460,000 callback rows, of which 37,274,573 require predictions after the
99-row warm-up per sequence. A 3,600-second limit averages 96.10 microseconds for every row and all work. The compact Transformer
requires approximately 20.186 trillion model MACs at W=256 or 34.916 trillion at W=1024 across all rows. If the whole time budget
were available to the model alone, that implies 5.61 or 9.70 GMAC/s respectively; the actual pipeline must also process features,
callbacks, scoring, and I/O. These are necessary arithmetic rates, not measured throughput claims.

The existing `torch.cat` cache update also copies on the order of a full K/V window per row: approximately 0.52 MB at W=256 or
2.10 MB at W=1024 per model step for this shape. Across 37.46 million rows that is roughly 19.6 TB or 78.5 TB of cache traffic. A
ring buffer removes that repeated full-window copy; it does not remove the per-row attention arithmetic.

## Distillation workflow

1. Freeze the baseline, raw112 input configuration, split/component manifests, seeds, checkpoint epochs, and teacher/student
   recipes. The selected packet uses fixed L207M outputs. Its inputs are complete fit-normalized raw112 rows; its anchored-prefix
   causal attention is distinct from the streaming student's local sliding-window attention.
2. Preserve the packet's teacher exposure: L207M targets are intentionally in-sample on outer-fitting components, while every teacher and all its ancestors exclude the outer evaluation components from fitting. Call these fitting-only in-sample targets, not OOF
   targets. Any OOF variant is a separate recipe.
3. Export ordered full-sequence predictions with checkpoint hash, split/component hash, fit membership, source hash, row-group
   order, prediction hash, dtype, and shape. Bind `seq_ix`/row-group identity and row order; reject missing, duplicated,
   non-finite, or reordered rows. The existing four-family cache builder cannot be repurposed without relaxing its source-family
   assumptions.
4. Add the missing Transformer model/state/checkpoint/cache registry integration and one-teacher cache binding to the canonical
   trainer. Reuse its runner, target/mask assembly, loss, and provenance components rather than creating a parallel trainer. The
   packet's student recipe fixes strict FP32 and KD alpha 0.9; keep that coefficient unchanged across cells and preserve global-WP
   validation/clipping. Existing `make_model` and four-family cache code do not implement this Transformer/cache route. The packet
   carries and detaches K/V at each 500-row training core; a later optimizer update can make cached history stale. This is a
   declared training approximation. Frozen-weight inference replays every row from sequence start with fixed serving semantics.
5. Run the packet's six matched arms: GRU and W256/W1024 Transformer, each with labels or the same L207M KD target. It specifies
   30 paired fold/seed cells per arm and alpha 0.9 for KD. Preserve the same cell, student seed, data membership, optimizer/update
   budget, checkpoint selection, and evaluation rows across contrasts. Report each cell, pooled score, prespecified
   architecture/KD contrasts, and uncertainty. Teacher gain alone is not evidence of transfer.
6. Freeze each scientifically accepted FP32 checkpoint as the release reference. The existing `build_submission.sh` currently
   packages the 211-input `solution.py`; a raw112 GRU needs a wrapper around `GruStepper` (or a verified equivalent) and a
   package-builder selection that loads its normalizer/state. The separate `export_native_weights.py` writes recurrent binary
   tensors and is not called by `build_submission.sh`. Verify parity over complete sequences, warm-up, chunk boundaries, and A-B-A
   replay.
7. FP32 must pass numeric, reset/state, and scientific-quality gates and remain the reference; its runtime or size may still
   exceed the release budget. After the FP32 endpoint is frozen, create separate reduced-precision candidates with a mode actually
   supported by the chosen native adapter. Keep `f16`/`i16` distinct from `q8*` integer-weight modes in reports. Compare each
   candidate against that FP32 checkpoint on identical rows and scores. The selected final package, quantized or FP32, must
   independently pass the package size, runtime, memory, and evaluator contract gates. Do not assume lower precision is faster.

## Direct Transformer serving design

The proposed Transformer receives raw112 directly. Its step must apply the frozen fit-only mean/std normalization, input
projection, layer 0 attention and FFN, layer 1 attention and FFN, direct readout, and state update in training order. It must not
append the native 99 event features or use that model's online z-score path. Compute enough of each warm-up row to update K/V
state, then return `None` until `need_prediction` is true.

Keep a fixed-size K/V ring per layer, batch one, with `W-1` previous entries. Track an absolute position counter per sequence. The
current query sees itself and at most the preceding `W-1` positions. Rotate each query/key at the same absolute position and at
the same stage as the training implementation; do not rotate already-rotated cached keys again. Reset ring indices, valid lengths,
position, and every member state on a new `seq_ix`. Position should advance over warm-up rows and must not derive from shuffled
sequence ids.

`W` is the attention width of each layer; stacking layers can propagate information across more than `W` original rows. The packet
keeps autoregression off: do not feed predictions back as features or state. If a later recipe adds prediction feedback, it needs
its own causal training and replay contract; labels are never callback inputs. If multiple fold students are blended, freeze
members and coefficients before evaluation, give every member an independent reset and state, and use the same fixed blend on
every required row. Do not select or drop members by `is_scored`, outer scores, or a wall-clock governor.

The GRU and Transformer have different integration gaps. `GruStepper` accepts raw112 model inputs, applies the checkpoint's
mean/std internally, and resets its hidden state; the current scorer `Predictor` instead validates 211 inputs and calls the
event-feature stream. A raw112 `PredictionModel` wrapper must load recurrent checkpoint tensors, call `GruStepper.step`, and
preserve warm-up/reset behavior in a Python 3.11 package. For the Transformer, both the model implementation and wrapper are
missing.

Two direct Transformer implementation routes are plausible, neither currently complete:

- **Rust/PyO3 native stepper:** add a bounded Transformer model/state type, ring-buffer attention kernels, exact FP32 reference
  math, weight export and load checks, CPU feature dispatch, and a Python `PredictionModel` adapter. Apply the student's raw112
  normalizer directly; do not reuse `StreamFeatures` or the 211-input native GRU stream. This is the closest fit to the existing
  native package but requires a new Transformer implementation and parity proof.
- **ONNX Runtime CPU:** export a one-row graph with explicit per-layer K/V state inputs/outputs (or a verified shared-buffer
  interface), position, and model inputs. Set intra-op threads to one and sequential execution; verify the execution provider and
  runtime are actually available offline in the scorer. Explicitly test graph export, cache ownership/copies, reset, and
  chunk-to-row parity. Repository `requirements.txt` pins ONNX Runtime 1.23.2 for its local verifier environment; that does not
  establish the hidden scorer's installed version or guarantee that ORT belongs in a 20 MB submission.

Python/PyTorch per-row inference can serve as a functional prototype, but it has no package-time or latency evidence here. The
public PyTorch `MultiheadAttention` API exposes query/key/value and masks, not a persistent streaming cache contract; the cache
must be carried and updated explicitly by the caller/model.

## Fair acceptance gates

**Research acceptance.** Pre-register teacher/student recipes and the minimum meaningful score change before looking at final
paired cells. Keep train/validation membership disjoint and immutable; use no test labels; use identical paired student
seeds/folds and update budgets; report full-panel pooled WP for both outputs. Separate in-sample teacher-target results from OOF
results. Report quality and fitting cost together so a teacher's extra compute is visible.

**Numerical and contract acceptance.** Compare FP32 teacher/student reference paths with exported inference on complete ordered
sequences. Check finite `float32[2]` predictions, `None` for all warm-up callbacks, state advancement on all 20,000 rows, reset on
each sequence transition, deterministic repeated outputs, first/last row, chunk boundaries, and A-B-A replay. Set explicit
absolute/relative tolerances before export. Proposed engineering tolerances are `atol=2e-5`, `rtol=1e-4` for per-output parity and
`|Δ global WP|≤1e-5` over the full panel; these are prospective acceptance thresholds, not measured parity or an official
evaluator rule. Score the exact submission callback with the official global metric implementation.

**Package and resource acceptance.** Measure the final ZIP, start the exact Python 3.11 package offline, and run every row of the
supplied full validation panel under one CPU and 16 GB with the same callback and evaluator used for acceptance. Proposed runtime
headroom is 15%, so require at most 3,060 seconds for the 37,460,000-row validation-shaped panel (about 81.7 microseconds per
callback row, all work); this is an engineering policy, not a hidden-test guarantee. For an evaluation of `N` callback rows, the
gross all-work limit is `3,600/N` seconds per row. Hidden `N` is unknown; a validation pass cannot establish the hidden-set
runtime. Bind the receipt to hashes of ZIP, input panel, verifier, and expected row count. A complete pass on supplied validation
is evidence only for that panel; it does not prove hidden-set runtime. Require a predeclared safety margin inside the declared
limit and memory headroom; record elapsed full-evaluation time, peak memory, rows, outputs, and exit status. Do not infer
full-panel readiness from a short prefix, package build time, or component microbenchmark.

Historical evidence is in `research/profiling-2026-09-15.md`: on an i9-11900H AVX-512 host under a constrained profile, the
reported native GRU model path was 112 microseconds/row and feature generation was reported separately at 18 microseconds/row.
That is older host-specific measurement, not a current scorer result and used the 211-input feature path rather than the proposed
raw112 model; the combined figures do not leave margin against a 96.10-microsecond all-work average. The same profile warns its
ONNX Runtime timings were distorted by cgroup CPU quotas. The legacy `scripts/check_submission_budget.py` gate declares 20,000,000
bytes, one CPU, 16,000,000,000 bytes memory, no network, and 3,600 seconds, but this note did not execute it and its source is on
the legacy branch only.

## Known implementation gaps

- No canonical Transformer option in baseline `training.models.make_model` and no baseline trainer-to-native- Transformer
  checkpoint format.
- No current scorer wrapper/package route for raw112 GRU weights: `GruStepper` exists, but packaged `submission/solution.py`
  selects the 211-input `Predictor`.
- No generic, single-source Transformer distillation cache builder with the family-specific assumptions removed and all
  source/membership hashes bound.
- No direct Transformer inference adapter for the scorer's one-row callback.
- No preallocated K/V ring, Rust/ORT exporter, explicit state serialization, Python 3.11 package, or CPU feature-dispatch path for
  a Transformer.
- No Transformer source-vs-streaming parity receipt, full-panel one-core result, measured memory, final package size, or
  hidden-evaluator evidence.
- The legacy full-package budget gate is not in the baseline branch; bring its exact-input hash and complete-row-count discipline
  into the baseline workflow before relying on it for candidate acceptance.

## Primary inference documentation

- [PyTorch `torch.nn.GRU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html): recurrent hidden-state input/output
  shapes for the current student route.
- [PyTorch `torch.nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html):
  query/key/value and mask interface; persistent cache updates remain model/caller state.
- [ONNX Runtime thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html): intra-op and
  execution-mode controls; defaults may use multiple physical cores, so one-core settings must be explicit.
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html): quantization
  formats, Transformer-specific optimization, and hardware-dependent benefit.
- [ONNX Runtime quantization method
  selection](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#method-selection): dynamic methods are
  commonly recommended for RNN/Transformer models, with online scaling overhead and hardware-dependent performance.
