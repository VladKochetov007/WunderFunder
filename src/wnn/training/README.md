# Modular recurrent training

Use `scripts/train_rnn.py` for the stable command line. New Python callers should import focused components directly:

```python
from wnn.training.environment import InputConfiguration, TrainingEnvironment
from wnn.training.inputs import build_inputs
from wnn.training.models import make_model
from wnn.training.runner import run_training

# Existing supplied data/cache/split validation still applies to any real run.
# run_training(["--name", "my_run", "--precision", "fp32", ...], environment=env)
```

`TrainingEnvironment` carries paths, causal input/readout state and explicit dependency hooks. `TrainingRun` has separate setup/data/network/optimizer/progress/batch records; phase functions retain local tensor aliases. The core never transports a `locals()` dictionary or executes generated source. The legacy facade alone reads historical imported settings and resolves a fixed export registry, including custom-name importlib consumers that never register in sys.modules.

## Components

| Module | Responsibility |
|---|---|
| [__init__](__init__.py) | Package marker; no training side effects. |
| [averaging](averaging.py) | Foreach EMA without new host synchronization. |
| [batches](batches.py) | One-time full-sequence input, mask, label and update-window preparation. |
| [cache_identity](cache_identity.py) | Source-qualified teacher and propensity-cache provenance. |
| [calibration](calibration.py) | Existing declared reference-mass calibration path. |
| [checkpoints](checkpoints.py) | Serialized arguments, inference representations, selection and strict continuation contract. |
| [cli](cli.py) | Assemble the unchanged CLI from reusable registration groups. |
| [cli_data](cli_data.py) | Ordered CLI registration for data options, preserving original declarations. |
| [cli_features](cli_features.py) | Ordered CLI registration for features options, preserving original declarations. |
| [cli_model](cli_model.py) | Ordered CLI registration for model options, preserving original declarations. |
| [cli_normalization](cli_normalization.py) | Ordered CLI registration for normalization options, preserving original declarations. |
| [cli_optimizer](cli_optimizer.py) | Ordered CLI registration for optimizer options, preserving original declarations. |
| [cli_performance](cli_performance.py) | Ordered CLI registration for performance options, preserving original declarations. |
| [cli_targets](cli_targets.py) | Main row weighting, shifted targets and next-change auxiliary labels. |
| [cli_views](cli_views.py) | Ordered CLI registration for views options, preserving original declarations. |
| [compatibility](compatibility.py) | Bounded legacy export adapters for bare/package/custom-name imports; no module monkey-patching. |
| [compatibility_classes](compatibility_classes.py) | Environment injection for legacy class exports, class-valued API and worker serialization. |
| [data_caches](data_caches.py) | Validated teacher, quote, spread and propensity cache bindings. |
| [data_loaders](data_loaders.py) | Loader memory admission, sampling order, pinning and drop-tail behavior. |
| [data_membership](data_membership.py) | Deterministic fitting/evaluation membership and sampling selection. |
| [datasets](datasets.py) | Lazy complete-sequence reading and memory-mapped teacher/auxiliary targets. |
| [environment](environment.py) | Explicit per-run input/readout configuration, source roots and injected I/O/model/backend hooks. |
| [epoch](epoch.py) | Epoch sampling identity, batch completion and gradient diagnostics. |
| [evaluation](evaluation.py) | Reusable prediction and official validation diagnostics. |
| [fit_state](fit_state.py) | Optimizer, persistent fit clocks and sequence/window state records. |
| [forward](forward.py) | Architecture-specific chunk dispatch and recurrent-state detachment. |
| [forward_backward](forward_backward.py) | Original student/EMA forward and scaled backward execution. |
| [inheritance](inheritance.py) | Parent preprocessing, feature expansion and clean-fold checkpoint ancestry. |
| [initializer](initializer.py) | Frozen teacher cloning, fingerprinting and causal target replacement. |
| [inputs](inputs.py) | Causal input assembly and fitting-only research-feature priors. |
| [losses](losses.py) | Main and auxiliary objectives with unchanged reduction/summation order. |
| [macro_updates](macro_updates.py) | Controlled macro losses and carried recurrent/initializer state. |
| [membership](membership.py) | Bounded component-subset manifests and membership validation. |
| [model_validation](model_validation.py) | Architecture/width restrictions and block-memory defaults. |
| [models](models.py) | Reusable architecture factory with the original constructor order. |
| [normalization](normalization.py) | Fitting-only normalization and numeric-map calibration. |
| [optimization](optimization.py) | Optimizer/scheduler/objective/averaging construction and preflight exits. |
| [outputs](outputs.py) | Evaluation, fixed/best checkpoint selection and terminal/resume recovery artifacts. |
| [prefetch](prefetch.py) | Original pinned-memory CUDA stream prefetch. |
| [profile_results](profile_results.py) | Profile receipts and parity artifact output without fitting/quality claims. |
| [profiling](profiling.py) | CUDA event timing, warning capture and finite-update evidence. |
| [provenance](provenance.py) | Run/source/cache/fitting provenance assembled before fitting. |
| [reporting](reporting.py) | Structured logging preserving machine-consumed legacy record content. |
| [row_views](row_views.py) | Existing causal row-thinning views and retained-mass handling. |
| [runner](runner.py) | Ordered orchestration; the only owner of epoch/batch/window traversal. |
| [setup_lineage](setup_lineage.py) | Source ancestry and split loading, device/runtime prerequisites. |
| [setup_model](setup_model.py) | Output layout, model construction and parent-state loading. |
| [setup_model_runtime](setup_model_runtime.py) | QAT/factorization/activation attachment and model identity. |
| [setup_preprocessing](setup_preprocessing.py) | Inherited/fitting-only preprocessing and frozen initializer-teacher construction. |
| [setup_progress](setup_progress.py) | Clocks, profiling setup and strict resume; RNG restoration remains last. |
| [setup_state](setup_state.py) | Options, source-qualified data and network state records. |
| [source_identity](source_identity.py) | Public entrypoint plus implementation hashes; missing portable dependency metadata is explicit. |
| [state](state.py) | Typed ownership of setup, data, network, optimizer, progress and batch state. |
| [tiedlead](tiedlead.py) | Fixed TL3 profile and labels-only teacher configuration checks. |
| [targets](targets.py) | Main row weighting, shifted targets and next-change auxiliary labels. |
| [updates](updates.py) | Optimizer/QAT/scaler/scheduler/SWA/EMA ordering and original synchronization policy. |
| [validation_inputs](validation_inputs.py) | Input/subset/target compatibility checks. |
| [validation_training](validation_training.py) | Model, checkpoint, precision and update/detach clock checks. |

## Preserved contracts

- Training arithmetic keeps the original tensor copies, gradient boundaries, loss reductions, dtype policy, sampler behavior and update order. Fit-only epoch telemetry accumulates detached scalar objectives and pre-update learning rates, then reads one scalar at epoch reporting and adds their mean with update counts to the existing `FIT`/`VALID` record. It does not affect evaluation or checkpoint selection. The original fast input allocation, foreach EMA and CUDA prefetch remain.
- Per-window order is unscale → update counters/preclip diagnostics → perturb/clipping → optimizer → QAT projection/invalidation → scaler → scheduler → attempted clock → SWA/EMA. Skipped AMP attempts retain the original scheduler/averaging clock semantics.
- Sequence state resets, prefix replay/discard, initializer-teacher state, teacher-target clone isolation, source-qualified folds and fixed checkpoint policy remain explicit.
- CLI defaults are preserved, including the legacy FP16 default. Strict-FP32 research recipes must still pass `--precision fp32`; this refactor does not rewrite numerical recipes.
- Strict resume hashes the public entrypoint and implementation sources. An old checkpoint cannot claim exact continuation after a code change. Existing parent initialization remains supported with its ancestry checks.
- Portable source packets can omit pyproject/uv.lock; provenance names these missing metadata files instead of failing or claiming a locked environment. Present sources are hashed and actual attrs version is recorded.

## TL3 tiedlead encoder

The canonical teacher adapter is the ordinary GRU architecture with `--enc tiedlead` and `--research-features tl3_lead`. It requires the unmodified 112 raw plus 99 event channels followed by 52 causal lead channels (263 inputs), a 256-wide two-layer GRU, one GELU readout layer, and `--precision fp32`. The plain 211-input student (`--enc linear --research-features none`) is unchanged. This is an architecture-matched direct-teacher ablation; it does not claim the historical TL3 teacher's distillation ancestry.

`tl3_lead` gathers the 52 raw i1 channels at the most recent i0 price event, treating row zero as a sequence-start event. Input assembly computes the gather over the full sequence before TBPTT chunks are passed to the model. The encoder applies one shared `Linear(104,128)-GELU-Linear(128,64)-GELU` map to both raw-scale instrument books and their row-to-row deltas. That same map receives the current i1 book and its difference from the lead snapshot. Zero-initialized `tied_out` and `lead_out` projections add those branches to the normalized 263-to-256 input projection before GELU, then the GRU and `512-to-256-to-2` readout run as usual.

Chunk state is `(GRU hidden, previous raw book[104])`; [`src/wnn/tiedlead.py`](../tiedlead.py) carries the preceding row into the next chunk so the first delta of a chunk is not discarded. Passing `None` resets both states at a sequence boundary. This is an intentional causal boundary repair: the archived implementation zeroed each encode call's first delta, so this adapter does not claim exact historical chunk-boundary parity. The tiedlead validator rejects parent initialization, incompatible preprocessing/encoders, extra inputs or output heads, and non-FP32 precision. `predict_run.py` rebuilds the model from checkpoint arguments and sets the saved research-feature mode before calling the same full-sequence input assembler. Trainer modules are covered by the normal training source identity; tiedlead provenance also hashes `src/wnn/nn.py`, `src/wnn/research_features.py`, and `src/wnn/tiedlead.py`.

## Quality checks

Static checks and checker unit tests do not require the training dependencies:

```bash
make quality quality-tests
```

The lightweight GitHub workflow checks PRs to main/research-team without installing Torch or loading competition data. New trainer modules are capped at300lines and functions at200lines; signatures/docs and absence of global/nonlocal are checked. Keep new orchestration out of the legacy facade. Add a component only for a clear responsibility, and reuse models/inputs/targets/checkpoints rather than forking trainers.

Independent Sol6.1/medium static review and the [original refactor receipt](../../../research/unified/methods/MODULAR_RNN_TRAINER_REFACTOR_20261017_R1.md) document that change's scope: Ruff/structure checks and normal CLI help, without unit/integration, numerical parity or throughput tests.

The subsequent [maintenance receipt](../../../research/unified/operations/MAIN_MAINTENANCE_20261017.md) adds synthetic CPU checks for feature values/gradients, causal behavior, resume/export and legacy class/pickle/spawn compatibility. Run `make training-tests` in a provisioned CPU environment; see [Development](../../../docs/DEVELOPMENT.md). CUDA parity, full-data scoring and end-to-end speedup remain unmeasured.
