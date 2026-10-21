# Independent implementation review: target-only adapter objective

## Scope and disposition

Read-only review of `adapter_objective.patch` against the canonical training components and the online-feature data contract. No actor fidelity conclusion, model/data payload, Torch runtime, native extension, fit, or quality artifact was read or executed before these findings were recorded. This review does not assess whether the new target-only mechanism improves score.

**Initial V1 disposition: hold pending an objective-gate correction.** V1's loss-function early return was correctly scoped to adapter-only/target-only and left the default `inherited` path unchanged, but it did not cover all teacher-derived main-loss paths. V2 adds those missing guards and passed the independent source-pinned mocked preflight checks below. This remains an implementation review, not fit admission or a quality verdict.

## Findings

### F1 — Teacher targets can still enter `main_loss` (blocking)

The proposed guard in `feature_adapter.py` rejects `args.distill`, but it does not reject `args.distill_cache_root` or `args.final_distill_initializer`. Both can populate `run.batch.teach` while `args.distill == ""`:

- `batches.py:75-89` loads columns from a validated distillation cache into `teach`; `main_loss` in `losses.py:48-86` then replaces the direct label target with a label/teacher blend. The early return proposed in `auxiliary_loss` cannot change this because `runner.py:78-80` calls `main_loss` first.
- `setup_preprocessing.py:126-150` builds a fixed initializer teacher for `--final-distill-initializer`; the same `main_loss` teacher branch uses it to form the blended target. Its existing validation requires a cache and suitable final-adaptation configuration, but does not conflict with the proposed target-only check.

Concrete accepted counterexample: `--research-feature-adapter-only --research-feature-adapter-objective target-only --loss wmse --distill-cache-root <validated-cache>` with `--distill ""`, `--td-alpha 0`, `--ema-teacher 0`, and `--ce-mse 0`. The proposed predicate passes, but batches supply `teach` and `main_loss` optimizes a teacher blend. The `final_distill_initializer` route is a second missed teacher source. This makes the recorded `auxiliary_terms: skipped` provenance misleading and violates the requested plain supervised target objective.

Correction: validate `args.distill`, `args.distill_cache_root`, and `args.final_distill_initializer` as absent in target-only mode before cache/data/model setup. Retain a defensive check in adapter configuration. Ensure `distill_family_weight == 0` too: that cache-backed family loss is added inside the same `main_loss` teacher branch (`losses.py:87-95`), and `setup_model.py:104-108` changes the output width/head when enabled.

### F2 — The objective gate is later than cheap CLI validation

The proposed mode checks live in `configure_feature_adapter`. `runner.py:38-54` calls `validate_inputs`, lineage/cache/loaders, model build/initializer load, and preprocessing before `configure_feature_adapter` is reached from `setup_model_runtime.py:146`. Thus unsupported target-only combinations are rejected before optimizer construction and any training update, but only after the potentially expensive cache and model setup. Move scalar mode compatibility checks into early CLI validation, keeping the adapter-time shape/zero-suffix/freeze guards as defense in depth.

### F3 — Running WMSE normalization remains accepted

The check requires `loss == "wmse"`, but does not require the standard `loss_norm == "chunk"`. With `--loss-norm running`, `main_loss` takes its separate running-normalizer branch (`losses.py:107-114`) rather than the configured chunk WMSE objective. This is still target-only in the sense that it uses no extra labels, but it changes the primary objective/update scale. If this arm is meant to isolate removal of auxiliary terms while holding the main WMSE recipe fixed, reject non-chunk normalization; otherwise name and predeclare it as a distinct main-loss variant.

## Checks that pass by source inspection

- The new CLI default is `inherited`; the loss early return requires both `research_feature_adapter_only` and `target-only`, so the historical/default loss path is untouched.
- The proposed early return precedes every addition in `auxiliary_loss`, covering `gate_ce`, structure, next-event, tier, mask, sign, quote, per-shift and EMA consistency additions in that function.
- Existing adapter guards remain in place for zero weight decay, no generic freeze/QAT/numeric mapping, FP32 AdamW, no optimizer resume, a zero-initialized 211+60 input suffix, and trainability only of `inp.weight[:, 211:271]`.
- Parent output-head configuration is inherited by `inherit_init_arguments`; target-only objective metadata is not an architecture input. The separate `distill_family_weight` option is the exception noted in F1 because it can add an eight-unit head before initializer shape validation.

## V2 independent recheck

The controller's V2 integration receipt pins the current `cli_features.py`, `feature_adapter.py` and `losses.py` sources to `integration_v2.json`. I extracted only the preflight part of `configure_feature_adapter` from the pinned source AST and ran it against mocked argument namespaces. `loss != wmse`, nonempty `distill`, nonempty `distill_cache_root`, `final_distill_initializer`, nonzero `distill_family_weight`, `td_alpha`, `ema_teacher`, and nonzero `ce_mse` each raised `ValueError`. A valid adapter-only target-only namespace passed that preflight; target-only without adapter-only was rejected; inherited/default arguments remained accepted. `ce_mse` is explicitly rejected by the guard, although the CE+MSE branch is inactive for WMSE; this is a conservative configuration restriction, not evidence that `ce_mse` itself adds a loss in this recipe. The check did not import Torch or construct a model.

The same checker compiled the actual early-return predicate from `auxiliary_loss`: it evaluates false for default non-adapter mode and for adapter `inherited`, and true for adapter `target-only`. The return precedes reads of auxiliary batch state. V2's patch only adds this gate to `auxiliary_loss`; it does not alter `main_loss`, output-head construction, or adapter parameter freezing. Static inspection found no predictor consumer of `research_feature_adapter_objective`; `predict_run.py:218-234` reads feature mode/control/prior for inference, while the objective remains training/provenance metadata.

### Remaining conditions

- `loss_norm == "running"` still passes the target-only preflight. `losses.py:107-114` selects a different WMSE normalizer in that case. The frozen recipe omits `--loss-norm`; `cli_normalization.py:9-12` defaults it to `chunk`, so this is not a blocker for that candidate. If another target-only run uses running normalization, name it as a distinct main-loss variant or add an explicit guard when plain chunk WMSE is required.
- Invalid objective combinations are rejected by `configure_feature_adapter` at `setup_model_runtime.py:146`, after the runner has validated and loaded membership, caches, loaders, model and preprocessing (`runner.py:38-54`), but before optimizer construction (`runner.py:55`) or a fit update. This is a setup-cost issue, not a path for teacher losses to reach a valid target-only fit; move scalar checks into early CLI validation if fail-fast behavior is required.
- The V2 integration receipt still states `fit_admitted: false`. These mocked checks do not prove fitted weights, gradient masking, a full update, serving parity, or benefit.

## Limits

The V1 and V2 reviews are static source and standard-library AST/mock checks only. No real parser-to-runner call, optimizer update, model, dataset, GPU, or prediction cache was used. Findings concern objective/CLI semantics, not fit validity or expected score.
