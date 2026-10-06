# Strong recipe transfer packet

Implemented and CPU-tested; independent scientific review and controller
admission still required. No scientific fits or provider jobs were launched.

`materialize_specs.py` only writes unsealed JSON drafts. It reuses
`scripts/continuation_recipes.py` parent/head/data-dependency functions and
records the exact seed4 public parent/head provenance in
`recipe_provenance.json`. It has no execution/submission function.

The two separate experiments share one explicitly declared fresh control per
seed/fold: control, fixed_mix, reanchor010, bootstrap_off. Initial screen is
seed3/fold0. Reserved confirmations are seed4/fold0 and seed3/fold1. There are
12 parent fits and 12 head fits if all blocks are admitted, 24 corresponding
FP32 prediction exports, and four initial full-shape probe drafts. Confirmations
are not admitted merely because their specifications exist. Readout and
supervision treatment are never combined in this block.

Every candidate is evaluated as fixed 50/50 parent ep3 plus head ep0 predictions,
with parent scores separately. Head seed remains3. This is a prediction mixture,
not averaging nonlinear head parameter tensors. `screen_*_{parent,complete}.json`
are inputs for the existing unmodified `scripts/screen_score.py`; the independent
grader must seal input/source identities and produce results in a new path.
All candidate/control exports use common FP32 precision. PUB143 remains an
inferred sensitivity scenario. Target/class/event-age tables and plots remain
required in per-experiment `eda/` directories when data are scored.

`dependencies.json` records path existence without reading multi-GB caches.
All parent inputs existed at materialization. Missing head/export inputs are
the intentionally uncreated parent/head checkpoints. Root/provider transport
must hash source and actual dataset/cache identities before admission, including
portable source manifest remapping for the private user dataset.

## Resource probe and launch order

1. Independently review source and unit evidence, seal a unique attempt identity.
2. Inspect provider effective RAM, device memory, quota and existing jobs. These
   drafts request one worker,2CPU,18 decimal GB RAM,4.2GB allocator/4.8GB VRAM
   stop,loader workers0. They require at least25.715GB effective remote RAM
   to stay within70%; a smaller provider needs a new reviewed memory design.
   Locally,18GB exceeds current headroom and must not launch. Parent support
   reserves,8GB MemAvailable reserve and cutoff remain mandatory.
3. Stage each production-shape resource probe using the actual full-data loader,
   B32/T1000,H256,two layers,teacher caches,auxiliaries,EMA and Muon. The existing
   profiler receives two measured batches after one warmup; probe has no
   training-state checkpoint and its predictions are never science candidates.
   Separately account loader parent-tree RSS, page cache,cgroup memory and
   peak observed accelerator bytes. Probes are charged attempts,not free.
4. If any probe is killed/overflows, save raw status,oom counters and all resource
   samples; amend only resource implementation and resubmit fresh reviewed
   identity. No blind retry. Two mitigated retries maximum; no post-cutoff starts.
5. Only measured feasibility plus controller GPU-hour reservation admits full
   fits. Fit control and each declared arm,then each head,then fixed exports.
   Hash/checkpoint dependencies mean downstream specs seal only after upstream
   artifacts exist. Independent grading and confirmation follow the contract.

Resource profiles and timing have not yet been measured. The18GB worker cap is
a draft request,not a claim that the loader fits. Existing compressed target
caches may need a separately reviewed transport/loader repair.

## Implemented interfaces and limitations

- `--readout-activation {gelu,fixed_mix}` defaults to historical GELU. Mixture
  introduces no parameters/RNG draws; all state-dict keys remain compatible.
  Head inheritance retains activation and rejects an explicit contradiction.
- `--distill-reanchor` defaults0; nonzero requires constant-alpha distillation
  and bootstrap horizon>0. Both ordinary and pooled paths use the same changed
  target; original-label metric weights are untouched.
- Candidate prediction via `predict_run.py` reconstructs from checkpoint args.
  `export_onnx.py` and `package_submission.py` now reject non-GELU readouts,
  including shared merged heads. Packaging rejects before replacing any
  existing package directory. Fixed_mix remains research-only until an
  independently reviewed native port and full deployment checks exist;
  provenance marks native_export_supported=false.
- Historical default model source is retained in `original_nn.py`, SHA256
  388af04570f0a3944159bf86c1a234dc55c78b951804c38ef59fa314f92f3db3.
  Historical trainer/controls snapshots live in ../supervision/.
- Optimizer continuation across source revisions remains rejected by existing
  frozen resume contracts. This change does not weaken source identity checks.

First bounded verification:27 tests passed in3.47s,including archived default
bitwise outputs/gradients/optimizer update,init bytes/RNG,checkpoint roundtrip,
head inheritance,causality/chunking,strict survival boundary/clip algebra,
pooled loss/gradient oracle and invalid-CLI fail-before-data. Newly introduced
modules and changed training_controls pass Ruff. Existing nn.py/train_rnn.py
and their exact archives fail repository-wide formatting already; they were
not reformatted to avoid unrelated churn. Existing test_training_controls.py
has two pre-existing NPY002 warnings for intentional legacy-RNG-state checks.

After adding export guards, the bounded suite passed **28 tests in 3.15s**;
`cpu_tests_v2.xml` contains the raw JUnit receipt. New guard checks exercise
ONNX, single native, merged native and package-entrypoint refusal, including
preservation of an existing package. `git diff --check` and Ruff on newly
introduced modules plus changed training_controls pass. E-493/H-121 and
E-494/H-124 ledger packets include exact source hashes and the test receipt
hash; root owns insertion into the shared ledgers.
