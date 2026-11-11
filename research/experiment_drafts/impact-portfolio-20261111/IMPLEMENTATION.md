# First three experiments: isolated B300 implementation

The user's 11 November instruction implements cards01–03 in this same PR, with
no changes to the existing pipeline and no optimization work. Cards04–10 remain
proposals. This supersedes the earlier suggestion to open separate implementation
PRs for these three cards, including the historical PR8 continuation.

All new executable files live under this packet. Existing `src/`, `scripts/`,
training configs, optimizer, training loop, validation and checkpoint code are
unchanged. The runner consumes the exact PR75 source commit
`8626000a4b965763a70011a0291a891aa6535d7e` from a separate read-only worktree.
It does not check out or patch a running team's directory. Both source trees are
hashed into profile/checkpoint identity, and dirty or wrong runtime sources fail.

## Models and fixed comparison

- **01 PLE:**16 train-only quantile knots,4 learned values per knot, continuous
  interpolation with a raw projection bypass. Duplicate knots are collapsed;
  constant features contribute zero. The added projection starts at zero.
- **02 Groups:** the actual raw112 schema forms16 current-row groups, each
  width64; one4-head within-row attention block with FF128 and mean pooling adds
  a zero-initialized residual to the raw projection. No temporal lookahead.
- **03 Target adapters:** each target gets a rank8 GELU residual before its
  original output row; both targets train the shared backbone. Up-projections
  start at zero. This is not the target-selective-loss screen from PR77.

The common Transformer is H768/L24/12heads, FF4, context1024, gradient horizon256,
batch128 per runtime semantics, accumulation4,2 B300 ranks. Four epochs, full train
and full valid, finalE4 primary; Muon/AdamW roles, LR3e-4, WD.01, clip1,
5% warmup/cosine floor.1, BF16/BF16x9 stay exactly as the pinned plain config.
Existing compiled temporal blocks remain as in that recipe; no new kernel,
precision, scheduling or throughput optimization is introduced. New parameters
use the existing optimizer's semantic fallback groups; no optimizer is replaced.

Each card has two controls and two treatments at seeds204101/204102: **12 new
scientific fits total**, zero reused controls and zero teacher/pretraining fits.
The twelve `configs/ip*.json` descriptors point to two native configs differing
from the saved PR75 plain config only in seed. Shared seeds are not independent
folds. No fitting has been performed or automatically scheduled by this PR.

## PLE preprocessing decision

The draft required train-only quantiles but did not specify an estimator. This
implementation freezes **64 evenly spaced row positions from each of all10,325
train sequences**, chosen without labels, for660,800 quantile samples. Quantiles
are linear empirical estimates of that fixed sample, not exact quantiles over
every training row. All fitting sequences are represented. All model training
still uses the complete dataset; the canonical normalizer still uses **all fit
rows**, unchanged. Valid/test data never enter knot construction.

`preprocessing.json` binds feature order, train membership, split, raw-source
fingerprints, canonical moments, knot counts/values and the estimator. The
receiving host writes it once; every arm uses the same artifact. Its temporary
sample file is removed only after successful preparation. Checkpoints embed the
small artifact, so inference does not recompute statistics or require that sample.

## Integration and evidence

`runtime/worker.py` admits the pinned runtime before importing models.
`impact81/fitting.py` owns setup/composition and calls the existing `run_profile`
and `run_full_fit`; it contains no copied epoch/update loop. It uses canonical
data validation, monitor, DDP, losses, optimizer, profile/reload, resume and
checkpoint functions. The standard train/validation telemetry is preserved.

`impact81/exporting.py` reconstructs the experiment-aware final model and writes
canonical prediction manifests. The stock model loader cannot reconstruct the
new modules; do not call it on these endpoints. Scoring delegates to the original
`sequence_score.score_export`, including pooled WP and full coverage. The
stdlib-only comparator requires all12 fixed endpoints or an explicitly selected
four-endpoint card; it retains both seed/target deltas without choosing a best
epoch or interpreting them as public improvement.

Server-only checks cover initial-function agreement, new-parameter gradients,
causal prefix/reset/chunk behavior and reconstruction. They use a small
Transformer to check semantics, not to predict full-model capacity or quality.
After initial agreement, probes give the added projections fixed nonzero weights
so the causal check cannot pass merely because every residual branch is zero.
Every full-size cell additionally requires its own canonical50-update profile,
model/optimizer/RNG reload and an operator-approved receipt before full fitting.
The profile measures early warmup; it does not certify peak-LR stability or
whole-run duration. The comparison still needs actual completed B300 evidence.

[Static implementation review](STATIC_IMPLEMENTATION_REVIEW.json) records source
parsing, lint, configuration and command-syntax checks only. It is not a model
test, profile, numerical parity result or receiver acceptance.

`runtime/launch.py` launches **one stage** with a named owner, explicit GPU IDs,
timeout and logs. It is not a queue. It never retries or searches for occupied
devices and terminates only its own subprocess group on timeout/interruption.
Existing jobs are untouched. Model/data stages reject macOS execution.

MIDAS was actually used for a metadata-only route preview. It reports no remotely
launchable worker. The separate [MIDAS handoff](midas/README.md) exports/imports
source-bound evidence after real runs; a registered handoff is not GPU execution.
Neither MIDAS nor this packet starts jobs from a PR publication.

See [B300_COMMANDS.md](B300_COMMANDS.md) for preparation, checks, profile, fit,
export, canonical score, paired comparison and MIDAS evidence commands.

Local validation is static only: no tests, models, training, inference or profiles
were executed. Runtime/numerical correctness and speed remain receiver-pending.
