# Point 5 B300 receipt: update telemetry, LR controls and runtime context

Status: implemented and locally checked; ready for root review. No B300 fit,
GPU execution, production data read or speed measurement was run.

## Sampled optimizer-step diagnostics

The optional config field `optimizer_update_diagnostics_every` enables the
`optimizer_update_diagnostics` object on sampled rank-log records. Sampling uses
one-based global update numbers: update 1 and each multiple of N are sampled,
so a 50-update profile captures updates 1 and 50. When the field is absent,
there is no weight snapshot or extra gradient-stat pass.

The main new standalone-label preset now opts in at cadence 50 and declares
`gradient_horizon_rows=256`. It sets `lr_head_factor`, `lr_body_factor` and all
four Muon/AdamW role factors to 1.0, so it records head/body and role groups
without changing their configured learning rates. The three profile-only configs
use the same neutral factors so their diagnostic group names match the main
preset. Frozen v1 configs remain unchanged.

At sampled steps the producer snapshots trainable FP32 weights to CPU before
the optimizer step. After the step it records actual applied learning rates
captured before `scheduler.step()`, global and per-group preclip/postclip
gradient norms, weight and delta norms/RMS, relative delta, and optimizer state
norms. `momentum_norm` reads `momentum_buffer`; `exp_avg_norm` reads AdamW's
`exp_avg`. Each count name distinguishes parameter tensors from scalar elements.
RMS divides the group norm by the square root of `parameter_scalar_count`.
`weight_norm_denominator_floor` is `1e-12`; the per-group
`zero_weight_denominator_count` is 1 when the pre-weight norm is at or below
that floor, otherwise 0.

The probe adds no forward pass and does not mutate RNG, gradients, parameters or
optimizer state. It uses the ordinary clipping, optimizer, scheduler and
zero-grad order. Unsampled steps do not clone weights. Large-model host-memory
and transfer overhead were not measured; this instrumentation is opt-in.

## Independent LR controls

New fields are `lr_role_factors`, `lr_head_factor` and `lr_body_factor`.
Role factors use the actual optimizer role names, including Muon roles such as
`muon_qkv`, `muon_hidden`, `adamw_matrix` and `adamw_no_decay`; plain AdamW uses
the role `adamw`. Head/body factors multiply the role factor. Parameters named
`head.*` or `output_head.*` are the head; all other parameters are the body.
Groups split into stable `<role>_head` and `<role>_body` names only when a
head/body factor is explicitly configured. Factors are applied to the base LR
before `LambdaLR` construction, so schedule ratios and resume state retain the
configured multipliers. Unknown or unused role keys are rejected; when
head/body factors are requested, models without both partitions are rejected.
Omitted controls preserve the old groups and rates;
the frozen four-fit v1 modes reject these new fields.

## Runtime context identity

The new runtime context contract is added to v2 data provenance before run
identity hashing. It describes 20,000-row sequences, Transformer attention
window size per layer, maximum cached KV rows (`context_rows - 1`), and GRU
continuous hidden state separately. It records the actual TBPTT detach chunk,
the nominal optimizer window (`chunk_rows * accumulate_chunks`), and the final
partial window. The carried state is detached at chunk boundaries but retained
across optimizer updates, so later chunks consume a stale approximation computed
with earlier parameter values. Accumulation does not extend the gradient horizon.
If `gradient_horizon_rows` is declared, it must equal `chunk_rows`.

Three complete profile-only configs are in [configs/profiles](configs/profiles):

- `profile_only_chunk512_accumulate2.json`
- `profile_only_chunk1024_accumulate1.json`
- `profile_only_world1_batch256.json`

They opt into cadence 50 and explicitly declare the matching gradient horizon.
The first two keep the 1,024-row nominal optimizer window; all three use the
same global batch of 256. Pure schedule arithmetic with the official 10,325
fitting-member count produces 20 optimizer windows per sequence and 820 updates
per epoch for each variant. This calculation reads no source data and makes no
throughput claim.
These configs were not executed and are not approved fit results.

## Checks

Pinned Python:
`C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe`.
The focused test run used one CPU thread per BLAS/OpenMP library and
`PYTHONPATH=<worktree>/src;<worktree>`:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONPATH='C:/Users/admin/Documents/Codex/reviews/WunderFunder-training-runtime-fixes-20261108/src;C:/Users/admin/Documents/Codex/reviews/WunderFunder-training-runtime-fixes-20261108'
Remove-Item Env:CUBLAS_EMULATE_SINGLE_PRECISION -ErrorAction SilentlyContinue
& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  -m pytest tests/test_sequence_update_telemetry.py tests/test_sequence_muon.py `
  tests/test_sequence_standalone_labels.py tests/test_sequence_monitor_diagnostics.py `
  tests/test_ensemble_objective.py tests/test_sequence_reference_cache.py `
  tests/test_sequence_blend_objective.py `
  tests/test_pr67_receiving.py::test_report_only_monitor_composes_before_clipping_and_restores_rng -q
# 82 passed in 6.05s
```

After the schedule arithmetic was corrected to the official 10,325 fitting
members, that focused update-count test was rerun separately and passed (1
passed in 3.69s); the other source code was unchanged.

After enabling cadence/horizon and neutral LR factors in the main v2 preset,
the full-preset assertions, profile-variant schedule check and opt-in validation
tests passed (3 passed in 4.07s).

The CPU checks cover sampled/unsampled actual-step parity after warmup,
pre/postclip metrics and counts, RNG/gradient/parameter/optimizer/scheduler
parity, first-and-50th sampling, zero-weight denominator handling, Muon role and
head/body LR composition, scheduler resume ratios, unknown roles, v1 config
rejection, horizon mismatch, runtime context, and update-count parity. The
legacy v1 identity fixtures also remain in the selected run.

Ruff check and format check passed for the changed Python files. The training
structure check reported 120 files, 0 errors (module cap 300 lines, function
cap 200 lines). `git diff --check` passed.

## Limits

The optimizer parity fixture uses a tiny CPU model and mocks the CUDA timing
event; it does not establish B300 memory, host-transfer, timing or throughput
cost. No official data, full validation pass, fit, checkpoint or prediction was
created. The profile-only candidates remain unrun. No quality or serving gain
is established.
