# Function-preserving GRU growth

[Root C/G/F/W fit delivery authority](ROOT_CGFW_FIT_DELIVERY_AUTHORITY_R1.json) conditionally releases the four frozen two-epoch fits after all four actual resource profiles pass. It binds exact parser-compatible worker vectors, a conservative6.5GB host reservation, explicit4.3GB allocator setting,5400s per-fit timeout and fresh aggregate admission each time. It assigns checkpoint validation, full-valid FP32 export and independent grading as owned continuation work; one exposed F0 seed does not establish a final architecture winner. This operational authority changes neither targets nor sampling/normalization/horizon.

Root owns H-STUDENT-PARENT-GROWTH-20261015 through fitting, export and grading.
This directory records an engineering result, not a trained quality improvement.

- [Complete-sequence result](COMPLETE_SEQUENCE_PROFILE_R1.json) pins the original
  F0 seed3 parent, fitting-only raw fixture, source hashes and derived initializer.
  H256×2→H384×2 preserves all23 outputs over20,000 rows to maximum absolute
  difference1.14441e-5 (RMS9.92251e-7); old hidden coordinates differ at most
  1.84774e-6. Reset replay is exact. This CPU-only check took33.09s and peaked at
  576,081,920 charged bytes. No labels, masks, fitting or quality score were used.
- [Complete-sequence admission](COMPLETE_SEQUENCE_ADMISSION_R1.json) records the
  fresh memory bound and two-core accounting while the existing GPU fit continued.
- [Focused tests](CPU_TEST_RESULT_R1.json) and [admission](CPU_TEST_ADMISSION_R1.json)
  record nine passing tests: four growth tests and five field-age tests, including
  the production Torch age transform. Growth checks cover gate layout, skip-column
  remapping, chunk/state parity, reset, representative new-unit gradients and
  unsupported-layout rejection. They do not prove every new gate receives gradients.
- [Growth implementation](../../../src/wnn/gru_growth.py),
  [tests](../../../tests/test_gru_growth.py), and
  [actual-parent consumer](../../../scripts/profile_gru_growth.py) preserve old
  gate rows while zeroing incoming connections from new units; old readout skip
  columns are relocated after the enlarged recurrent block. New units remain
  trainable. Independent Luna/max source review found no blocker; root added
  embedded membership/split checks following that review.
- [Matched next wave](WARM_MECHANISM_WAVE_R3.md) shares one newly fitted control
  across the age and growth questions. It supersedes unlaunched age R2; no fit
  admission or new GPU profile is claimed here.

The ignored initializer is at
`artifacts/runs/gru_growth_combo_s3_f0_h384_r1/model_ep3.pt` in the main repository,
SHA256 `622a4afc1423fa9fdb976f73dd786e29c946f4710e72c373a69385f1220f1ce0`.
Its sibling provenance preserves the parent ancestry. Epoch3 denotes the source
parent epoch, with **zero new training updates** and no optimizer state carried.
The original parent checkpoint SHA is
`d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e`.
It is a derived initializer, not an independently fitted checkpoint.

FP32 mathematical equivalence permits floating-point differences; stochastic
training trajectories and native/quantized serving equivalence are not established.

[Independent consumer/recipe review](../../unified/reviews/GRU_PARENT_GROWTH_WARM_WAVE_R3_REVIEW_20261015_R1.md)
permits the four bounded resource profiles, with the corrected seeded-order
description. [Delivery authority](PROFILE_DELIVERY_AUTHORITY_R1.json) assigns
them to the local delivery actor after the healthy alpha0 chain, subject to exact
source/input checks and fresh aggregate admission. Full fits are not admitted yet.

[Prepared trainer vectors](WARM_MECHANISM_COMMANDS_R3.json) make the four-arm comparison reviewable. They require exact parser/wrapper/source binding and actual GPU profiles before execution.

[Trainer source manifest](TRAINER_SOURCE_MANIFEST_R3.json) pins the frozen implementation worktree at56d900e3. Direct trainer commands set explicit PYTHONPATH and data root, avoiding the historical wrapper that would select old trainer source. Serial profiles use five B16 batches (50 attempted C2000 updates), resource/rate evidence only.

[Initial controller prep R1](WARM_MECHANISM_PROFILE_CONTROLLER_PREP_R1.json)
preserves the first command materialization; it was superseded before launch
when root fixed the intended allocator cap and H256 host reservation.
[Current controller launch vectors R2](WARM_MECHANISM_PROFILE_CONTROLLER_PREP_R2.json)
freeze serial C/G/F/W `bandit_resources` commands from the reviewed immutable
source (`56d900e3…`). All64 manifest files were rehashed with no mismatches;
the packet binds the split, original parent, grown initializer, teacher-cache
metadata, propensity model/sidecars, and the verified dataset-body/identity
receipt. Its only worker-argv delta from the R3 recipe packet is explicit
`WNN_GPU_MEMORY_BYTES=4300000000`; no model, data or training arguments change.
Independent paired grades are terminal per root's update. R2's C launch reached
the immutable trainer parser and failed on unsupported arguments before CUDA;
G/F/W were not launched. The exact failure and the corrected successor are
indexed below.
Root directed 5.5GB host reservations for H256 C/G/F (C's matched A0 P1 peak
was4,952,576,000B) and6.5GB for H384 W (historical H384 peak was
6,499,999,744B). G/F's CUDA age transforms may raise working memory, so the
profile must measure it. Each probe is 50 attempted B16/C2000 FP16 updates and
provides resource/rate evidence only; G/F construct features through frozen
CUDA `build_inputs(x)` with loader workers at zero. The5GB observed-VRAM ceiling
is an operator stop threshold because the controller records VRAM but does not
enforce it as a hard GPU cap. No fit is admitted by R2.

R2's first C attempt is preserved as an execution failure: the immutable trainer
parser exited2 before CUDA/model work because it rejected the four options
`--distill-loss mse`, `--distill-channel-weights 1,1`,
`--label-weight-power 1`, and `--label-weight-floor 0`. The [attempt receipt](../../campaign_2026-10-06/bandit_continuation/runtime/attempts/warm-growth-r3-profile-c-20261015.json)
SHA256 is `97a2853d6bd2dcf39fe6e1f5d110efd47ace34c02091e42c8c9a0c35249517a7`;
the [worker log](../../campaign_2026-10-06/bandit_continuation/runtime/logs/warm-growth-r3-profile-c-20261015.log)
SHA256 is `7bd7aeff56847b8a7c81c8d65a94e4159d342d828a32a925b0696d0bd49963d3`.
No profile metric was produced, so this is not a resource or scientific result.

[Controller prep R3](WARM_MECHANISM_PROFILE_CONTROLLER_PREP_R3.json) is the
compatibility successor. It preserves R1/R2 and the failed C evidence, proves
from the immutable cached-MSE implementation that these four default-valued
options are identities, and removes only those pairs from the C/G/F/W trainer,
fit and profile vectors. It retains the4.3GB allocator setting, per-arm host
reservations, data/model/recipe pins and5GB observed-VRAM operator threshold.
The [parser-helper failure receipt](WARM_MECHANISM_PARSER_PREFLIGHT_HELPER_FAILURE_R1.json)
preserves a brief vector-extraction error; root superseded that custom helper
gate. A separate [input-check helper failure](WARM_MECHANISM_PRELAUNCH_HELPER_FAILURE_R1.json)
preserves an incorrect guessed JSON key (`train_and_valid`); inspection of the
actual trusted receipt established top-level `datasets.train` and `datasets.valid`.
No Parquet payload was read. The [first prelaunch check](WARM_MECHANISM_PRELAUNCH_CHECK_C_R1.json)
preserves a single manually mistyped historical-attempt hash; the [corrected check](WARM_MECHANISM_PRELAUNCH_CHECK_C_R2.json)
derives that pin from the file and passes the unchanged source, command, data
identity, model/cache, propensity and output-absence checks. The 34GB Parquet
bodies were not rescanned. The corrected C profile itself is the actual
parser/consumer check.
Fresh source, input, absence and resource checks still precede each 50-update
profile.


## R3 profile consumer failure and R4 correction

The R3 corrected C attempt reached the immutable trainer's post-argument guard, which rejects `--save-training-state` when `--profile-steps` is active. It exited 2 after 5.06s with the exact error `profiling/parity early exits cannot save epoch continuation state`; the profile run directory was absent afterward. This is a profile-command compatibility failure before model work, not a training or quality result. The [attempt receipt](../../campaign_2026-10-06/bandit_continuation/runtime/attempts/warm-growth-r3-compat-profile-c-20261015.json) SHA-256 is `fe4f3a398a542f4508c1e1ae683204ec73dcd743badc38ec2ac04f0653227c21` and the [worker log](../../campaign_2026-10-06/bandit_continuation/runtime/logs/warm-growth-r3-compat-profile-c-20261015.log) SHA-256 is `c77423f7ae934d3348a678d45632681330dbc9a7161222baf98f7e2edb7366ba`.

[R4 controller prep](WARM_MECHANISM_PROFILE_CONTROLLER_PREP_R4.json), SHA-256 `b5b2691568cbddf3b4a6e4399d5a6cf6b0a33ff39dc7f71b305269a38365754d`, makes the narrow profile-only repair: it removes `--save-training-state` from the profile-worker vectors because these five-batch early exits cannot save continuation state. The base trainer and full-fit vectors are unchanged, as are the architecture, data, initialization, optimizer, schedule, 4.3GB allocator setting, per-arm reservations and 5GB observed-VRAM stop threshold. The next C run is the actual immutable trainer consumer check; G/F/W follow serially only after the prior arm terminates successfully.


## C profile result

The [C resource-profile receipt](WARM_MECHANISM_PROFILE_C_RESULT_R1.json) records a successful five-batch/50-update run through the R4 command: 50 optimizer attempts and 50 completions, zero AMP skips, and finite loss and gradients. It completed in 20.21s. The trainer reports a median full-sequence-batch wall time of 1216.41ms (B16 across 20,000 rows), 13.15 sequences/s, and 632.13ms median loader wait. Each batch covers ten sequential 2,000-row chunks; dividing the batch timing gives an estimated 121.64ms/chunk, not a separate measurement.

The worker peaked at 4,953,374,720B host memory under the 5,500,000,000B reservation, and 2,178,940,928B observed VRAM, below the 5,000,000,000B operator stop threshold. Worker and parent OOM deltas were zero. This supports resource/rate feasibility for C only; it does not establish fit duration, quality or a full-fit admission. G is next in the serial profile order.


## G profile result and reservation note

The [G resource-profile receipt](WARM_MECHANISM_PROFILE_G_RESULT_R1.json) records 50 of 50 optimizer updates completed, zero AMP skips, and finite loss and gradients. The attempt took 20.19s; its median full B16×20,000-row batch time was 1252.38ms (12.78 sequences/s). Peak host memory was 3,612,647,424B, observed VRAM 2,669,674,496B, and worker/parent OOM deltas were zero. As with C, the divided 2,000-row chunk timing is an estimate; the receipt makes no fit or quality claim.

The [reservation erratum](WARM_MECHANISM_PROFILE_RESERVATION_ERRATUM_R1.json) records that the R4 descriptive reserve for G/F is 5.5GB, but their frozen outer commands and the actual G admission use 6.5GB. This conservative over-reservation is within the reviewed aggregate target and leaves the model command unchanged; F/W proceed at their frozen 6.5GB outer-vector reserve.


## F profile result

The [F resource-profile receipt](WARM_MECHANISM_PROFILE_F_RESULT_R1.json) records a successful five-batch run: 50/50 optimizer updates, no AMP skips, finite loss and gradients. It took 15.19s and measured 1120.99ms per full B16×20,000-row sequence batch (14.27 sequences/s). Peak host use was 4,267,720,704B and observed VRAM was 3,263,168,512B, with zero OOM events. The run remains resource/rate evidence, not a fit or quality result.

F used the frozen outer command's 6.5GB host reservation; the [reservation erratum](WARM_MECHANISM_PROFILE_RESERVATION_ERRATUM_R1.json) now includes its terminal receipt. W is the final serial profile.
