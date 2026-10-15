# Function-preserving GRU growth

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
Independent paired grades are terminal per root's update; R2 remains
`PREPARED_NOT_LAUNCHED` until the independent reviewer signals CPU-handoff-DONE,
then it still requires fresh per-profile controller source/resource checks.
Root directed 5.5GB host reservations for H256 C/G/F (C's matched A0 P1 peak
was4,952,576,000B) and6.5GB for H384 W (historical H384 peak was
6,499,999,744B). G/F's CUDA age transforms may raise working memory, so the
profile must measure it. Each probe is 50 attempted B16/C2000 FP16 updates and
provides resource/rate evidence only; G/F construct features through frozen
CUDA `build_inputs(x)` with loader workers at zero. The5GB observed-VRAM ceiling
is an operator stop threshold because the controller records VRAM but does not
enforce it as a hard GPU cap. No fit is admitted by R2.
