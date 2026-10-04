# E425 soft-route queue admission draft

This document describes the source-only controller at
`checks/run_e425_sequence.py`. It does not admit or start work. Root retains
authority to seal the smoke/full-fit specifications and start the controller.

The controller will wait for all of the following before touching E425 jobs:

- `checks/e426_sequence_v1.json` reports `fits_complete`;
- `wnn-e426-sequence-v1.service` is inactive and reports exit status 0; and
- all four named E426 smoke/constant/aligned/shuffled attempts report finished,
  exit code 0, and verified teardown.

It refuses a stopped E426 sequence, a missing live handle before completion,
or any failed attempt. It does not restart failed work. For each E425 attempt,
it consumes only the root-sealed JSON, checks the exact record/spec identity,
and observes that attempt's recorded service handle to terminal success.

The first E425 attempt is `e425-soft-smoke-2b-v1`, a one-epoch diagnostic
using two complete batches. The acceptance check expects two batches, 40
optimizer attempts and 40 EMA updates; completed optimizer updates plus AMP
skips must equal 40. It also requires withheld evaluation, fixed epoch 0,
finite route entropy in `[0, log(4)]`, normalized finite route masses, and
well-formed per-expert gradient and movement diagnostics. A nonfinite gradient
is retained as an explicit count; it is excluded only from that expert's
finite-gradient mean. Since the model records fast-path route probabilities
after FP16 conversion, the controller allows a conservative 0.001 probability
mass error and 0.002 entropy excess above `log(4)`; these are telemetry
rounding tolerances, not model acceptance criteria.

After the smoke succeeds, `e425-soft-s3-final-v2` runs the full four-epoch
recipe with 378 complete batches per epoch. Each epoch must report 7,560
optimizer attempts and EMA updates, with completed optimizer updates plus AMP
skips equal to attempts. The selected checkpoint is fixed to epoch 3. The
sealed recipe retains the combo training pool, distillation teachers, EMA and
auxiliary losses, propensity source, and optimizer controls. It does not
select a checkpoint from holdout scores. Full-fit provenance must match the
learned and universal E424 controls for fit membership, normalization, and
propensity checkpoint identity. The GRU-budget control has a documented
normalization difference and is not used for this identity assertion.

Only after the full checkpoint and provenance pass the control-identity checks
does the controller write and seal the reviewed FP32 all-VALID prediction spec
`specs/e425_predict_c20261004_h106_soft_s3_final_fp32_v1`. It pins the actual
`model_ep3.pt` hash, the all-VALID control sidecar, and the staged predictor
plus its complete staged/root WNN source closure. The prediction stage refuses
existing unmatched outputs and checks the fixed checkpoint, all 1,873 VALID
row groups, prediction hash, soft-route identity, FP32 mode, and source-closure
hash. It keeps those exported groups distinct from the 368 historical holdout
groups in provenance, which remain the paired-score panel. The terminal status is
`fits_and_exports_complete`, with paired scoring still `pending` for the
root-owned scoring controller.

The full-fit sealed resource limits currently specify an 8-CPU allocation, a
24-GiB host-memory cgroup, a 4.2-GB GPU allocator cap, a 4.8-GB GPU stop, and a
5,400-second attempt timeout. The smoke has a 900-second timeout. The
controller uses the campaign pause/authorization guard and adds no new
resource policy or automatic retry.

The soft-route expert-gradient diagnostics cover gradients from the complete
objective flowing through expert predictions (main, auxiliary, distillation,
and EMA consistency terms where enabled), after unscale and configured
gradient dynamics. They are not isolated main-label gradients. Route usage,
gradient support, and actual parameter movement are diagnostics; the
controller does not reject a fit for expert collapse or low movement.

No E425 queue job or prediction is launched by this draft. E426 completion,
the root-sealed fit specs, and the exact attempt records remain external
admission inputs.
