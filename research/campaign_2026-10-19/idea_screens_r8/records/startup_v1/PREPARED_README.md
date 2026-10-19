# R8 constant-variance control

R8 adds one CPU arm to test whether the input-dependent variance objective in
the exposed R7 pilot benefits from row-varying logvariance. The frozen
DESIGN.md defines the fixed model, loss, 756-update schedule, single final
endpoint query, and interpretation limits. Independent pre-code review passed
in DESIGN_REVIEW.json.

model.py preserves the R7 parameter topology and makes the logvariance head
constant by applying it to hidden multiplied by zero. execution.py calls the
unchanged R7 train_arm with the heteroskedastic branch explicitly as an
implementation alias; scoring.py evaluates the one new endpoint once with the
canonical global weighted-Pearson scorer and component sufficient statistics.

The hash-gated entry point is:

    python -m idea_screens_r8.run_probe \
      --sample B/research_tmp/nf19-sample-r3-v2 \
      --output B/research_tmp/nf19-r8-v1 \
      --expected-sample-sha256 <frozen-R3-sample-sha256> \
      --expected-preparation-manifest-sha256 <R8-manifest-sha256> \
      --expected-r7-result-sha256 dd36e82a51672a9b9e2ab694dd92fcd827a2445d607927548200d333d1f4b9e0

The manifest pins the exact R7 runtime/import closure, R3 metadata fixture,
reviewed design, this implementation and the compact stored R7 RESULT.json.
The R7 result is read only for its recorded scalar scores and component
leave-one-out summaries. The runner rejects any runtime other than Torch
2.14.0+cu130 and NumPy 2.2.6 before loading panel arrays.

This folder contains the prepared implementation and synthetic test evidence;
actual panel execution and output capture are owned by the root task. Any
result remains one seed on an adaptively exposed panel, not a fresh full-fold
estimate or latent-noise recovery claim.
