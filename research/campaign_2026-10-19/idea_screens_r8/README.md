# R8 constant-variance control

**Completed and independently audited.** [Results](RESULTS.md): learned constant-variance WP0.5013935202, versus stored input-dependent variance0.5142969022 and WMSE0.5019213692. The primary contrast is+0.0129033820 and all32 leave-one-component primary contrasts remain positive. This tested global-scaling control does not reproduce R7's gain, but removing variance-to-trunk gradients also changes optimization: no isolated noise-recovery or full-model/public claim. [Audit](RESULT_AUDIT.json), [resources](RESOURCE_RECONCILIATION.json), [records](records/README.md) and [continuation](CONTINUATION.md) bind actual756updates/0nonfinite, successful exit/capture/stop and untouched F3 maintenance.

The text below originated as the prepared README transported at launch. Later annotations do not change the frozen42 source pins; README is intentionally excluded from that manifest.

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

The source/evidence map is [model](model.py), [single-arm execution](execution.py), [single endpoint scorer](scoring.py), [report/archive assembly](reporting.py), [hash-gated CLI](run_probe.py), [synthetic tests](test_probe.py) and [package marker](__init__.py). [Design](DESIGN.md), [design review](DESIGN_REVIEW.json), [manifest](PREPARATION_MANIFEST.json), [test receipt](TEST_RECEIPT.json), [implementation review](IMPLEMENTATION_REVIEW.json), [root startup review](ROOT_STARTUP_REVIEW.json) and [publication validation](PUBLICATION_VALIDATION.json) separate readiness, actual measurement and publication; none substitutes for the others.
