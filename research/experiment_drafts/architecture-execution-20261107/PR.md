# 1. Question and hypothesis

Execute the authorized compact architecture panel: fresh GRU control versus
local-attention GRU, pinned Mamba-2 and causal EMA/custom TabM-style readout.
Does the fixed final endpoint improve quality or prediction diversity?

# 2. Idea branch and compute advantage

Student-capable regressors: four families × two matched seeds, eight parallel
L4 fits. Different parameter/compute costs are measured, not equated. This is
not an exact reconstruction of GALAM or official TabM.

# 3. Mechanism and falsifier

Test selectable recent context, selective state and cheap multiscale features.
Retain negatives. Falling loss or best intermediate valid does not promote an
architecture when its fixed final predictions fail to support the mechanism.

# 4. Method and frozen recipe

See CONTRACT.md, EXECUTION_R1.md and l4/configs-runtime-r3. Raw112/full20k
sequences, chunk2000, B16/A1, AdamW3e-4/decay.01/clip1, warmup5%/cosine,
fixed12E, seeds101001/101002. Fresh initialization, no parents or KD. FP32;
Mamba Triton FP32 dot mode is explicitly IEEE and checked on actual kernels.

# 5. Bounded fitting panel

Exactly eight fresh L4 fits, one/GPU, no automatic expansion. The four B300
teachers are a separate allocated packet. L4 has24h from first actual profile,
including a1h collection reserve. Engineering profiles are not scientific fits.

# 6. Validation and uncertainty

Official earlier train10325 only; full later valid1873/3528273 scored rows.
Metric global_wp_clip2_v1, fixed epoch12. Two seeds cannot establish reliable
training variance or significance. Valid is development-exposed. Diagnostic
train-eval and valid masks differ and are labelled; no public gain is forecast.

# 7. Leakage and coverage

ADR0044; split0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c.
Zero valid fitting/ancestry; fitting-only normalization, causal inputs, reset
between sequences. No random-fold primary selection or inferred seq_ix order.

# 8. Decision-changing result

Return each fixed endpoint and paired control. Promotion needs complete
source-qualified predictions and independent evidence review, including cost
and any explicit ensemble contribution. No numerical gain is asserted at launch.

# 9. How to run

Already launched by ROOT's l4/receiving controller; do not duplicate it. Runtime
/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1. Immutable
source-r1 and exact capsule/config/profile/owner receipts are described in
EXECUTION_R1.md. Detached workers survive SSH loss; cleanup is ownership-bound.

# 10. Evidence to return

Native metadata, objective/LR/preclip gradients, clip incidence, update/weight
diagnostics, full-valid each epoch, resource telemetry; safe/all-epoch checkpoints,
predictions, hashes, fixed endpoint report and separate evidence/weights archives.
Actual local dashboard: http://127.0.0.1:8769/. Immutable native-prefix snapshots
and real PNG/SVG epoch plots are retained. GitHub reporting is separate from fit.

# 11. Results and trace

All eight workers produced actual updates/full-valid events; these are partial
development observations, not complete epoch12 comparisons. See EXECUTION_R1.md
and receiving-evidence. Independent receiving/profile/precision reviews preceded
admission. Old sources/failures remain retained. No competition upload or extra
fit is authorized by this draft.
