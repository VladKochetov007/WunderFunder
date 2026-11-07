# 1. Question and hypothesis

Execute PR67's fixed clean-TL3/Transformer complement comparison. Does fitting
the actual0.9TL3+0.1Transformer output improve the final fixed blend compared
with a label-trained Transformer? This extends receiving support, not evidence
that the original proposed mechanism works.

# 2. Idea branch and compute advantage

Teacher/link improvement, four fresh teachers on eight owned B300, two GPUs
each. Existing qualified train-only TL3 parents are reused; their native exports
and baseline replay are engineering work. No student fitting is included.

# 3. Mechanism and falsifier

Curvature-scaled composite WMSE targets the deployed fixed blend rather than
teacher standalone quality. Retain both seed outcomes, prediction diversity
and fixed-blend deltas; lower scaled train objective alone cannot support gain.

# 4. Method and frozen recipe

S3 source6ab2f679e70928f913b97d5a29f0d1e09ceacaa3; T768/L24/H12,
170198786 parameters, context1024, chunk256, B4/A4/world2, Muon3e-4,
warmup5%/cosine floor.1, clip1, fixed30E/774600 updates. BF16/BF16x9 with
FP32 master/gradient/loss/state tensors; no IEEE-FP32 parity claim.

# 5. Bounded fitting panel

Labels/complement × seeds101001/101002 =4 fresh fits,12 engineering profiles,
no automatic expansion. Measured~.71s/update implies~6.4days core fitting;
6–8days is planning, not a guarantee. Before first fit ROOT amended initial24h
to finite9days, preserving caps and1h collection reserve. Native single-worker
observer is capped7days; exact same-fit continuation requires expiry/ownership
and safe checkpoint review within the same9-day maximum.

# 6. Validation and uncertainty

Official earlier train10325 only, zero valid fitting; later full valid1873 /
3528273 scored rows, global_wp_clip2_v1. Final epoch30 and fixed blend are
primary; best observed valid is diagnostic. Two seeds do not establish robust
training variance/significance. Valid is development-exposed; public effect unknown.

# 7. Leakage and coverage

ADR0044 and split0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c.
Fitting-only normalization and qualified TL3 ancestry; causal input/reset,
complete sequences, no seq_ix-inferred chronology or valid-fitted blend weights.

# 8. Decision-changing result

Compare both paired final fixed-blend deltas, uncertainty, cost and serving
relevance after source-qualified complete grades and independent review. Do
not promote from teacher-only quality, partial fits or zero-crossing intervals.

# 9. How to run

Already running under ROOT-owned receiving/controller.py; do not duplicate.
Use the mandatory short alias /data/epanov_s/tdn7run, frozen source-r3 and
wrapper/run commands in README/EXECUTION_R1.md. Detached processes survive SSH
loss; cleanup and any continuation are receipt/PID-birth bound.

# 10. Evidence to return

Source/config/data/environment/profile/ownership receipts; train/valid curves
each epoch, objective/LR/preclip gradients/clip incidence, resources, safe state,
final checkpoint/predictions, fixed blend score, separate evidence and weights
archives with hashes. Dashboard http://127.0.0.1:8769/ mirrors native bytes.
Terminal reporting distinguishes fit completion, grading and artifact upload.

# 11. Results and trace

Actual81 native tests plus compiled BF16/BF16x9 and two-rank DDP/Muon checks
passed; all12 actual profiles passed and independent review preceded ROOT
approval. Clean-TL3 full-valid replay .6923774248169172 matches the qualified
reference. All4 fits started18:36UTC and made actual optimizer updates. No
final endpoint, new public result or improvement is claimed. See EXECUTION_R1.md
and exact receiving-evidence; failed preparation is preserved, not relabelled.
