## 1. Status, lineage and owner

Implement a six-fit teacher-quality screen of post-RoPE query geometry:
pairwise rotations versus matched pairwise gain noise and a fresh plain control.
H/E-B300-QUERY-GEOMETRY-20261114, preparation owner Ivanhacker13/Codex.
Native source PR75 `8626000a4b965763a70011a0291a891aa6535d7e`.
Prepared code/configs/commands; zero model tests, profiles, fits or inference by
this task. Actual MIDAS metadata preview succeeded; GPU execution remains unadmitted.

## 2. Why it could work

Attention may depend on brittle query directions or pair amplitudes. Train-time
rotations force nearby directions to remain useful; gain noise tests amplitude
robustness at matched conditional mean, energy and displacement. The strongest
alternative is harmful destruction of predictive alignment or a gain caused by
shared mean-logit contraction. No label symmetry or transfer gain is assumed.

## 3. Compute advantage

Six complete paired Transformer endpoints can decide whether either regularizer
merits new folds or student distillation. No added parameters, optimizer changes,
performance optimizations or claims of improved serving. Profile the unchanged
real shape with the new query computation before allocating full runs.

## 4. Frozen comparison

Three arms, seeds209401/209402, post-RoPE query only. Rotationθ=±0.35rad;
gain factors cos(.35)±sin(.35). Native 24×768/12heads/context1024, chunk/horizon256,
batch128 per rank, accumulation4, two B300, BF16/BF16x9, Muon+AdamW, LR3e-4,
warmup5%/cosine floor0.1, clip1, E4/3280updates. Clean-label WMSE and original
normalization. Native KV caches and causal masks are not directly mutated.
Evaluation bypasses the perturbation; checkpoint shapes are unchanged.

## 5. Bounded panel

Two plain, two angular, two radial fits: six total, zero reused historical
controls or student/teacher-target fits. Fresh owner/allocation/time/memory limits
are mandatory. No automatic admission or combined perturbations from PR99.
All epoch snapshots are retained; only configured E4 is the primary endpoint.

## 6. Validation and uncertainty

Independent full-valid export and official scorer. Two versus-plain primary
contrasts plus angular−radial diagnostic, all paired seed deltas, equal mean
and conditional per-seed component intervals. Valid is development-exposed;
two seeds do not establish population significance or public transfer. No
best-seed/epoch selection, public-query optimization or formal p-value claim.

## 7. Leakage and coverage

Fit10325 train-source sequences, linked-component exclusions and zero valid-source
fit. Evaluate1873 sequences/3528273 scored rows on the original hash-pinned split.
Random signs depend on native RNG only, not labels, future inputs or evaluation
scores. Query transforms are row-local with signs shared within a native chunk.
All targets, weights, loss denominators and data ordering remain native.
Prepared B300 probes check causality, direct cache behavior and active integration.

## 8. Decision-changing result

For either arm: mean paired WP≥+.001 and both seeds positive relative to plain
justifies separately budgeted confirmation. A rotation-specific explanation also
requires angular−radial positive. Shared query-mean contraction remains a limit
versus plain; covariance/logit variance/entropy are not matched. Preserve mixed
and negative outcomes. A zero-crossing interval alone is not a universal stop rule.

## 9. Run

[GPU_COMMANDS.md](GPU_COMMANDS.md): isolated prepare→check→one-B300 numerical
probe→each two-B30050-update profile→explicit review→fit→E4export/scorer→readout.
Native pipeline files in the repository remain untouched; the patch affects only
the receiver's extracted source. No queue manager, auto-resume or GPU job launched.

## 10. Evidence and MIDAS

Return pinned sources/configs/split, owner/allocation/environment, native loss/LR/
gradient/timing/memory logs, profile+reload receipts, terminal/epoch-index,
checkpoint and prediction manifests, canonical grades and all contrasts.
Keep weights separate from evidence and preserve failures. MIDAS preview and
invocation are saved; its prepared importer accepts independently graded E4
metadata, not invented metrics or a plan presented as a running GPU job.

## 11. Results and trace

[README](README.md) indexes the implementation, six recipes, source/duplicate
audit, independent review and literature limits. Unified ideas and append-only
H/E ledgers link the packet. PR99 input/history deletion, GRU-state consistency,
raw-volume scaling, MixStyle and SAM are explicitly distinguished in the audit.
No numerical acceptance, measured speedup, quality gain or submission is claimed.
