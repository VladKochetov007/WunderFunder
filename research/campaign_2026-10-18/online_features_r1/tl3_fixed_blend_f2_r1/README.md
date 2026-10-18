# TL3/R3 fixed-mixture robustness replay on F2

## Question and scope

This bounded inference-and-grading experiment asks whether the exact R3
submission endpoint contributes to the exact TL3 team package on another fold.
F0 showed that both predeclared fixed mixtures trailed TL3, so F2 checks whether
that result repeats without fitting mixture weights. This tests endpoint
complementarity only; it does not test online learning, a new training recipe,
or a teacher/student relationship.

The predeclared comparison is R3 versus TL3, plus fixed 75/25 and 50/50
TL3/R3 mixtures versus TL3. The model archives, split, weights, metric,
bootstrap, and replay code were frozen before inference in [recipe.json](recipe.json).
The initial scorer had a keying bug; the failed first attempt is preserved, and
the grader-only repair is separately hash-bound in
[recipe_gradefix_r1.json](recipe_gradefix_r1.json). No candidate, score, or
bootstrap definition changed. The experiment is registered as
`H-TL3-R3-F2-COMPLEMENTARITY-20261018-R1` /
`E-TL3-R3-F2-FIXED-MIXTURE-20261018-R1`.

## Evidence limits

F2 has been inspected previously. The TL3 ZIP metadata calls its parents
all-data, while exact training, teacher, initializer, and sequence membership
is unknown. R3 derives from the R2 checkpoint and its alpha was selected on
exposed F0. Therefore any F2 score is cross-fold robustness evidence only; it
is not clean validation, seed-level evidence, public-transfer evidence, or a
submission selection rule. The 10,000-resample component intervals condition on
fixed package predictions and do not cover those other uncertainties.

Replay reads only sequence ID, position, prediction flag, and the current
112-feature input. It processes the first 99 warmup rows, verifies an exact
A-B-A reset on the first sequence, and never reads target or scoring-mask
payloads. A label-blind preflight pins ZIPs, replay outputs, source identities,
and code before the separate score phase maps the official target cache.

## Resource plan and output location

Both exact packages first replayed eight complete groups under a 3 GiB Docker
limit. Each passed exact A-B-A reset validation; cgroup peaks were 0.40 GB for
TL3 and 0.30 GB for R3, below the 2.5 GB full-replay gate. The full TL3 run then
peaked at 2.99 GB under its 3.22 GB cap. Before R3, the support/container split
was adjusted to 1.0/3.5 GB, preserving a 4.5 GB task reservation under the
5 GB cap; R3 peaked at 0.37 GB. The execution receipt records the pre-existing
parent OOM counter baseline and its unchanged final value. Each full prediction
cache is about 70.9 MB; both stay on physical NVMe scratch at
`/home/vlad/development/wunder/research_tmp/online-features-tl3-f2`, not in
`/tmp`, `/dev/shm`, or the repository.

For TL3, the local process was limited to one CPU/1.5 GB and Docker to one
CPU/3 GiB with swap and network disabled. For R3, the local cap was reduced to
1 GB and Docker was capped at 3.5 GB. During both runs the internal campaign
slice was temporarily set to one CPU; it returned to two CPUs after each
container. Full-replay prediction files remain only on physical scratch. The
grader uses scratch-root relative symlinks into `out/` and `receipts/`, because
the protected-hardlink setting rejected links to root-owned Docker outputs; the
71 MB prediction files were not copied. Current admission review is indexed in
the [resume review](../../../../unified/operations/RESEARCH_RESUME_REVIEW_20261018_R1.md).

## Implementation

- [f2_candidates.py](f2_candidates.py) pins both ZIPs and output names.
- [f2_source.py](f2_source.py) validates the frozen F2 membership and feature-only
  producer schema.
- [f2_native_runtime.py](f2_native_runtime.py) loads both native callback names
  while ensuring the extension came from the exact extracted ZIP.
- [f2_replay_engine.py](f2_replay_engine.py) streams complete sequences and
  writes a memory-mapped output cache.
- [f2_receipts.py](f2_receipts.py) records data, archive, callback, reset,
  resource, and output identities.
- [replay_f2.py](replay_f2.py) is the replay CLI.
- [f2_grade_preflight.py](f2_grade_preflight.py) verifies exact predictions and
  sources without mapping label arrays.
- [grade_f2.py](grade_f2.py) applies unchanged `global_wp_clip2_v1` and the
  source-qualified connected-component bootstrap.

## Run sequence

The pinned offline image is
`wnn-python311-submission-check:latest` (Python 3.11.16, NumPy 2.2.6,
PyArrow 19.0.1). The replay command is invoked inside `scripts/limited` with
one CPU, a 1.5 GB outer limit, and the Docker limits in the recipe. Mount the
repository read-only at `/repo` and the physical scratch directory read-write
at `/work`; pass these CLI arguments to
`research/campaign_2026-10-18/online_features_r1/tl3_fixed_blend_f2_r1/replay_f2.py`:

```text
--repo-root /repo
--candidate tl3|r3
--zip /repo/<candidate ZIP>
--split /repo/research/campaign_2026-09-29/artifacts/splits/fold2.json
--scratch /work/tmp
--output-predictions /work/out/<candidate cache>.npy
--output-sidecar /work/receipts/<candidate sidecar>.json
--output-receipt /work/receipts/<candidate replay>.json
--reset-check
--resource-profile "outer 1 CPU/1.5 GB; Docker 1 CPU/3 GiB; no swap/network"
```

Add `--max-groups 8` for the profile. Full replay omits that flag. Run TL3 and
R3 sequentially; do not overlap their containers. After both full replay
receipts exist, run the label-blind preflight before the separate score phase.
The first score attempt read the first group's target and mask, then failed with
`KeyError: 'r3_vs_tl3'`; no result file was written. That failure is documented
in [initial_grade_failure.json](initial_grade_failure.json). The repaired code
allocates paired-statistic arrays by comparison key. A new label-blind
preflight then bound the amended grader and unchanged prediction artifacts
before the successful score. Both preflight snapshots are retained.

The predeclared interpretation threshold is a fixed blend gain of at least
0.0005 WP with a conditional component interval above zero. Even then, this
only warrants one more fixed confirmation with known model ancestry; it does
not make the blend a submission or justify a new large B300 direction. An
uncertain positive remains eligible for bounded confirmation, and a negative
result narrows only these exact coefficients and endpoints.

## Observed result

On 443 previously inspected F2 groups (144 connected components, 903,071
scored rows), TL3 scored 0.723887 WP, R3 0.715413, the fixed 75/25 mixture
0.722678, and the fixed 50/50 mixture 0.720950. Paired component-bootstrap
95% intervals for differences versus TL3 were [−0.001915, −0.000643] for
75/25, [−0.004429, −0.001736] for 50/50, and [−0.011844, −0.005776] for R3.
These intervals condition on the four fixed predictions; they do not cover
model-selection, fit-seed, ancestry, or public-transfer uncertainty. No
multiplicity adjustment was applied. F2 and F0 are exposed, TL3 training and
teacher ancestry are unknown, and R3 was selected on F0. This is not clean CV
or submission evidence. It repeats the qualitative F0 ordering for this exact
pair and weights; it does not reject other students, teachers, or conditional
mixture mechanisms.

No large new training direction emerged from this no-fit comparison, so it
does not warrant another B300 draft. Existing causal-state and predictive-state
handoffs remain [PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17)
and [PR #18](https://github.com/VladKochetov007/WunderFunder/pull/18).

## Receipts and artifacts

- [Admission](admission.json) and [profile gate](profile_gate_receipt.json) —
  frozen source checks, profile results, and full-replay admission.
- [R3 resource adjustment](r3_resource_adjustment.json) — records the cap
  change made after TL3 used 2.99 GB; the task reservation stayed below 5 GB.
- [Replay execution receipt](replay_execution_receipt.json) — actual full-run
  timings, resource peaks, hashes, grader attempts, and score provenance.
- [TL3 profile receipt](profile_tl3_replay.json) and
  [R3 profile receipt](profile_r3_replay.json); their sidecars are recorded
  alongside them.
- [TL3 full replay receipt](tl3_f2_replay_receipt.json) and
  [R3 full replay receipt](r3_f2_replay_receipt.json); their sidecars are
  recorded alongside them. Large prediction payloads stay in physical scratch.
- [Original label-blind preflight](preflight.json) and
  [initial scorer failure](initial_grade_failure.json) preserve the first
  attempt; [grade-only amended recipe](recipe_gradefix_r1.json) binds the
  repaired scorer without changing the scientific comparison.
- [Corrected label-blind preflight](preflight_gradefix_r1.json) and
  [successful grade](result_gradefix_r1.json) record the final fixed-endpoint
  comparison and conditional intervals.
- [Independent evidence review](../../../../unified/reviews/TL3_R3_F2_FIXED_BLEND_EVIDENCE_REVIEW_20261018_R1.md)
  confirms the reported result and records the missing preflight hashes for
  imported `gmetric.py` and `evaluation.py` as a reproducibility caveat.

If local or cross-fold evidence reveals a genuinely large mechanism, create a
separate draft PR into `research-team` with runnable code, a matched
seed/fold/teacher-student plan, full logging, resource profile, and the explicit
evidence limits. Existing substantial causal-state and predictive-state
directions are already represented by [B300 draft PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17)
and [B300 draft PR #18](https://github.com/VladKochetov007/WunderFunder/pull/18).
