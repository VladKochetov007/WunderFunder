# Zero-fit blend support audit

## Frozen contract

This is a bounded, 12-minute analytical audit. No fitting, GPU use, network access, upload, coefficient search, or mutation outside this folder is authorized. The concurrent local fit retains its two CPU cores; this audit is limited to one core (`WNN_CPU_QUOTA=100%`) and 3 GiB memory. Python analysis runs through `scripts/limited` with OMP/BLAS thread counts set to one.

Primary outcome is the unchanged official `global_wp_clip2_v1` on all scored rows of the full official valid set. Evaluation units are connected components from `research/campaign_2026-10-29/teacher_track_r1/chronological_primary.json` (`sequences.valid`, mapping row group to component). The full valid set is exposed and repeatedly used for development; all estimates are discovery evidence conditional on prior selection.

Frozen candidate outputs, aligned by source row-group identity:

1. Equal 50/50 `l14_kd50_s3` + `l14_kd50_s4` pair.
2. `l14_kd90_s3` single.
3. Locally reproduced, train-only `tt29rep_fbc1k_s5` teacher single.
4. Equal 50/50 `l14_kd90_s3` + `tt29rep_fbc1k_s5`.
5. Historical native INT8 TL3C blend (`blend-int8-full-v3/predictions.npy`).

Weights are fixed at 0.5 wherever stated. The optional 50/50 KD90 + TL3C output blend is excluded before scoring: the existing TL3C full-valid native report records a 4,436.4-second validation loop, already above the competition's 3,600-second whole-test-set limit. Adding another recurrent model has no demonstrated runtime margin, so this candidate is not considered serveable on current evidence. No package or runtime experiment is part of this audit.

The paired contrasts are every candidate versus the fixed KD50 pair, plus the KD90+teacher mean versus each member and versus TL3C. No validation-selected weights or other post-hoc candidates are allowed. Uncertainty uses 2,000 fixed-seed paired component bootstrap resamples (seed 20261114), resampling the same component draws for both sides of each contrast; intervals are percentile 95% intervals. Component influence is leave-one-component-out change in each paired delta, with the ten largest positive and negative changes reported. Per-component raw sufficient statistics are retained in the result JSON.

Before bootstrap, each complete endpoint's official score is checked through `local_r1/grade_submissions.py` against its independent `reference_global_wp` implementation. Canonical `wp_sums` are also accumulated by component and recombined with `wp_from_sums`; the aggregate must agree with the reference score within 1e-8. A failed identity, coverage, finite-value, or metric-parity check aborts the audit without bootstrap output.

These intervals quantify evaluation-component resampling conditional on fixed fitted predictions. They do not measure fitting-seed variability, repair the exposed validation set, establish unseen-CV significance, or resolve transfer uncertainty. The train-only candidates follow the frozen same-count split recipes; TL3C's original split hash differs from the local reproduction, so byte-level training-membership equality is not claimed. The source artifacts and their hashes are recorded in `RESULTS_R1.json`.

## Artifacts

- [`audit_blends.py`](audit_blends.py) maps prediction rows, checks identities and metric parity, aggregates component sums, and computes the frozen paired bootstrap.
- [`RESULTS_R1.json`](RESULTS_R1.json) contains hashes, endpoint scores, paired deltas and intervals, component influence, per-component sufficient statistics, and limitations.

The unchanged official metric implementation is [`src/wnn/gmetric.py`](../../../../src/wnn/gmetric.py). The existing row-group mapping and independent metric-reference check are reused from [`grade_submissions.py`](../../local_r1/grade_submissions.py). No root or unified index was edited; root integration remains separate.

## Results

All five endpoint arrays passed complete row-group/layout checks on 1,873 sequences and 3,528,273 scored rows. Official scores and `reference_global_wp` agree within 8e-16; component-aggregated `wp_sums`/`wp_from_sums` agree within 5e-16 before bootstrap. The split map yields 579 connected components.

| Fixed endpoint | Full-valid WP |
|---|---:|
| TL3C native INT8 blend | 0.692335900 |
| Equal KD90 + fbc1k teacher | 0.691897947 |
| KD90 s3 | 0.691442250 |
| Equal KD50 s3/s4 pair | 0.691422906 |
| fbc1k teacher s5 | 0.691420480 |

| Paired contrast | Delta WP | 95% component-bootstrap interval |
|---|---:|---:|
| KD90 − KD50 pair | +0.0000193 | [−0.0006391, +0.0006667] |
| fbc1k − KD50 pair | −0.0000024 | [−0.0008658, +0.0007903] |
| KD90 + fbc1k − KD50 pair | +0.0004750 | [−0.0002157, +0.0011607] |
| TL3C − KD50 pair | +0.0009130 | [+0.0002236, +0.0015877] |
| KD90 + fbc1k − KD90 | +0.0004557 | [+0.0001890, +0.0007244] |
| KD90 + fbc1k − fbc1k | +0.0004775 | [+0.0001998, +0.0007493] |
| KD90 + fbc1k − TL3C | −0.0004380 | [−0.0010616, +0.0002343] |

On this exposed panel, the fixed KD90/teacher mean improves over either member under component resampling, but remains 0.000438 below TL3C at the point estimate. TL3C is +0.000913 over the KD50 pair, with a positive component interval. KD90 and the teacher alone are effectively tied with the pair at this scale. These are conditional fixed-prediction comparisons; the intervals omit seed variation and do not establish unseen-fold performance.

Leave-one-component-out influence is nonzero: component 2957 (12 row groups, 49,665 scored rows) shifts the KD90/teacher-versus-pair delta by −0.0001347 when omitted, and the TL3C-versus-pair delta by −0.0001237. Its presence therefore supports both blends against the pair on this panel. Full positive and negative influence lists and all component sufficient statistics are in `RESULTS_R1.json`.
