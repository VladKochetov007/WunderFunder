# Independent review: fixed two-teacher diagnostic, R1

## Disposition

The narrow claim is supported as a **single exposed-valid teacher-target diagnostic**: the fixed equal mixture of `q14t_fbc_s5` and `q14t_tlead_s5` scores 0.6922604481 on the recorded full-valid panel, versus 0.6914204799 for `tt29rep_fbc1k_s5`, a difference of +0.0008399682. Both target components improve (+0.0009050715 for t0 and +0.0007748650 for t1). This does not establish a student gain, a new-seed recipe effect, or public transfer.

This review audited the grader, bindings, metadata handling, and recorded arithmetic. I did **not** load and independently rescore the prediction arrays. The grade implementation itself calls the official metric and its independent reference formula, but I did not rerun that comparison here.

## Audit findings

- **Fixed mixture:** [`TWO_TEACHER_PROBE_R1.json`](TWO_TEACHER_PROBE_R1.json) records the two coefficients as 0.5/0.5. In [`grade_teacher.py`](grade_teacher.py), `--available-pair` selects those same weights, loads only those two teachers plus the declared baseline, mixes predictions before scoring, and writes a separate two-teacher grade. The grade receipt repeats the coefficients. The arithmetic in [`TWO_TEACHER_GRADE_R1.json`](TWO_TEACHER_GRADE_R1.json) is internally consistent: the mean of its two pool target scores is the pooled score, and subtracting the baseline gives the recorded delta.
- **Metric and row alignment:** the grader pins the target-cache digest, checks 1,873 evaluation groups and 3,528,273 selected rows, and obtains predictions through the existing helper that maps source row-group IDs to prediction rows before applying the target mask. `score()` uses `global_wp` and checks it against `reference_global_wp`; the official implementation clips targets and predictions at ±2, weights by absolute clipped target magnitude, pools selected rows, then averages the two target correlations. The blend is scored after mixing, rather than by averaging member scores.
- **Bindings:** for each member and the baseline, the grader checks the prediction file digest against its sidecar, the checkpoint digest against both the checkpoint bytes and provenance, and the provenance-file digest. It also checks the fixed split manifest, evaluation membership, training source, and fit-group membership. The output retains prediction, checkpoint, and provenance hashes for all three inputs. This is sufficient to identify the compared model and prediction artifacts within the recorded setup.
- **Baseline scope repair:** the grader now expects `evaluation` for `tt29rep_fbc1k_s5` and `full-valid-feature-only` for the two imported teachers. This matches the inspected sidecars. The scope exception does not relax the membership checks: each sidecar must still identify the same fixed evaluation groups. I found no remaining scope mismatch in this comparison.

## Limits and gate status

The valid panel is repeatedly exposed, and this is one fixed mixture comparison. The result supports the observed score difference on that panel, not a generalization estimate. The grade JSON also omits the target-cache digest and grader/scorer source identity, although the grader checks the target-cache digest before scoring; including those identities in future receipts would make the result more self-contained.

The partial S6 export remains ungraded. `diagnostic_delta_at_least_0005` is true, while `target_quality_trigger` is false because this was explicitly the available-pair path. The separate frozen three-teacher gate is therefore **not unlocked**, and this review makes no student or public-score claim.

Reviewed inputs: [`TWO_TEACHER_PROBE_R1.json`](TWO_TEACHER_PROBE_R1.json), [`TWO_TEACHER_GRADE_R1.json`](TWO_TEACHER_GRADE_R1.json), [`grade_teacher.py`](grade_teacher.py), [`IMPORT_R1.json`](IMPORT_R1.json), and [`INDEPENDENT_REVIEW_R1.md`](INDEPENDENT_REVIEW_R1.md). The import ancestry and tensor-hash audit from that review were not repeated here.
