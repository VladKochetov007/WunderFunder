# H218 fold-0 reference-blend grader

This is a code-only delivery for root review. It has not opened the H218 candidate prediction, computed scores, or written a grade report. The required root admission file is intentionally absent; without it, the grader exits before reading candidate/scoring data.

The preregistered candidate contains only 468 sorted valid-source groups. The own GRU reference and target cache contain the full 1,873-row valid source. The grader joins all three by `(valid, row_group)` from their sidecars/cache and the frozen split; it never treats equal integer row-group IDs from `train` and `valid` as the same observation. Candidate fit membership and component exclusion are checked with source-qualified keys. The reference GRU, its since-teacher ancestors, and propensity source are rechecked through the existing H217 reference-lineage helper.

It calls the frozen `campaign_score.mmap_stored_npz`, per-sequence centered moments, pooled scorer, and paired component bootstrap. Each mixture is formed from raw predictions and then passed to the official clipping/scoring implementation. It reports the five registered H218 weights, paired component intervals, and per-target prediction covariance/correlation. It has no winner-selection, all-data comparison, or public forecast path.

Root creates `H218_FOLD0_REFERENCE_BLEND_GRADE_ADMISSION_20261008_R1.json` only after the delivered fit/export artifacts are sealed and this code is independently reviewed. The JSON must use schema `h218-fold0-reference-blend-grade-admission-v1`, `status: "approved"`, `approved_by: "root"`, preregistration record ID `H218_FOLD0_REFERENCE_BLEND_PREREG_20261008_R1`, the exact preregistration SHA from the code, and the final grader file SHA. Its `candidate_artifacts` object must contain exactly these keys: `fit_terminal_receipt`, `export_terminal_receipt`, `checkpoint`, `provenance`, `prediction`, `sidecar`, and `source_lock`. Each artifact entry includes a repository-relative `path` and exact `sha256`. Root's fit/export receipt entries attest the associated top-level preregistered `fit_attempt_id`, normalized `exit_code: 0` and `terminal_status: "success"`, and exact output hashes. The export receipt entry also records the actual export `attempt_id`. The grader separately parses each referenced raw bandit attempt record: `status: "finished"`, `exit_status: 0`, `phase: "terminal"`, and exact attempt identity. The fit record identity is the top-level preregistered `fit_attempt_id`; the export record identity is root-attested because it is created only when the separately admitted export launches. These producer records do not use the root admission's normalized `terminal_status` field. The grader also verifies fixed epoch-3 provenance, exact source-lock closure, the actual H218 exporter sidecar layout, membership, shape, dtype, and finiteness before scoring.

After root approval, run under the local support ceiling:

```sh
PYTHONPATH=src:. WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/attention_training_H218/grade_fold0_reference_blends.py
```

The test module uses only small synthetic arrays and metadata fixtures. It covers row reordering, source-qualified aliases and component leakage, nonfinite predictions, blend-before-clip, pooled-score agreement with the official metric, paired-bootstrap output, refusal without root admission, the exact frozen preregistration's top-level fit ID, actual bandit terminal-record fields, and a fixture matching the H218 exporter's emitted sidecar keys. Tests do not access competition predictions or produce a score.
