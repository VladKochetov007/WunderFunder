# Grade full-sequence evaluation exports for A/B/C

This separate receiver script uses the official pooled `GlobalAccumulator` and
honors an external `WNN_PROJECT_ROOT`; the historical `scripts/score_run.py`
assumes its data live inside the source checkout. The split loader verifies source
and component identities. The predictor sidecar must match the exact evaluation
groups and their digest, fold SHA, run, epoch, checkpoint hash, prediction hash,
feature-only export scope and FP32 mode. The grader hashes and compares a
pre-pinned target cache, verifies unique target groups and complete 20,000-row
shapes, masks the first 99 warmup rows and computes the official pooled metric.
It writes an exclusive new receipt with all identities. The target cache digest
is necessary but does not by itself prove its labels came from the correct parquet;
independently verify cache/data provenance before freezing it. An old exposed fold
remains exposed. Do not run on an active worker before its corresponding fit
completes. Exact numerical and receiver compatibility checks are still required.

On the Linux receiver in the reviewed checkout, use one GPU and set the values to
the completed run, its original clean split, and the independently verified digest
of `"$WNN_PROJECT_ROOT/cache/valid_targets.npz"`:

```bash
cd "$WNN_REPO"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${WNN_PROJECT_ROOT:?}" "${RUN:?}" "${SPLIT_MANIFEST:?}" "${EPOCH:?}" "${TARGET_SHA256:?}"
# Model loading and GPU inference happen on the receiver, not on the local Mac.
python scripts/predict_run.py "$RUN" valid --epoch "$EPOCH" \
  --campaign-split-manifest "$SPLIT_MANIFEST" --campaign-evaluation-only \
  --device cuda --inference-precision fp32
python research/experiment_drafts/gold-frontier-hypotheses-20261102/legacy_grade/score_evaluation.py \
  --data-root "$WNN_PROJECT_ROOT" --split-manifest "$SPLIT_MANIFEST" \
  --run "$RUN" --epoch "$EPOCH" --target-cache-sha256 "$TARGET_SHA256" \
  --output "$WNN_PROJECT_ROOT/artifacts/runs/$RUN/evaluation-grade.json"
```

`predict_run.py` replaces predictions and sidecars with the same run name and
deletes `cache/seqsums_<run>.npz` if present. Reserve unique output names and
do not execute against an active or irreplaceable export.
A, B and C use their configured fixed zero-based epoch 5. Repeat
for each control and treatment with its own matching fold. Hash and retain both
prediction sidecar and grade receipt. This is not a public leaderboard result and
never authorizes a submission.
