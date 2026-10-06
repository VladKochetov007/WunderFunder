# Fixed uncertainty calculation, not yet executed on study data

Both contrasts are frozen: candidate4−reference4 and candidate4−reference3. Use 2,000 resamples, NumPy PCG64 seed0, paired whole chains within outer folds0/1; the same draws feed both contrasts. Each sampled chain contributes all its sequences' six sums, then the existing frozen pooled WP is recomputed. Report percentile95% intervals overall and separately for each fold. Do not average per-sequence correlations or fold correlations.

The evaluator's `sufficient.npz` contains no class field. No extra metadata is read: chains retain their internal sequence/class composition, while class shares can vary between resamples within a fold. These are pointwise intervals conditional on the fitted models and observed pure chains, not simultaneous intervals, adaptive-selection corrections, seed uncertainty or hidden-domain guarantees. No new gate or selection rule is introduced.

After root authorizes the actual readout and supplies the verified sufficient-file hash, run from the creative checkout:

```sh
timeout --signal=TERM --kill-after=5s 30s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python research/iterations/2026-10-01-continuation/student/bootstrap_student.py --input research/iterations/2026-10-01-continuation/student/evaluation-01/sufficient.npz --input-sha256 ROOT_VERIFIED_SUFFICIENT_SHA256 --output research/iterations/2026-10-01-continuation/student/evaluation-01/chain-bootstrap.json
```

`ROOT_VERIFIED_SUFFICIENT_SHA256` is an intentionally unresolved binding, not a guessed identity. Output must not already exist. No actual statistics, predictions or targets were opened in preparation. Generated smoke used1,200 sequences in400 unequal-length chains; a literal row-list oracle agreed within1.1e−15, and identical recipes yielded exact zero intervals. The full2,000-resample calculation took0.17s locally; <30s on one server CPU is a conservative estimate, not a measured server promise.
