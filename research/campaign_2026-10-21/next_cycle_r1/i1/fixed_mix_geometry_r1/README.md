# Fixed F0 mixture geometry adapter

This code-only packet prepares a posthoc diagnostic of the already-frozen F0/seed1001 i1 A/B/C predictions. It compares B with the fixed `0.9*B + 0.1*C` endpoint and reports the same fixed `0.9*B + 0.1*A` mixture as a control. It does not search coefficients or change the panel grader. The underlying F0 grade is exposed development-fold evidence and remains conditional on those fits.

The adapter reuses the existing A/B/C validator and weighted centered-moment helper. Before it opens target arrays, it checks the frozen A/B/C result SHA, all pinned grader/scorer/metric sources, the F0 split and 10% fit-subset SHAs, shared initialization and normalization identities, each fit's provenance, and each prediction's sidecar, FP32 NPY header, payload length, full group order, and exact coverage. It compares returned identities with the frozen F0 grade. Its memory gate includes all three prediction files and the established 1.4 GB ceiling. The predictions remain read-only memory maps; the script processes one sequence at a time and rehashes bound inputs after scoring.

For each target and the whole scored region plus four fixed position bins, the output separates raw predictions from clipped predictions. It reports weighted covariance and variance with the official `abs(clip(y, -2, 2))` weights, official scores and their reconstruction from clipped moments, raw-output mean and standard deviation, and scored-row clipping fractions. For each fixed mixture it defines `D=M-B`, reports `Cov(y,D)`, `Cov(B,D)`, and the residual covariance after linearly projecting both target and mixture difference on B. It also checks `Cov(y,M)=Cov(y,B)+Cov(y,D)` and `Var(M)=Var(B)+2Cov(B,D)+Var(D)` numerically. Mixture clipping is applied after the fixed blend, matching the official scorer.

The report keeps empty rows or zero-weight targets explicit. It computes no bootstrap intervals because this is a descriptive decomposition on already graded endpoints, not a new independent evaluation. F2 is excluded until the completed panel grade certifies all three full-fold exports. No predictions, target labels, model tensors, or tests were loaded or executed while preparing this packet.

After source review and admission by the root operator, the prepared F0 command is:

```sh
scripts/limited \
  /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-21/next_cycle_r1/i1/fixed_mix_geometry_r1/fixed_mix_geometry.py \
  --output /home/vlad/development/wunder/research_tmp/i1-fixed-mix-geometry-r1/F0_FIXED_MIX_GEOMETRY_R1.json
```

The command writes a create-only result under physical scratch. This packet itself is preparation only; execution remains with the root operator.
