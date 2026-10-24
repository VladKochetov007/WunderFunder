# Clock redundancy EDA

Both root-owned scans are complete. The [findings](FINDINGS_R1.md) distinguish scored-row overlap from all-row clock identity and the current seven-group representation choice. There are 65 distinct sampled event-pattern classes; grouping changes 14.392% of scored field–row ages for seven groups versus 11.533% for nine families. These are descriptive results, not model gains.

- [Aggregate result](CLOCK_STATISTICS_R1.json), [execution receipt](EXECUTION_R1.json) and [log](EXECUTION_R1.log): 96 sequences, 192,763 scored rows, 10.914 seconds, no targets or fits. Censoring counts independently agree with the notebook.
- [Geometry plan](CLOCK_GEOMETRY_PLAN_R1.json), [source](clock_geometry.py), [result](CLOCK_GEOMETRY_RESULT_R1.json), [execution](GEOMETRY_EXECUTION_R1.json) and [log](GEOMETRY_EXECUTION_R1.log): same sample, 6.974 seconds, all-row event hashes and seven/nine-group age differences. The formatting amendment is explicit and precedes data reads; Ruff checks pass.

This folder contains a frozen no-target plan and a small streaming script for
the next EDA question: how much the 112 causal field-change clocks overlap,
and how activity and observed age vary over sequence position. It reuses the
same 96 valid row groups from the parent [EDA_PLAN_R2.json](../../eda/EDA_PLAN_R2.json)
and the definitions in the existing live marimo notebook
[research_frontier.py](../../eda/research_frontier.py), which was read but not
edited. The parent sample plan is SHA-pinned. The live notebook is mutable UI;
its reviewed digest is recorded for context, while execution uses the frozen
field order, event definition, and statistics in this plan and script. A later
notebook UI edit does not invalidate this separate no-target analysis. The
plan and script define the executable contract:
[CLOCK_EDA_PLAN_R1.json](CLOCK_EDA_PLAN_R1.json).

`clock_eda.py` reads only the 112 feature fields, `seq_ix`, `need_prediction`
and `is_scored` from `datasets/valid.parquet`; it never requests targets,
labels or prediction columns. It holds one 20,000-row sequence at a time. A
change is exact adjacent-row float32 inequality, with row zero excluded, as in
the notebook. The eligibility denominator is `need_prediction AND is_scored`.
Age is computed from every row in the sequence, stays start-censored until
the first change, and is summarized on eligible rows.

The frozen output contains:

- A 112×112 matrix of cochange counts and pairwise Jaccard, plus a 9×9 upper
  triangle of mean pair Jaccard for `i0_p`, `i0_v`, `i0_dp`, `i0_dv`, `i1_p`,
  `i1_v`, `i1_dp`, `i1_dv`, and `a`. Feature order and group membership are
  included for a grouped heatmap.
- Per-position eligible-row counts, per-feature update/censored/age-threshold
  counts and rates, and family-aggregated update and age rates. These support
  activity and age versus position plots.
- One record per row group/sequence with eligible rows and update-event counts
  by family.

Root owns the raw-data run. The intended command is bounded to one CPU core,
400,000,000 bytes, zero swap and 120 seconds:

```bash
WNN_MEM_MAX=400000000 WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_RUNTIME_MAX=120 \
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 POLARS_MAX_THREADS=1 ARROW_NUM_THREADS=1 \
scripts/limited .venv/bin/python \
research/campaign_2026-10-24/resume4h_r1/feasibility_scout/clock_eda/clock_eda.py \
--plan research/campaign_2026-10-24/resume4h_r1/feasibility_scout/clock_eda/CLOCK_EDA_PLAN_R1.json \
--output /home/vlad/development/wunder/research_tmp/resume4h-r1/clock-eda-r1.json
```

[Four synthetic invariants](test_clock_eda.py) pass under the same 400 MB/one
core bound, and Ruff format/check pass. Root subsequently completed the raw-feature runs and [notebook plots](../../eda/README.md); neither clock scan read predictions, targets or label values.
