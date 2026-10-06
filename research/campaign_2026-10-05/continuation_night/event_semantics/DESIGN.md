# E492: fit-only raw event semantics

Hypothesis H120. Before execution, freeze the 24 fit components from the existing
target-mechanics cohort. Load one 20,000-row parquet group at a time, only selected
raw fields, with Arrow threads disabled. No models, fitting, teacher outputs,
evaluation-role rows or public-score estimates are involved.

Compare bid-only, ask-only, bid-OR-ask and bid-AND-ask change definitions, retaining
the production AND definition as historical behavior.
Bid and ask clocks each include rows where both coordinates changed; they are
individual-coordinate clocks, rather than exclusive single-coordinate categories.
Count target changes missed by each clock after the first99 warmup rows, and equality of a6-change with each
clock. Group class is label-derived diagnostic metadata, never inference input.
Report all eligible and selected-mask counts separately, exact raw target-change
semantics, clipped joint target support and absolute-target mass. A clock change
association is descriptive, not proof of conditional predictability or gain.

Save per-group numerical rows, aggregate tables, source/cohort/data identities
and PNG figures under eda/v1. Execution: one CPU, two GiB, no GPU, ten-minute
ceiling through the independently reviewed continuation runner. Independent raw
count reproduction precedes supported conclusions; no hypothesis is vetoed by
this diagnostic alone.
