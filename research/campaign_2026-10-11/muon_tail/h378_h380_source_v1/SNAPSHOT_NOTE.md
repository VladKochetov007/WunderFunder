h230_source_v4_tailweight = h230_source_v3_lineard2z plus opt-in distillation row-weight options
`--label-weight-power p` (w = |clip(y)|**p) and `--label-weight-floor f` (w = max(|clip(y)|, f)**p).
Defaults (p=1, f=0) take the unchanged v3/v2 code path (weights |clip(y)|, matching the metric).
Only the cached-distillation loss branch is affected. Created 2026-10-09 by root (Claude Code).
