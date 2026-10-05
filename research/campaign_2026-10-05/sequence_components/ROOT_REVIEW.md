# Root admission review, E467

The independent implementation review came from a different Luna agent than
the component author. Root inspected the actual code, CPU receipts and GPU
probe and made the following corrections before scientific fitting:

- Attention now has GRU and SSM static eight-dimensional branch controls.
  Each adds 787 branch parameters versus attention's 793, with the same
  widened 40-dimensional prediction head. This supports a memory-versus-static
  comparison; it still does not separate content retrieval from smoothing.
- Seven spline knots now span [-1,1], the guaranteed bounds of the input
  GRU/diagonal/attention states. Outer knots are no longer unreachable by
  construction. Occupancy under actual fitting is a result to report.
- The model count is ten, preserving all original candidates and adding
  controls. Both hybrid head variants are tested, so a hybrid score can be
  compared with its immediate head and memory alternatives.
- Role/component/group uniqueness, full 20,000-row shape, finite targets,
  frozen cohort hash and actual parameter counts are recorded or checked.
- The scalar scorer calls the existing centered-moment implementation;
  the official metric and old guarded runner were not changed.
- The training loss's per-chunk mass normalization and carried-state TBPTT
  mismatch are explicitly disclosed. They are shared training choices, not
  fixed-weight full-sequence gradient claims. Six sequence-level accumulated
  updates would change this screen's learning budget, so that alternative
  requires a separately budgeted comparison rather than an undeclared change.

CPU tests verified streaming reset/parity, prefix causality, exact zero spline
residuals and finite gradients across 20,000 rows. The actual B4/chunk256 GPU
probe includes an optimizer update and subsequent carried-state prediction.
The first eight-arm GPU probe exited 0, teardown verified, peak observed device
memory 381,681,664 bytes, peak worker RAM 1,613,172,736 bytes, 4.26 s charged,
zero worker or parent OOM increments. Its original source is archived next to
the probe result. The corrected ten-arm v2 probe must finish successfully before
the scientific spec is admitted; the v2 receipt is the authoritative source.

Authorize only the ten-arm fixed small-cohort experiment with the declared
single-worker 3600-second ceiling and existing campaign budget. No additional
seeds, full-data fits, model selection or competition upload are implicit.
Keep partial results and failures. No score result follows from source tests.
