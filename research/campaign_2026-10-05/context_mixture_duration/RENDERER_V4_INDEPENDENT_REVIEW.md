# E488 renderer v4 source review

The currently present renderer v4 fixes the v3 plotting defect: it draws percentile intervals as horizontal segments and point estimates as separate scatter markers, so a percentile interval may exclude its point without producing a negative Matplotlib error-bar length. The synthetic test explicitly checks this case, and the renderer continues to emit all 24 arm rows, 27 global contrast rows, and 54 target contrast rows. It reads score/receipt JSON and hashes the export file only; it does not load the NPZ contents, Parquet rows, or checkpoints.

Current file hashes:

- `render_report_v4.py`: `66a446aec01ab4f0207e350f3769d445bcbac51bec6de2a4b963e4a05d2fce03`
- `test_render_report_v4.py`: `05fb7c5d5277b437868515eabeba4f11b86dee96b95bbdd0d7c5d810f84d2582`

Under `scripts/limited` with 1 CPU and 2 GiB, all 5 synthetic tests passed; Ruff and `py_compile` passed. The direct CI-outside-point test passed for the global plot, and the synthetic output test exercises both global and target plot generation.

## Integration identity

The initially relayed v4 hashes were provisional; the reviewed frozen files are the current hashes above. Root clarified that the renderer consumes `checks/context_duration_export_reproduction.json`, which uses the exact schema this version requires (`all_raw_export_scores_and_draw_summaries_match` and `score_sha256`). The separate wrapper receipt with `all_checked_values_match` is not a renderer input. This resolves the integration mismatch without weakening receipt binding. No actual result/export inputs were read in this review.

**Disposition: GO for rendering the already-graded report with these exact frozen v4 bytes and the intended raw arithmetic reproduction receipt.**
