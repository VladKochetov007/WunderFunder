# V5 local checks

Only the new AST-only namespace fixture and the source-delta proof were run.
No V4 tests or other unchanged suites were replayed.

| Check | Result |
|---|---|
| `python -B -m unittest -v test_score_result_capture_v5_namespace.py` | 3 focused tests passed |
| `ruff check SCORE_RESULT_CAPTURE_TEMPLATE_V5.py test_score_result_capture_v5_namespace.py prove_score_result_capture_v5_rebase.py` | passed |
| `ruff format --check SCORE_RESULT_CAPTURE_TEMPLATE_V5.py test_score_result_capture_v5_namespace.py prove_score_result_capture_v5_rebase.py` | passed |
| `python -B prove_score_result_capture_v5_rebase.py` | `PASS_EXACT_V4_UNIT_D2_CORE_REBASE` |

The proof pins the V4 base reader at 26,354 bytes / SHA-256
`88968f6dd73d4b489c4fc57308906c5e390137a49389ab991408408f353b9be6` and the
V6 scorer core at 26,965 bytes / SHA-256
`961f566eaeec0cd7e9ab48788666d03da9b0482810a11f16a019b083cfc65005`. The
reader source differs only in `SCORE_D`, `SCORE_UNIT`, and `SCORER_CODE_PINS`;
the core value changes while the remaining three scorer code pins stay fixed.

The fixture and proof read source text only. No score receipts or results,
systemd/NVIDIA state, B paths, models, predictions, data, caches, or arrays were
read by these checks.
