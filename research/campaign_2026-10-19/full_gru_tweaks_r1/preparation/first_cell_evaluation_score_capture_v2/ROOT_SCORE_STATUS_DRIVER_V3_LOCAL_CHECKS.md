# ROOT status driver V3 local checks

Only the new source-only V3 fixture was run.

| Check | Result |
|---|---|
| `python -B -m unittest -v test_root_score_status_driver_v3_rebase.py` | 3 focused tests passed |
| `ruff check ROOT_CAPTURE_SCORE_STATUS_V3.py test_root_score_status_driver_v3_rebase.py prove_root_score_status_driver_v3_rebase.py` | passed |
| `ruff format --check ROOT_CAPTURE_SCORE_STATUS_V3.py test_root_score_status_driver_v3_rebase.py prove_root_score_status_driver_v3_rebase.py` | passed |
| `python -B prove_root_score_status_driver_v3_rebase.py` | `LOCAL_V3_REBASED_TO_REVIEWED_SCORE_CAPTURE_V5_NOT_EXECUTED` |

The proof verifies the exact reviewed V5 template/review pins and the four
assignment-only delta from the frozen V2 driver. It also confirms the five
future score receipt literals are still unbound. No systemd/SSH calls, B
queries, receipt/result reads, or payload access occurred.
