# V5 local checks

The body `REMOTE_SUBMISSION_V5.py` is 710,848 bytes, SHA-256 `4d555fd69954d94415c135e467eadd081aa6fd915686197e5c7e628e935b9374`. `V5_BINDING.json` records the exact V4 candidate/rejection pins, the 22-byte insertion and its byte offset, a reverse reconstruction to the full V4 body, and all 15 unchanged payload pins.

Run only the new static contract fixture with the repository interpreter:

```sh
./.venv/bin/python -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_verifier_v5/test_child_allthree_terminal_verifier_v5_contract.py
```

The fixture passes 5 tests. It checks source AST and synthetic metadata only; neither body nor checker is imported or executed. The only response reply tested is the already-defined local durable-receipt guard.

Run Ruff against the new fixture only:

```sh
./.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_verifier_v5/test_child_allthree_terminal_verifier_v5_contract.py
```

No V4 tests were rerun. No B/network query, source restaging, support submission, model/state/data/cache payload read, or checker execution occurred. V5 remains unreviewed and unstaged.
