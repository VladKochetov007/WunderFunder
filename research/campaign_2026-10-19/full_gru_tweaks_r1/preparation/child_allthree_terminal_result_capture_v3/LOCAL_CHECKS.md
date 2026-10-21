# V3 local source checks

The focused AST contract fixture passed:

```sh
./.venv/bin/python -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_result_capture_v3/test_result_capture_v3_contract.py
```

It parses the frozen V2 template, V3 candidate and V5 producer source only. It imports no production module and makes no B/SSH request. It confirms the normalized V2/V3 AST delta, optional absence behavior, print-only checker result line, and stdout/stderr append to `unit.log`.

Formatting check passed for the V3 template and fixture:

```sh
./.venv/bin/ruff format --check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_result_capture_v3/RESULT_CAPTURE_TEMPLATE.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_result_capture_v3/test_result_capture_v3_contract.py
```

The focused Ruff check passed with the inherited `UP017` style rule ignored:

```sh
./.venv/bin/ruff check --ignore UP017 research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_result_capture_v3/RESULT_CAPTURE_TEMPLATE.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_result_capture_v3/test_result_capture_v3_contract.py
```

An unfiltered check reports only `UP017` for the unchanged `dt.timezone.utc` expression copied from V2; the frozen V2 template has the same finding. It was left unchanged to keep this source delta scoped to optional `VERIFICATION.json` handling. No `__pycache__` or `.pyc` was created by the `-B` proof run.

No production source was imported or executed. The V3 reader's actual receipt and invocation pins remain unbound. No support unit, B path, model/state/data/cache payload, or network endpoint was accessed by these local checks.
