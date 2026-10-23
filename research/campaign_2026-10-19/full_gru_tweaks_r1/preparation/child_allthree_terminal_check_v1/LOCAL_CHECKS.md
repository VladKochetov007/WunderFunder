# Local checks

The new metadata fixtures passed (6 tests), Ruff reported “All checks passed!”, and its formatter reformatted the checker file before the final test/lint run. Those actions were limited to this packet:

```sh
.venv/bin/ruff format research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_check_v1
.venv/bin/python -m unittest discover -s research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_check_v1 -p 'test_*.py'
.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_check_v1
```

These fixtures exercise only tiny Python metadata values and strict native receipt boundaries. They do not load a model, state, source payload, representation artifact, or real terminal receipt, and they do not invoke `verify_terminal`. No unchanged suite is part of this packet's validation.
