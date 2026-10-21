# Local checks

These checks cover only the conditional AllThree verifier template and renderer. They do not import or execute the rendered support body and do not access B.

- From the repository root, `./.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_verifier_v1/REMOTE_SUBMISSION_TEMPLATE.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_verifier_v1/render_child_allthree_terminal_verifier_v1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_verifier_v1/test_render_child_allthree_terminal_verifier_v1.py` — passed.
- From the repository root, `./.venv/bin/ruff format --check` on those same three files — passed.
- From this folder, `../../../../../.venv/bin/python -B -m unittest -v test_render_child_allthree_terminal_verifier_v1.py` — 15 focused synthetic tests passed. They cover typed/raw terminal mismatches, strict integer/timestamp validation, exact import-proof fields including the empty preloaded-module set, launch-producer/result-capture identity fields, payload-pin and stdin self-read guards, Path arguments, and preflight/GPU/output-mutation ordering.
- AST comparison against `child_nll_terminal_verifier_v2/REMOTE_SUBMISSION_V2.py` — all six copied GPU-ownership functions were identical after parsing: `_bounded_gpu_query`, `_memory_bytes`, `_bounded_proc_read`, `_proc_starttime`, `_gpu_process_owner`, and `check_selected_gpu_ownership`.
- Running the renderer without `--render` reported `WAITING_FOR_ROOT_PREPARATION_AND_NATIVE_EXIT0_PROOF`, with `body_rendered=false` and `remote_or_B_action=false`. The root preparation, typed native terminal record, and matching raw capture were absent at check time.

No source, model, state, data, or cache payload was loaded. The focused test command used `-B`; no bytecode cache was written. The actual body-render path remains uncalled.
