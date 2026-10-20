# Parent-fit R2 terminal verifier V3

V2 reached the runtime contract check after the saved model and state, RNG, AdamW/scheduler, loader, counter, and history checks had passed. It stopped because its cuDNN deterministic expectation was opposite to the pinned source. `src/wnn/resources.py` configures `torch.backends.cudnn.deterministic = True` whenever CUDA is available, and the source manifest pins that file at `95fdc839ff1561622ac06c465cfc9a01370130d99be1e832b3a33c79bcb37930`. V3 changes only that runtime expectation to require the exact boolean `True`, verifies the manifest's source hash, and reports this source pin. The remaining post-contract provenance check reconciles checkpoint hashes emitted by the pinned `training/outputs.py` implementation.

The rendered packet targets `fg19_parent_h256x2_f0_s1001_fit_r2_v1`, unit `computer-vision-homework-fg19-parent-fit-r2-verify-v3.service`, and output `/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_parent_fit_r2_verify_v3`. It retains V2's successful CUDA RNG diagnostic report, capture acceptance, exact stop, native terminal evidence, source, split, cache, limits, and fresh-gate logic. V1 and V2 packets and failed units remain separate. No local preparation step reads model, training-state, dataset, or prediction-array payloads. Root owns all remote actions.

The focused V3 tests require exact `True` (rejecting false, integer aliases, `None`, and strings) and check the immutable source file bytes, manifest row, and AST assignments. The full cold import proof is [`parent_fit_terminal_cold_import_proof_v4.py`](../parent_fit_terminal_cold_import_proof_v4.py) with receipt [`PARENT_FIT_TERMINAL_COLD_IMPORT_V4.json`](../PARENT_FIT_TERMINAL_COLD_IMPORT_V4.json); it imports the V3 checker and source-qualified helpers, exercises the V3 bool guard, and records all 26 `wnn` runtime modules. The run reported no model/array/data reader calls and did not initialize CUDA.

Run from the repository root with the already-installed environment:

```sh
.venv/bin/ruff format research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_check_v3.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v3 research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_cold_import_proof_v4.py
.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_check_v3.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v3 research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_cold_import_proof_v4.py
.venv/bin/python -I -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v3/prepare_parent_fit_terminal_verifier_v3.py
.venv/bin/python -I -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_cold_import_proof_v4.py
.venv/bin/python -I -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v3/check_parent_fit_terminal_verifier_v3.py
```

The body records submit timeouts and post-submit query errors durably and never retries. It does not stop either failed verifier unit. Fresh remote acceptance, output, source, resource, and ownership gates remain root's responsibility immediately before any new submission.
