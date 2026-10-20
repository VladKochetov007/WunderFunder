# Parent-fit R2 terminal verifier admission

This packet prepares one root-reviewed, CPU-only integrity-checker admission for the already captured parent-fit native exit. It is a pre-acceptance verifier: the pinned checker must validate the actual parent checkpoint and continuation state before root issues acceptance and exact-stop records. No local preparation step launches or queries a remote unit.

The body binds parent run `fg19_parent_h256x2_f0_s1001_fit_r2_v1`, native unit `computer-vision-homework-fg19-parent-fit-r2-v1.service`, invocation `5186b3016ff54b1fa754c10a1f87b3e4`, source commit `68ff682977b822f38351aac0f621146f5f13b440` with 167 files, and the accepted/stopped fit-only cache. Its embedded inputs include the typed native terminal evidence and its bounded raw capture, the parent admission/launch records, source manifest, checker and two reviewed helpers, and cache acceptance/stop metadata.

The remote adapter stages only those pinned Python files and bounded JSON metadata after fresh identity/resource checks. It does not read or copy model, optimizer, dataset, or prediction-array payloads. The checker itself runs under a 0.5-CPU quota, 1.4 GB `MemoryHigh`, 1.5 GB `MemoryMax`, zero swap, 600-second runtime, hidden CUDA, and one BLAS thread. Fresh checks include the exact source tree, registered data and split identities, cache metadata and array stats, empty populated parent descendants, the 17 GB parent-headroom target with reserve, physical scratch, and at least 25 GiB free on project storage. The checker runs once and the body durably records submission and post-submit query outcomes; it never retries.

The parent-profile ancestry is transitively bound by the exact parent-fit admission bytes captured in the native records. This packet does not repeat the profile check. The 29,840-update parent endpoint is checked by the pinned terminal checker, not inferred by this preparation receipt.

`prepare_parent_fit_terminal_verifier_v1.py` renders `REMOTE_SUBMISSION.py` from the frozen `REMOTE_SUBMISSION_TEMPLATE.py` and source-qualified local inputs. `check_parent_fit_terminal_verifier_v1.py` verifies the local pins, syntax, one-submit/no-stop structure, durable post-submit error recording, and a cold metadata-only child with `.pt`, `.npy`, `.npz`, and `.parquet` reads denied. `LOCAL_CHECKS.json` records that local-only receipt. The checks used these commands from the repository root:

```sh
.venv/bin/ruff format research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v1/REMOTE_SUBMISSION_TEMPLATE.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v1/prepare_parent_fit_terminal_verifier_v1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v1/check_parent_fit_terminal_verifier_v1.py
.venv/bin/python research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v1/prepare_parent_fit_terminal_verifier_v1.py
.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v1/REMOTE_SUBMISSION_TEMPLATE.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v1/prepare_parent_fit_terminal_verifier_v1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v1/check_parent_fit_terminal_verifier_v1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v1/REMOTE_SUBMISSION.py
.venv/bin/python research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v1/check_parent_fit_terminal_verifier_v1.py
```

No unchanged model/source tests are rerun.

The current files are prepared for independent source review. Root owns fresh remote preflight, any submission, terminal acceptance, and exact stop.
