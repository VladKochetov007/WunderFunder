# Parent-fit RNG diagnostic packet

This is a separately prepared CPU diagnostic for the saved parent `fg19_parent_h256x2_f0_s1001_fit_r2_v1`; independent review and root admission are pending. It is bound to the successful parent native invocation `5186b3016ff54b1fa754c10a1f87b3e4`, source commit `68ff682977b822f38351aac0f621146f5f13b440` with 167 files, the accepted/stopped cache metadata, and the exact failed checker V1 capture `7761927b6bcacd034af294ee5f37c30561b6ff3b07b846f0e7393324fea1a359`. It also binds the exact `training_state.pt` stat captured before the failed verifier; fresh remote stat gates must still match before launch and during the restricted load.

The remote unit `computer-vision-homework-fg19-parent-rng-diag-v1.service` writes only `RNG_DIAGNOSTIC.json` under `research_tmp/fg19_parent_rng_diag_v1`. It stages the exact f270 checker and its two pinned helper files, then imports `_load_checkpoint` from that source and uses its 64 MB `weights_only=True` restricted loader and NumPy/TorchVersion safe-global allowlist. The diagnostic reports the state format/next epoch, installed Torch version, RNG keys and structure/cardinality, plus only dtype, shape, numel, and SHA-256 for CPU/CUDA RNG tensors. It does not inspect or emit model or optimizer values, dataset/cache arrays, a score, terminal verification, or acceptance.

The admission body reuses the reviewed fresh source, split, registered-data stat, cache metadata/output-stat, parent terminal, populated-descendant, memory headroom, physical-scratch, and 25 GiB storage-floor gates. It checks only the new unit and output for absence; it does not query, stop, restart, or retry failed verifier V1. All gates precede create-only staging and the single 0.5-CPU / 1.4 GB high / 1.5 GB max / zero-swap / 600-second unit submission. Root owns fresh preflight, review, and any launch.

`prepare_parent_rng_diagnostic_v1.py` renders the body from the frozen reviewed gate-template SHA in `PREPARATION.json` and captured metadata. `check_parent_rng_diagnostic_v1.py` performs static checks and a cold isolated read sentinel that denies `.pt`, `.npy`, `.npz`, and `.parquet` opens. Run the local checks from the repository root:

```sh
.venv/bin/ruff format research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_rng_diagnostic_v1/prepare_parent_rng_diagnostic_v1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_rng_diagnostic_v1/rng_diagnostic_v1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_rng_diagnostic_v1/check_parent_rng_diagnostic_v1.py
.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_rng_diagnostic_v1/prepare_parent_rng_diagnostic_v1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_rng_diagnostic_v1/rng_diagnostic_v1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_rng_diagnostic_v1/REMOTE_SUBMISSION_TEMPLATE.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_rng_diagnostic_v1/REMOTE_SUBMISSION.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_rng_diagnostic_v1/check_parent_rng_diagnostic_v1.py
.venv/bin/python -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_rng_diagnostic_v1/check_parent_rng_diagnostic_v1.py
```

No checkpoint or array data is loaded during local preparation or checks.
