# OOD shared-cache retry packet R2

R2 repairs the source-selection and disk-reservation defects that blocked R1 admission. It preserves R1's four fixed OOD arms and homogeneous target recipe; it does not assert that any fit or cache build has run.

The launch contract is in [`OOD_R2_BUNDLE_MANIFEST.json`](OOD_R2_BUNDLE_MANIFEST.json), and the four concrete commands are argv arrays in [`OOD_R2_RETRY_JOBS.json`](OOD_R2_RETRY_JOBS.json). Run them **serially**. Each outer command starts with GNU `timeout --signal=TERM --kill-after=10s 14400s` and `/opt/conda/bin/python`. Environment is a separate JSON object, not shell assignment text embedded in argv.

Before importing NumPy, torch, or project modules, the runner checks the launch-pinned bundle-manifest SHA, the staged runner and builder hashes, the bundled copy of `SOURCE_MANIFEST.json`, and every one of the 58 Python source files in `/root/wnn/lr377_source_r1`. It checks the frozen snapshot aliases and the fold-1 split/input-manifest hashes as well. The builder repeats the source checks before its numerical/project imports. `WNN_PROJECT_ROOT=/root/wnn` is used only for the audited data and artifact roots. `wnn.distillation.combine_teacher_chunk`, `train_rnn.py`, and `predict_run.py` resolve from the frozen snapshot; the runner does not import `distill_pipeline` or call `dp.predict`.

R2 stores only `main_homogeneous_train.npy` and `main_homogeneous_valid.npy`. The target is computed with the frozen source arithmetic from the fixed family/source coefficients and the strong-family mean scaled by `0.8125`. The fit membership is fold 1's 10,533 train and 1,416 valid groups, totaling 11,949 rows; the prediction export reserves all 1,873 valid target-index rows. The runner validates the cache with the frozen consumer before fitting. It never exports teacher predictions, deletes cache contents, or removes partial outputs.

The runner takes a nonblocking sequence lock, refuses existing student outputs and prediction `.partial` files before fitting, and takes a separate cache-creation lock. Before aliases or cache writes, it estimates the compact cache payload, remaining valid predictions for all four arms, 64 MiB per arm for checkpoint/training-state output, and a 16 MiB metadata margin. It also requires the job's explicit external reservation and preserves at least 20 GiB free. The packet carries the root-supplied 6,000,000,000-byte external reservation. At the latest supplied 31,525,744,640-byte free-space snapshot, the computed margin above all reservations and the 20 GiB floor is 655,135,488 bytes. Root must refresh both free space and the external reservation before dispatch; the runtime preflight fails closed if the filesystem no longer fits.

The four fixed arms are `splice_r2` (splice, 0.50), `spmix_r2` (splice and mixup paths, 0.50), `swn_r2` (splice, warp, and noise, 0.50), and `swn25_r2` (splice, warp, and noise, 0.25). Each uses fold-1 teachers `combo_scratch_ep3s4_onecycle_f1@ep2` and `combo_scratch_ep3s5_onecycle_f1@ep2`, parent `c20261003_combo_s3_f1@ep3`, and the original two-epoch fixed-epoch-1 student recipe. Fold 1 was exposed before these retries; this packet makes no held-out or public-score claim.

Root owns source staging, a fresh disk/admission review, sequential dispatch, and terminal output collection. This packet itself performed no remote mutation, queueing, fit, provider action, or spending. R1 remains preserved at parent commit `b7adaeb7` and remains blocked from admission.

The focused local checks are:

```sh
scripts/limited .venv/bin/python research/ood_cache_repair_20261011/r2/test_ood_r2_packet.py
scripts/limited .venv/bin/ruff check research/ood_cache_repair_20261011/r2
scripts/limited .venv/bin/python -m py_compile research/ood_cache_repair_20261011/r2/*.py
```
