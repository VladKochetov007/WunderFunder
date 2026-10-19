# R9 fixed-checkpoint recurrent-state history probe

R9 measures whether one frozen R01 F0 wide checkpoint's outputs change when its GRU carry is rebuilt from the last 256 rows or set to zero. All three branches receive the same canonical causal input features; only the recurrent state is changed. This is a checkpoint-conditional sensitivity diagnostic, not evidence that erased history improves labels or that longer input history is useless.

The fixed run uses the first eight fit components in the accepted R3 sample, two anchors per sequence, and 128-row rollouts. It never reads evaluation arrays, scores outcomes, backpropagates, or updates weights. The shared sample loader hashes and maps the selected x, y, and scored NPY files for integrity (24 files total, including eight y files); the probe does not access target values. Result JSON contains scalar summaries only.

On the admitted B bundle, invoke `python -m idea_screens_r9.run_probe` with `--sample`, `--checkpoint`, `--output`, `--expected-sample-sha256`, `--expected-checkpoint-sha256`, and `--expected-preparation-manifest-sha256`. The CLI verifies fixed identities and the source manifest before loading the checkpoint; the checkpoint is reconstructed before the selected fit arrays are mapped. The root-owned admission process performs the full B-shaped import/read-sentinel smoke and resource checks before launching the fresh CLI process.

The public `smoke_imports(bundle_root, source_root, pins)` function in `run_probe.py` imports the pinned runtime closure without opening sample arrays or a checkpoint. It requires a fresh interpreter before WNN modules have been imported.
