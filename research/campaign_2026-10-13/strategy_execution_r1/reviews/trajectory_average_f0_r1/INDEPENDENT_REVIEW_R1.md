---
kind: review
status: passed_artifact_and_consumer_preflight_quality_pending
created_utc: 2026-10-13T00:27:00Z
hypothesis_ids: [H-STRATEGY-AVERAGING-20261013]
experiment_ids: [E-STRATEGY-SWA-F0-20261013-R1]
evidence_paths:
  - ../../trajectory_average_f0_r1/CONSTRUCTION_RECEIPT_R1.json
  - ../../trajectory_average_f0_r1/S3_MANIFEST_R1.json
  - ../../trajectory_average_f0_r1/S4_MANIFEST_R1.json
  - ../../trajectory_average_f0_r1/EXPORT_AND_GRADE_COMMANDS_R1.json
  - INDEPENDENT_REVIEW_R1.json
---

# Independent review: fixed F0 trajectory averages

## Verdict

The S37 artifact construction and exact staged CPU model consumer pass independent checks against the real seed-3 and seed-4 producer files. I found no checkpoint, ancestry, averaging, architecture-load or filename-matching blocker. The two averages are ready for feature-only prediction and official grading once the shared GPU controller grants a fresh slot. This review makes no quality claim: neither averaged model has predictions or scores yet.

## Scope and independent checks

The reviewed implementation is `scripts/average_trajectory_checkpoints.py` at candidate commit `f5c3c6bb06d819b8ab2cf0cef2abdc1e5b81a73d` (parent `5836dbc83e13031ff398ab3c883f2469df84bfd9`), SHA256 `8646ccc12b537a1bf2f2544dffd85d10287076451e5bab86a4b2d6379e39bbb3`. The three unit tests in that candidate are reported passing by its construction receipt; this review did not rely on that claim alone. I independently loaded the actual manifest-bound producer checkpoints and continuation states from Main, reconstructed the complete model-state mean, and used the exact R2-overlay predictor to rebuild each derived model on CPU and load its state strictly.

For both seeds, all three producer file hashes matched their frozen manifests. The scientific arguments, fold/split, fit membership, seed-specific normalization, initializer/propensity ancestry and shared distillation-cache identity matched. Epochs 4 and 5 carry the same epoch-3 continuation record. The referenced continuation-state file's hash, parent directory, `next_epoch=4`, seed, fit-membership and normalization contract, and saved model tensors match the epoch-3 checkpoint. Each derived model tensor exactly equals the FP64 accumulation and cast specified by the constructor; fixed `mean`/`std` buffers and non-floating tensors are identical. Output checkpoint and external provenance hashes match the construction receipt. The `model.pt` alias has the same hash as `model_ep5.pt`.

The staged R2 `predict_run.py` reconstructs both derived models successfully with strict `load_state_dict` on CPU. The retained epoch-5 training arguments do not determine the serving run identity: the exporter takes the distinct run name from argv to resolve `artifacts/runs/<run>/model_ep5.pt`, and uses that same name for its prediction and sidecar. Each paired grade argv requests that exact run at epoch 5; the grader reads `cache/pred_<run>_valid.npy`, validates the run/epoch/split and row-order sidecar, then grades the fixed 468-group F0 panel with `global_wp_clip2_v1`.

The command packet invokes the actual `source_overlay_r2/scripts/cleanfold_exec.py`, whose child is the matching overlay `scripts/predict_run.py`; its environment pins `WNN_PROJECT_ROOT` to Main and the model is loaded from the derived run directory there. The campaign split path is absolute. The full-valid selector obtains only row-group order from `valid_targets.npz`, and `FeatureOnlySequences` reads parquet features with `targets=False, mask=False`. Targets and the fixed scoring mask enter only in the grading stage. The expected prediction is a float32 memmap of shape `[1873, 20000, 2]` (299,680,000 payload bytes); both unique prediction and grade destinations were absent at review time, so the packet will not mistake an old export for a new result.

The exact run IDs and outputs are `clean_t3_f0_control_s3_swa345_strategy_r1` → `cache/pred_clean_t3_f0_control_s3_swa345_strategy_r1_valid.npy` → `grade_s3_r1/eval468_pred_clean_t3_f0_control_s3_swa345_strategy_r1_ep5.npy` and its `.grade.json`, with the corresponding `s4` run and `grade_s4_r1` path. The matched CP5 receipts both use `global_wp_clip2_v1`, the same 468 evaluation groups and split SHA `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`: [seed 3](../../../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/student_fit_r2/f0_grade_r1/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1.grade.json) scores `0.6811553603333363` (receipt SHA256 `c518f7cdb5f3e5148f72de152c3aaecd308ff97c83aa74169fdaa95442e085a0`); [seed 4](../../../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/seed4_confirmation_r1/f0_grade_r1/clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1.grade.json) scores `0.6812584217032286` (receipt SHA256 `b9b189772581f35df1d3b3187b41d40e0410194161b266cba22033b90781c3fc`). These are appropriate paired reference endpoints, not evidence about the averages.

## Bounded verification receipt

The independent verifier ran read-only under the approved local wrapper with one CPU, a 1,200,000,000-byte memory maximum and a 120-second runtime maximum. It used CPU-only `torch.load`, independent tensor arithmetic, provenance/hash checks, and the overlay model loader; it did not run a forward pass, access a GPU, fit a model, or read the supplied parquet data. The exact wrapper was:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1200000000 WNN_RUNTIME_MAX=120s OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python -
```

The verifier exited 0 for both seeds. Systemd recorded scope `run-p293543-i312697.scope`, 1.645 seconds wall time, 1.621 seconds CPU and a 288.9 MB memory peak. The parent admission bracket included the full remaining worker reservation plus root/operator support and an extra 1 GB margin, totaling 14,975,744,000 bytes against the 17 GB target. This check did not consume GPU capacity or alter artifacts.

Exact producer, output and source hashes are in [the machine-readable review receipt](INDEPENDENT_REVIEW_R1.json). The frozen export/grade command packet has SHA256 `6f4da732aaf43db42111e2d01e4c2b098097c034b4bec2c871740625a9579ece`; its overlay `cleanfold_exec.py`, overlay `predict_run.py` and Main grader hashes are respectively `246586bfd4be3a4e3d95ac026860bb835611f09958193b75798124c9500455ab`, `30647bb74da64ddab4ccaa5e2916338bbb4826466b47cd68ec61593269554dd0`, and `b6f80ef2d5323bdd9f9941f8bb2b698d61d87ac05fe918d53343a52aee0e6654`.

## Interpretation and remaining limits

This is a fixed equal parameter mean over epochs 3, 4 and 5 of each existing six-epoch OneCycle trajectory. It does not reproduce the constant/cyclical-tail form commonly called classical SWA, and the averaged student seeds are not independent teacher replications. The manifests show one epoch-3 checkpoint followed by a shared epoch-3 continuation for epochs 4 and 5; describing these as three independent trainings would be inaccurate. The existing experiment note's wording “three independently trained own checkpoints per trajectory” should be read as own checkpoints from a seed-specific trajectory, not three independent trajectories.

Score each average against its same-seed CP5 control using the identical fixed F0 panel and official metric, then report the paired changes and their mean. F0 has been exposed repeatedly and the two student seeds share one teacher cohort. These results can support a discovery-stage averaging decision, not fresh-fold robustness, teacher-seed variation or public transfer. Do not promote the method without actual predictions, score receipts and broader cross-fold evidence.

The reviewed packet's `WNN_GPU_FRACTION=1.0`, 1 GB live VRAM reserve, one loader-free worker and shared-controller requirement are consistent with current authorization. The active GPU chain still owns the single local lane; do not launch directly or concurrently. Before admission, confirm the output filesystem is the approved physical cache volume, reconcile the current shared lock and memory bracket, and retain the exact attempt/grade receipts. These are execution gates, not defects in the averaged artifacts.
