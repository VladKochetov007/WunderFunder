# F0 old-Z4 student update/reload diagnostic (R1 preparation)

This packet is frozen for root review and remains **not admitted**: both `fit_authorized` and `launch_authorized` are false. It does not launch a worker. Root owns any later admission, resource-profile creation, attempt receipt, and execution.

The diagnostic follows the exact own-source H230 H256 seed-3 F0 recipe and inputs bound by the admitted profile R2 contract. It uses `oldz4_h230_parent_f0_s3_r3@ep3`, the fold-0 split, the eight-teacher own family cache, the same own propensity model, raw family targets, and alpha 0.7. The exact absolute trainer path is the frozen `research/.../continuous_execution_E529/h230_source_v3_all_data_distill/scripts/train_rnn.py` snapshot with its pinned `PYTHONPATH`; no trainer file or R2 evidence is changed.

The intended prefix is only two loader batches, or 20 optimizer attempts at batch 16 and chunk size 2,000. It intentionally runs a one-epoch OneCycle schedule with a fixed epoch-0 checkpoint and `training_state.pt`. That schedule and horizon differ from the full student recipe. This checks finite updates, source/input bindings, checkpoint/state serialization, and exact no-op reload; it gives no recipe-quality score and cannot authorize the later full fit.

`ROOT_STUDENT_UPDATE_DIAGNOSTIC_CONTRACT_R1.json` stores the complete timeout/environment/trainer argument vectors for both stages and the content hashes of the source snapshot, parent, split, propensity, and cache metadata. The prefix command has only these changes from the actual R2 profile trainer command: fresh run name; epochs 1 / fixed checkpoint epoch 0; profile flags removed in favor of `--max-batches 2`; and `--save-training-state` added. The reload command preserves those same arguments, points `--resume-state` at the prefix state, and uses a separate output run name. The prefix and reload run/output IDs are checked unused by static validation.

The requested resource envelope is 2 reserved CPUs, 6.5 GB memory reservation/maximum, `MemoryHigh=6.1 GB`, no swap, 600-second timeout with 20-second kill grace, GPU allocator 4.3 GB, observed VRAM ceiling 5 GB, zero loader workers, and one thread. Parent boundaries remain 18 GB high / 20 GB maximum. The increased `MemoryHigh` responds to R2's measured 5,852,094,464-byte peak and 16,069 high-event delta. Root must confirm live aggregate capacity and apply the matching actual profile before any launch.

The acceptance gates are in the contract. Both update counters must be explicit and sum to 20; checkpoint and complete training state must be finite; no new OOM/max/kill event is allowed. The no-op reload must pass the source-bound resume-contract check, add no updates, and re-export tensors exactly equal to the prefix checkpoint. A nonfinite `GradStats` norm alone is not an AMP-skip counter; preserve it beside `amp_skipped_updates` and `optimizer_completed_updates` for root's diagnosis.

## Static preparation checks

The contract builder re-derives both command vectors from the pinned R2 profile command, verifies ten source/input digests against the R2 pins, checks the resource and environment shape, rejects already-used run/attempt IDs, and refuses to overwrite the frozen contract. The static receipt is `STATIC_VALIDATION_R1.json`, schema `wnn.oldz4_student_update_diagnostic_static_validation.r1`, status `PASS_NOT_ADMITTED`.

Run the code-only contract check from MAIN (no fit/provider actions):

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python \
  /home/vlad/development/wunder/wnn_old_pipeline_reconstruct_20261011_r1/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/local_oldz4_f0_student_update_diag_r1/freeze_update_diagnostic_r1.py \
  --check
```

No GPU command or `bandit_resources.py launch` command is issued by this packet.
