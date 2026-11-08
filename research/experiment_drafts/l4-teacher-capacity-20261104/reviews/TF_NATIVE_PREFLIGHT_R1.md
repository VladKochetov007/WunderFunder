# Transformer teacher native preflight R1

## Findings

1. **Dispatch is blocked until the receiving panel is bound.** The reviewed local `TRANSFORMER_TEACHER_PANEL_R1.json` still has null `source_binding_sha256`, `data_binding_sha256`, and `fit_members_sha256`, and its status says it is waiting for root admission and device verification. Do not dispatch from this panel as-is. ROOT must bind the separate `source-tf-r1` source closure, data closure, official split, and exact 10,325-train/0-valid membership, then verify the current UUIDs for slots 6 and 7. Do not reuse the GRU source-binding hash: this is the separate S_TF source family.

2. **The trainer receipt validator does not enforce the declared 50-update profile length.** `run_profile` runs the requested count and enforces at least `diagnostics_every_steps`, but `validate_profile_receipt` only checks that `updates` is positive. The panel and receiving notes require exactly 50. The independent profile review must check `updates == 50` and evidence that the step-50 diagnostic ran before either fit. No trainer refactor is required for this cycle.

No source-critical adapter, mask, precision, optimizer, normalizer, ancestry, or checkpoint mismatch was found in static review. The code and recipes are ready for the root-bound 50-update profiles; no actual Transformer profile receipt exists in this local review packet.

## Recipe and implementation review

| Area | Reviewed recipe/source | Verdict |
|---|---|---|
| Adapter | Commit `ffebdbdc573e4494e0b53bd661c544ecc490b3df` changes `sequence_config.py` to allow `event211` for Transformers while preserving the GRU256x2-only restriction; `sequence_models.py` sizes Transformer projection and normalizer buffers from the selected input width. | Pass. Patch is narrow and does not route the GRU model through a new adapter. |
| Inputs | `SequenceRecords` builds the existing 99 event features from each complete 112-column sequence before chunking. The NumPy normalization reader and Torch training reader both reset event state per sequence. The Transformer projection and fitting normalizer are 211-wide. | Pass by source inspection. Existing event features use prior/current rows, held-event state, and cumulative recency; no later sequence rows enter earlier feature rows. |
| Architecture | Both seeds: causal RoPE Transformer, hidden 384, 8 layers, 8 heads, FF width 1536, context window 1024 including current row, dropout 0, checkpoint blocks on, math SDPA, compile off. Cache keeps at most 1023 prior rows and carries absolute positions across chunks; batch sequences begin with empty state and chunk state is detached. | Pass. Head width is 48; rolling mask and cache sizes agree. The layered maximum assembled-row receptive field is 8,185. |
| Precision and optimizer | `precision_mode=fp32`; model, floating buffers, batches, attention caches and optimizer state are FP32. Precision setup disables autocast, TF32 matmul, cuDNN TF32 and reduced matmul precision. AdamW beta=(0.9,0.999), eps=1e-8, weight decay .01; LR 3e-4, global clip 1. | Pass by source/config inspection. No AMP/scaler, TF32, quantization, or Muon path is configured. |
| Schedule and budget | Batch 8, chunk 1,024, accumulation 4, four fixed epochs. For 10,325 fitting sequences, ceil(10,325/8)=1,291 batches/epoch and 5 optimizer windows/batch yield 25,820 updates. `warmup_cosine_v2` uses ceil(25,820*.05)=1,291 warmup updates and reaches zero on update 25,820. | Pass; arithmetic agrees with the runtime update counter formula. |
| Fit and normalizer membership | Official chronological manifest SHA256 `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`; expected train 10,325, fit-valid 0, official valid 1,873, 3,528,273 scored rows. `all_fit` normalizer members come from `data.fit.members`; reader requests features without targets or masks and hashes split, source fingerprint, members, schema and moment bytes. | Pass in runtime source, conditional on root receiving admission matching those exact identities. No validation normalizer ancestry found. |
| Targets, masks, ancestry | Scratch fits: no initializer or teacher cache; config defaults `kd_weight=0`, `hint_weight=0`. Training and fixed train diagnostic mask rows >=99; official valid uses its source `is_scored` mask plus rows >=99. No random folds or `seq_ix` chronology. | Pass. The official valid is exposed development reporting, not an independent confirmation set. |
| Checkpoints and monitoring | Fixed 4 epochs, `checkpoint_retention=all_epochs`; fit command uses the same config/source/device as its profile, attaches the approved profile receipt, reports full official valid after each epoch; intended endpoint is epoch index 3. | Pass by config/runtime path. Best-observed monitoring remains report-only for the fixed endpoint comparison. |

## Machineable acceptance gates

Before a profile, ROOT must populate the source, data, and fit-member SHA fields in the frozen Transformer panel from a receiving admission; verify split SHA and exact train/valid membership; verify actual physical UUIDs for slots 6 and 7; enforce the shared deadline, 45-minute collection reserve, CPU/RAM/output caps and 3 GB free-space floor. Keep the Transformer source closure separate from S_GRU.

Run each panel profile command once for 50 real-data optimizer updates. Before fit, independently bind the exact panel, admission, source/data/membership, config and argv hashes, owned process receipts, trainer profile receipts and GPU UUIDs. Require process exit 0 without timeout; trainer receipt `profile_complete`, `updates == 50`, finite losses and optimizer state, positive consistent row counts, no errors, no scientific checkpoint, and profile device identity equal the selected GPU. Review memory and throughput, then mark only the unchanged, identity-matched profile receipt approved. The fit trainer rechecks exact config/data/source/runtime identity.

During each fit, retain all four epoch snapshots/history, training telemetry and sampled diagnostics, and full official-valid reports after each epoch. Keep epoch 3 as the predeclared checkpoint regardless of observed best epoch. Do not claim independent validation evidence.

## Review scope and limitations

Reviewed the contract, Transformer configs and panel, receiving notes, adapter diff, data/record/feature path, normalizer identity and cache path, attention/cache path, FP32 contexts, optimizer/schedule, fit/profile admission and monitoring/checkpoint path. Current local config hashes are:

- seed 1001: `49d576575c97b90b4237c674407efac7ea0da7830476fca029dc8af018f24f5a`
- seed 1002: `1513436a0f5f3a868ce713f0dc9c99e8fe6eee729fd226eb45b9fc6c93b957ab9`

No SSH, GPU, fit, profile, test, or CPU-check script was run. The data/member SHA and current GPU inventory are admission-time root checks, not locally independently re-read by this review.