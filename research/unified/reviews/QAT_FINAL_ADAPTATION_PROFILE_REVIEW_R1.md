# Final QAT adaptation recipe: profile readiness review R1

**Verdict: GO for the four root-controlled actual-shape profiles after fresh resource admission. This review does not clear any fit.** The source gate is explicit and defaults off; the fit vectors are matched within each seed, and I found no recipe or parent-identity blocker for collecting profile evidence.

The frozen source is `e1ab711c`, based on the independently reviewed LSQ source `6dde7d9d`. The recipe pins SHA256 `b4fda57f4ca8680289fd0252d8c4a4077f0871d1fe0b4c94e34ea43c28c29eed`; the two source files it declares match their frozen bytes. A direct diff of `scripts/train_rnn.py` against the LSQ base shows only the final-adaptation CLI gate, its all-data initializer validator, and passing that opt-in to the initial-checkpoint validation call. `--final-model-adaptation` is `store_true` with no explicit true default. The previous clean-fold initializer-validation branch remains intact, and the separate propensity-checkpoint validation still calls the strict default path.

## Recipe and identity checks

The eight frozen command vectors form two same-seed comparisons:

| Seed | Floating control | QAT treatment | Initializer checkpoint SHA256 |
|---:|---|---|---|
| 3 | FP32 continuation | `recurrent_w4` with learned row scales | `04675f8220ba949c1dfb6c5796a3f25a9b8f655bb48f83c188bf8f9bacd361e0` |
| 4 | FP32 continuation | `recurrent_w4` with learned row scales | `cefee5b51cae7b72247a26558444f526e915a9fb7ad3927abd6cc02e63018419` |

For each seed, the two fit vectors are identical after removing run name and the expected QAT selection flags; profile and corresponding fit vectors differ only by run name and profile instrumentation. Both arms use the same seed-specific all-data H390 CP5 parent, AdamW (`lr=1e-4`, `wd=0.01`), OneCycle, FP32 arithmetic, gradient clip 1, fixed raw epoch 0, one epoch, and no fit-time evaluation, EMA or SWA. The treatment is the reviewed LSQ-inspired learned-scale implementation, not canonical LSQ; `recurrent_w4` uses its existing mixed Q4 recurrent/Q8 output-head selection.

The actual parent checkpoint and `provenance.json` byte hashes match the recipe for both seeds. Sidecar provenance records `train_source=train+valid`, `include_heldout_valid=true`, no split manifest, no fit-time evaluation, fixed checkpoint policy and checkpoint epoch 5. The source guard checks these same fields in the initializer checkpoint metadata. I did not deserialize the `.pt` files; byte hashes and provenance were checked without Torch. Each candidate arm has `--inherit-init-preprocessing`, so the FP32 and QAT arms within a seed inherit the same parent normalization. The two seed-specific parents have different normalization hashes, as expected for these distinct ancestors; the recipe does not claim cross-seed normalization identity.

Both seed pairs use the fixed F0 component-subset manifest (SHA256 `8aab0e18cef6bca99a8c0b6e7c7260fcb4eceb9d6cd67efd1f7437755b98cdf4`), the same 315 selected components and member digest `254723b332b28eafa1e8b22dc283e8f886dbc6fa954894fdf46b769e5ebb1c30`, and F0 split SHA256 `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`. The teacher cache metadata is SHA256 `1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8`; the F0 propensity checkpoint is SHA256 `b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7`. The cache metadata, split and checkpoint bytes were independently hashed. Propensity validation remains on the clean-fold path; the new initializer exception does not cover it.

The training target is intentionally a provisional transfer. The original all-data final target cache is unavailable, so these student arms retain the existing clean-T3 F0 teacher targets with `distill-alpha=0.9`, homogeneous-cache scale 1, `aux-struct=0.3`, `aux-next=0.3`, and the same component-complete 10% subset. The recipe records this change and explicitly disclaims a held-out claim. The all-data parents themselves include validation membership; therefore neither these endpoints nor the F0 panel can support a clean validation or public-score forecast.

All commands use B16/C1000, random batch order, the same frozen F0 split and subset, propensity power 2 with epsilon 0.03 and `valid-true-mask=0.03`. The objective is cached-target wMSE with chunk normalization. Each profile requests two warmup and three measured full-sequence batches; with 20 chunk updates per batch this exercises 100 optimizer updates total. The corresponding fit requests 1,440 updates (72 batches over the frozen 1,152-sequence subset, 20 chunks per batch). Profile and fit inputs, loss, optimizer, treatment and initialization are otherwise aligned.

## Scope and dispatch condition

The recipe's job array lists each profile adjacent to its fit. Root owns the phase gate and should dispatch only the four `phase=profile` entries now; all four 100-update results must be finite and admitted before a separate fit decision. The profile runs can establish engineering feasibility only. They do not establish quality or authorize the 1,440-update jobs.

I inspected only source text, small JSON metadata, checkpoint hashes and command vectors. Bounded standard-library assertions passed for all within-seed arm parity and profile/fit parity. No Torch import, model deserialization, GPU run, fit or prediction read occurred.

Source: [`train_rnn.py`](../../../artifacts/qat_renewable_r1/final_adaptation_source_r1/scripts/train_rnn.py), [`training_qat.py`](../../../artifacts/qat_renewable_r1/final_adaptation_source_r1/src/wnn/training_qat.py). Frozen inputs: [execution recipe](../../campaign_2026-10-13/qat_renewable_r1/final_release_r1/RECIPE_R1.json), [component subset](../../campaign_2026-10-13/qat_renewable_r1/final_release_r1/fit_components_10pct.json), and the actual seed-3/seed-4 parent [provenance](../../../artifacts/runs/vt_all_h390_ds_homogeneous_s3_r3/provenance.json) and [provenance](../../../artifacts/runs/vt_all_h390_ds_homogeneous_s4_r3/provenance.json). The machine receipt records all hashes and bounded checks. Experiment `E-QAT-FINAL-ADAPTATION-20261013-R1`; no score or package-ranking claim is made.
