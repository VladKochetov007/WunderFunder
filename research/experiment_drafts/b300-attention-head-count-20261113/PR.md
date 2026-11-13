# Does a wider attention head help at fixed model width?

**Status.** Prepared B300 screen for a standalone causal Transformer teacher, E-B300-HEAD-COUNT-20261113 / H-B300-HEAD-COUNT-20261113. Four new native configurations, a staged receiver, independent grading, paired readout and MIDAS handoff. The native runtime is pinned to PR75 at `8626000a4b965763a70011a0291a891aa6535d7e`. Implementation owner is Ivanhacker13. Execution belongs to the receiver named in the stage receipts. No allocation or model result is claimed.

**Mechanism.** Wider heads can score more feature combinations jointly in each temporal map, while fewer maps reduce the number of independent time patterns a layer can represent. We compare six heads of dimension 128 against twelve of dimension 64 at width 768. The two arms have identical trainable parameter shapes. Native RoPE frequency grids and SDPA scaling follow head dimension.

**Why spend B300 time.** This tests a trunk change without enlarging the teacher or modifying the model factory. If it earns another allocation, the next question is whether a stronger teacher improves the existing small student. Distillation costs zero fits in this screen. PR48 varied model width and preserved dimension per head. PR91 changed the input map and independently scored a negative E4 delta for bilinear input. [Prior comparisons](IDEAS.md)

**Frozen comparison.** Two matched seed pairs, 207301 and 207302. New controls use plain native attention. Both arms share 24 layers, width 768, 1,024-row context, 256-row chunks and gradient horizon, batch 128 per rank, four-chunk accumulation, two B300s per fit, BF16/BF16x9, Muon with AdamW groups, peak LR 0.0003 and 5% cosine-schedule warmup. Training uses label-only WMSE with the native chronological loop, per-sequence resets and fit-only normalization. The official grade is fixed at E4 / 3,280 updates, regardless of interim peaks.

**Budget.** Two fresh 12-head fits and two fresh 6-head fits. Each cell needs a separate real-shape 50-update profile, a named owner and a reviewed allocation. Physical GPU time and memory depend on the receiver's measurements. No previous fit or expired compute budget is reused.

**Validation and leakage.** Fit on 10,325 official train sequences, excluding 282 linked sequences, and independently grade all 1,873 official valid sequences with 3,528,273 scored rows. The split digest, normalized configs, dataset identity, checkpoint and prediction lineage are verified. This is one development-exposed fold. Report both seed differences, each target, and a conditional interval over validation components. Do not interpret two seeds as a significance test or add local gain to public WP.

**Decision.** Both paired E4 deltas must be positive with mean at least +0.0005 WP to prioritize confirmation. Mixed signs or smaller gains are inconclusive. Retain consistently negative endpoints along with the resource cost. No later checkpoint selection or hyperparameter tuning on these four grades.

**Run and return.** [Receiver commands](B300_COMMANDS.md) cover qualification, profile, approval, fit, canonical export/score and four-cell readout. [MIDAS instructions](midas/README.md) describe the observed preparation route and subsequent evidence export. Preserve all logs, grade manifests, data/config/source identities, environment, peak resources and hashes. Weights stay in the receiver's artifact store.

**At publication.** Zero local tests, model runs or profiles, zero remote scientific runs for this packet. Numerical acceptance and leaderboard impact are unknown. Existing training jobs and shared code were untouched.
