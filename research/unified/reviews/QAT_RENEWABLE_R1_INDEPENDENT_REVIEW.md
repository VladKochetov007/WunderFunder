# Independent QAT implementation, recipe, and ancestry review — 2026-10-13

**Verdict:** the Python QAT policies and packed formats match the admitted specification, and the R2 native consumer passed independent actual-data transport, score-mask, callback-reset, and state checks. The frozen R2 evaluator is clear for the controller-owned full-fold comparison. The reviewed small profile is an unchanged-parent PTQ screen, not a QAT result. The current QAT recipe's post-fit FP32 panels are diagnostics; promoting trained QAT models requires the separate matched native-export comparison the plan specifies.

## Scope and source identity

I reviewed the admitted [QAT plan](../../campaign_2026-10-13/qat_renewable_r1/PLAN.md), [versioned policy/format specification](../../campaign_2026-10-13/qat_renewable_r1/QUANTIZATION_SPEC_R1.md), Python source at commit `7c64fa8f`, native exact-import/state source at `41746a5a`, the actual ignored MAIN artifacts, and the actor's isolated recipe/ancestry files. The actor packet was read from `/home/vlad/development/wunder/wnn_qat_delivery_r1/research/campaign_2026-10-13/qat_renewable_r1/delivery_r1/`; its `RUN_RECIPE_R1.json` SHA-256 was `43c05cbe9d9504eff2d0736c7dfd63432a0879842eaf5ca101b73860f835e2e9` and its ancestry receipt SHA-256 was `386504aa81b5cb1895c7919ada9fdc223ec8141a1deb0f15d86d4652cd2ceea7`. These actor files were still isolated and unmerged at review time.

The admitted evaluator correction is commit `d0650d24`. Its frozen source snapshot has SHA-256 `6e2c32088afbcca517d14bb40d08359eb03d93cdc829812f15f8a89ba5316f8a`; the R2 manifest SHA-256 is `8894af708dcba1500fc8c1b431d30468a29a9999531dd7133ec982c3b1d19556`. The native library hash is `a4390ecf7bcd54bb6826b43f5dcf6d7a40064c64ed1450bdc1780c34c2a6e108`. The manifest binds the CP5 parent checkpoint (`e30577ff…776eda`), source-qualified F0 split (`09263ad4…16a3d`), evaluator/engine/packager and quantization sources, four transport archives, and the native library.

## Implementation and serving semantics

The reviewed policy selects every GRU input and recurrent matrix for uniform W4; mixed mode selects GRU input matrices for W4 and recurrent matrices for W8. Both split `head.0.weight` at the 256-wide hidden boundary into two Q8 blocks. Projection weights, biases, and the final head remain float32. Row scales use float32 `max(abs(row))/7` or `/127`, floored at the smallest positive float32 subnormal for nonzero rows, with `rint` nearest-even and exact `code * scale` represented values. Zero rows map to zero scale/codes; invalid signed minima (`-8` for Q4, `-128` for Q8), inconsistent zero rows, and nonzero Q4 padding are rejected. The STE passes identity gradients to latent parameters; cache invalidation and checkpoint continuation paths are covered by the focused Python tests.

Packed Q4 uses low-nibble-first signed two's-complement codes with zero high-nibble padding for odd counts. The actual archive reader preserves the decoded code and float32 scale bits. I independently imported the packed-Q8, W4, and mixed archives into the exact hashed native library and compared exported codes and scale bits; all matched. Packed Q8 also matched the native Q8 export from the actual parent. The native path retains the existing I16 input projection, A8 activation quantization at recurrent/head matvecs, and FP32 recurrent state. QAT does not simulate the I16 projection or A8 activation rounding. Q4 codes are expanded into byte `DynQ8` matrices: the smaller archive does not mean resident INT4 or a native INT4 speedup. ISA dispatch supports VNNI, AVX2, and portable kernels; this review did not rerun the Rust suite or benchmark each ISA on target hardware.

The actual 16 parent tensors in `parent.npz` match the frozen manifest's per-tensor hashes and expected 211-input, two-layer H256 shapes; normalization values are finite with positive standard deviations. This independently binds the profile's parent export to the declared CP5 model.

## Recipe and ancestry

The nine fit vectors share all trainer arguments within each subset except run name, quantization policy, and subset manifest. They use the same F0 seed-3 CP5 initialization, FP32 training, fresh AdamW state, max LR `1e-4`, one-cycle schedule, batch 16/chunk 1,000, fixed raw epoch-0 checkpoint, `ema_teacher=0`, and `swa_frac=0`. The common teacher targets, auxiliary terms, propensity settings (`c20260929_prop_f0`, power 2, epsilon .03), and target-cache path are fixed. The actual nested manifests select 315/945/3,148 fitting components (10/30/100%) with 1,048/3,130/10,535 train groups and 104/387/1,405 valid-fit groups; each is component-disjoint from the 468-group F0 evaluation panel.

I independently checked the seed-3 parent P1→P2→P3 training-state hashes and the CP5 initializer hash. The CP5, combo initializer, and clean-T3 teachers bind the exact F0 fit pool and exclude evaluation components. The propensity checkpoint is fit only on the 1,405 source-qualified F0 fitting-valid groups. The homogeneous target-cache metadata binds the same fit pool, names three clean-T3 teacher sources, and records zero evaluation targets. The propensity sidecars bind the exact model checkpoint digest. Smaller subset results remain adaptation diagnostics because their CP5 initializer and cached teachers were fitted on the full F0 fitting membership. The current student recipe covers one fitting seed; broader seed/fold replication remains required by the plan.

The recipe's listed post-fit predictions explicitly use `--inference-precision fp32`; they measure the latent trained checkpoint, not the compressed deterministic endpoint. I found no QAT-checkpoint W4/mixed native export in those vectors and no matched PTQ export of the FP32 continuation. This is a limitation on interpreting those panels, not a blocker for the matched fits: the controller owns exporting each trained QAT checkpoint and the matched floating continuation through the frozen native evaluator. No QAT quality or deployment claim should be made from the FP32-latent panels alone.

The bounded ancestry audit checked cache metadata and membership but did not reread/re-hash the large distillation payloads. The trainer is configured to record SHA-256 values for consumed cache files in run provenance. The independent verification therefore confirms declared source/membership ancestry, not the full payload bytes before fit.

## Actual profile evidence and its limits

The R1 profile artifacts in MAIN compare unchanged CP5 endpoints on two hashed fitting components, with complete 20,000-row trajectories. The independent callback replay used the actual parquet rows and masks. All five scores below are pooled `global_wp_clip2_v1` values over 1,589 selected rows; they are fit-only engineering diagnostics.

| CP5 endpoint | WP | Archive bytes | Local callback seconds, two trajectories |
|---|---:|---:|---:|
| FP32 | 0.273464 | 3,628,746 | 5.466 |
| Native Q8 | 0.272609 | parent archive | 0.753 |
| Exact packed Q8 | 0.272609 | 1,086,824 | 0.760 |
| Uniform recurrent W4 | 0.254276 | 678,883 | 0.914 |
| Mixed input-W4/recurrent-W8 | 0.250039 | 882,966 | 0.806 |

These numbers describe two fitting components, not the 468-group full fold, cross-validation quality, or a new candidate. The local host timing is not an EPYC guarantee. W4 imports use the same 1,071,552 streamed native weight bytes as Q8 because Q4 expands into byte kernels; only archive size shrinks at this stage. The lower two-component W4 scores are a signal to measure on the full declared evaluation and after QAT, not a family-wide rejection.

## Verification performed

- Against the frozen R2 source and actual MAIN artifacts, `checked_manifest` passed for the admitted SHA. The R2 guard receipt records rejection of a wrong manifest hash, partial fold, and reordered fold. R2 adds source/hash and full ordered-fold checks while preserving the R1 archive bytes and inference code.
- Independently reconstructed the actual selected mask as `is_scored & need_prediction`, checked actual sequence identities and shapes, and replayed native-Q8 and packed-Q8 callbacks in A→B→A order. A replay matched the first A bit-for-bit; native and packed Q8 matched stored profiles and each other.
- Read-only native hidden-state snapshots at 1,000-row intervals matched the stored native/packed Q8 state diagnostic for both actual sequences. The actual callback's state getter is intentionally not exposed on `EnsemblePredictor`; the sequence-change behavior was tested through its `predict` callback.
- Recomputed the official two-group Q8 metric from actual selected labels, masks, and replayed predictions; `t0`, `t1`, `wp`, and selected-row count matched the profile receipt.
- Ran reviewed-source tests under the prescribed 1-CPU/1.5-GB bounded wrapper: 56 passed, 2 skipped because an explicit native test library was unavailable. The actual ABI3 artifact import/callback checks above are separate from those skips.

## Decision

**Independent engineering clearance: proceed with the controller-admitted R2 full-fold baseline comparison** using the exact R2 manifest SHA and current source hash. Preserve all outputs as exposed F0 development evidence. Continue separate evaluation of matched FP32-PTQ versus exported QAT-trained endpoints before ranking QAT quality, package size, or runtime. No public submission, clean holdout, or QAT gain is supported by this review.

Related IDs: `H-QAT-W4-20261013`, `H-QAT-A8-20261013`, `H-QAT-LSQ-20261013`, `E-QAT-DEPLOYMENT-20261013-R1`.
