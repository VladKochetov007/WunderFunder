# Student capacity: preserve width, remove one recurrent layer

**Recommendation: one fixed h256×1 causal student, using the original d256 recipe and its original plain h256 seed3/seed4 teachers at α=.5.** This is a compression hypothesis with plausible competition utility, not evidence of a better predictor. Keep the 256-wide input decoder and head; ask whether the second recurrent layer is worth its serving cost. Do not simultaneously switch to a distilled d256 teacher, add a nonlinear encoder, factorize matrices or introduce QAT.

This assessment reads source and existing reports only. No model, dataset, prediction array, metric, timing probe or training was executed. SSH source retrieval failed, so historical shell scripts and teacher-cache files have not been freshly verified. The old overnight sources and delivery files remain unchanged.

## What history actually supports

- **E226 is weak feasibility evidence, not a clean depth result.** One-layer EMA+since, seed4/fold0/fixed ep3: single gain +.0005±.0009; pure-chain pair +.000359±.000522 (exact mask), below its +.0005 gate. Its two-layer seed4 comparator lacked since-weighting. No second-fold replication followed. The ledger does not bind hidden width; do not silently infer an exact modern h256×1 recipe from this entry. [Ledger E226](../../experiments.jsonl)
- **Width can matter after head FT.** The September29 journal reports h256 head-FT seed triples +.0007 versus h128 triples; the later matched three-seed audit has per-model deltas +.000073/−.000339/+.000792, so optimizer variability is substantial. A wider encoder is a better-supported prior than “every extra recurrent dimension is useless.” Two-layer h192 was not a smooth halfway improvement; width384 was +.0005–.0010 at the parent but −.0003–.0006 after head FT on one fold. Neither result directly tests shallower h256. [Journal](../../quant-probes-2026-09-29.md), [matched-seed audit](../2026-09-30-creative/M01-paired-seeds.result.json)
- **Original distillation helps h256 more reliably than h128.** The h128 distilled pair gained only +.0002, folds +.0006/−.0001. Original d256 gained +.0016 against h256 seed3, positive on both folds; replacing B's wide member added +.0004. Its actual metadata uses teachers `h256_f{fold},h256s4_f{fold}`, α=.5, EMA1/.999 and since32/.25. These are logical historical identities, not newly verified cache paths. Generation2/stronger privileged teachers did not improve the serving ensemble. [Journal](../../quant-probes-2026-09-29.md), [checkpoint metadata inventory](../2026-10-01-overnight/training/checkpoint-inventory.json)

The combined prior is therefore: **retain the width that narrow students may lose, remove depth whose marginal value is less established, and keep the proven teacher/optimization recipe fixed.** This is not proof that the one-layer student can represent the teacher: recurrent state shrinks from512 to256 floats and compositional depth disappears.

## Exact arithmetic scale from the actual model

For input211, hidden H, L GRU layers, each layer has two 3H×H matrices. The head is 2H→H→23 during training and 2H→H→2 at serving. Thus training parameters are `(6L+2)H²+(236+6L)H+23`; dense serving MACs/row are `(6L+2)H²+213H`. Normalization buffers, nonlinearities, bias adds, feature construction and Python overhead are excluded. One MAC means one multiply-add, not one FLOP.

| Model | Training parameters | Dense serving MAC/row | Recurrent state floats |
|---|---:|---:|---:|
| Current h128×2 |261,143|256,640|256|
| h192×1, cost context only |341,399|335,808|192|
| **Proposed h256×1** |**586,263**|**578,816**|**256**|
| d256/h256×2 |981,015|972,032|512|
| Historical h384×2 |2,159,639|2,146,176|768|

The candidate removes **40.5% of d256 dense MACs and40.2% of parameters**, but costs2.26 current h128 members. The existing native engine discovers GRU layers from weight keys and already supports one layer; no new recurrent kernel is conceptually needed. Export/parity and actual complete-callback timing are still required. See [RNNModel](../../../src/wnn/nn.py) and [native loader](../2026-10-01-overnight/critic/native_order_v2/include/wnn/model.hpp).

**Do not infer a fitting ensemble from MAC savings.** Proposed wide student plus two h128 members is1,092,096 MAC/row versus769,920 for current fixed3: **+41.8%**. The candidate plus one h128 is835,456, **+8.5%**, but drops an independent member and is a different composition. The candidate's half dense matrices alone occupy roughly1MiB, before its ~211KiB FP32 input projection; cache behavior can make arithmetic ratios optimistic on the host's1MiB L2. Equal WP to d256 at materially lower measured cost is already useful compression; it is not automatic qualification of fixed3/fixed4 under the time budget.

## Why not low-rank recurrence or QAT first

1. **Low-rank recurrence ranks second.** Historical D3 truncated an older h128 model on just7 sequences: rank32 output correlation≥.9996 still included sequence WP changes down to−.006. Spectrum energy is not a forecasting noninferiority bound; the old report's assertion that width256 adds unused dimensions predates the later width gains. A hypothetical *joint-gate* rank64 factor of each 3H×H recurrent matrix at H256 costs4Hr instead of3H², reducing a two-layer member to709,888 MAC/row (−27.0%, before optional diagonals). Input matrices and head remain dense. It requires different training parameterization and a factorized native operator; materializing UV during serving would remove the speed benefit. It is a more invasive candidate with weaker current-model evidence. [Historical source](../../lit/opus_arch.md)
2. **QAT ranks last for this bottleneck.** Q3 dynamic-INT8 lost only .00001145 WP versus identical FP16 composition, whereas the full four-member quant engine still cost56.57min against its50min gate. Training away that already tiny quantization error does not accelerate the same kernels or remove model members. New smaller quantization schemes would be a different unsupported numerical candidate, not free recovery of d256 capacity. [Closed frontier](../2026-10-01-overnight/frontier.md)

## Cheapest decisive forecast screen

Reuse verified **same-fold d256 parent4→ordinary H2** anchors as the depth-control lineage. Train only h256×1 seed3 on folds0/1: two parents at fixed ep3, then two ordinary H2 fits at fixed ep1, **four new fits**. Parent: original causal h256 teacher pair α=.5, EMA/since/aux/masks/schedule unchanged. H2: freeze input and GRU, leave the head trainable, true-mask pooled-correlation recipe unchanged, no distillation targets. No width/rank/teacher/epoch sweep.

Fix comparisons before any result: replace only d256 in the historical fixed4, retaining plain h128 seed3, plain h128 seed4 and c500, all at weight .25. Compare this candidate fixed4 to the corresponding d256 fixed4. Separately compare that same candidate composition to current plain-h128 fixed3, whose members have weight 1/3. The primary comparison asks whether cheaper recurrence preserves d256's useful contribution; the secondary asks whether the unchanged candidate composition improves the operational incumbent. No blend search or alternative member set is included. A paired seed3 screen is not a depth theorem; conditional seed4 replication is needed before a stable architecture claim. Actual streaming cost is a separate necessary gate and can reject a quality-preserving student. Candidate fixed4 has 1,348,736 dense MAC/row versus 1,741,952 for original fixed4 (−22.6%), but still +75.2% versus current fixed3; quantized native inference may be needed, and would require its own numerical and runtime validation.

**Unresolved before allocation:** exact teacher-cache availability/lineage, exact ordinary H2 anchor hashes, native runtime budget and noninferiority/utility margins, and sampler RNG parity. Model construction with a different layer count consumes a different number of RNG draws; merely reusing seed3 does not guarantee the old data order. This screen is a competition-recipe comparison, not a perfectly controlled causal ablation of depth; disclose the potential ordering difference. Do not expand the wrapper merely to claim matching seeds imply matching trajectories. No runnable launch command or empirical authorization is supplied here.
