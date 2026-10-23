# Fresh reframe: cheap B300 transfer to Challenge 2

**Status:** proposal only. No model fits, runtime profiles, prediction exports, Kaggle quota calls, or provider launches were made.  
**Scope:** the assigned 7,200-second inclusive screen ceiling, maximum six new fits, strict FP32, 5 GB task VRAM/RAM bounds, physical-disk scratch, and protected 12-hour production reserve. All existing validation folds are exposed.  
**Objective:** improve a complete deterministic endpoint scored by unchanged `global_wp_clip2_v1`; current public TL3 is user-reported 0.66832. Local evidence selects candidates; it does not forecast public score.

## What I used

The directed primary source is [Transformer-XL](https://arxiv.org/abs/1901.02860), Sections 3.2 and 4.2. It caches prior-segment hidden states as context while keeping gradients inside the current segment; its experiments also separate longer context from context-fragmentation effects. This is a language-model result, not evidence of a gain on anonymized market rows. The repository's B300 design already proposes recent detail plus compressed completed-block history, so I use Transformer-XL to motivate a compact memory arm rather than restating the large-teacher design.

I read the current contract, the October 21 frontier and the B300 results review. The latter records a partial matched comparison where H512×1 beat H384×2 by 0.00077732 equal-fold WP (9/12 pairs); because width and depth both differ, this is a directional capacity prior, not a pure width effect. The tested broad geometry bundle was negative, while B300 age/innovation lines remain ungraded. The current local dynamics-auxiliary screen is already separately proposed, so I do not duplicate it. Sources: `research/experiment_drafts/b300-fast-local-kaggle-20261023/CONTRACT_R1.json`, `research/campaign_2026-10-21/next_cycle_r1/FRONTIER.md`, and `research/unified/reviews/B300_PR_RUN_UPDATE_REVIEW_20261019_R1.md`.

**Seeded library passage.** I sampled one PDF uniformly from the sorted 417-PDF pool under `/home/vlad/Documents/Trading`, whose sorted relative-path list SHA256 is `7bccbfa26a5ded5cd54276a43de0209e4a6cb018b6090fc1c83354814dc9329c`. Procedure: Python `random.Random(20261023)`; document index 129, then page index 4 (page 5, 1-based), then flattened-page character offset 103. Source: `Trading/Naive/Non-Linearity Without Machine Learning - by Quant Arb.pdf`, SHA256 `74ee76f385908a0eac0a0c5beac53054df086d40a017192b90b8148729995114`. Selected passage (17 words): “We take the z-score of our alpha, on top of the absolute value transform we already made.” This is a practitioner article, not controlled evidence. It suggests testing whether a cheap sign/magnitude basis helps the model represent anonymous heavy-tailed channels.

## Lenses applied

- **Factor:** keep recurrent depth fixed and change width alone to read a local capacity rung.
- **Change time / transfer:** reuse past segment state instead of paying full 20k-row attention cost.
- **Re-encode:** expose nonlinear magnitude while preserving feature direction.
- **Instrument / invert:** break grades out by sequence position and activity regime; define the no-gain and leakage outcomes before spending.

## Proposal cards

### 1. One-layer width rung: H512 → H640 (recommended first)

- **Mechanism family:** local recurrent capacity scaling.
- **Intervention:** keep the native 211-input, one-layer GRU, objective, fitting membership, seed, schedule, checkpoint, and sequence semantics fixed; change hidden width from 512 to 640 only. Use F0/1001 and F2/1002. Reuse the existing H512 B checkpoints only if source, initialization, membership, and recipe hashes match exactly; otherwise count two H512 controls plus two H640 fits (four new fits total).
- **Why it might help:** the partial B300 comparison modestly favors a wider shallow model over a deeper near-parameter match. A single-layer wider GRU retains the low-cost recurrent serving path. A width increase could add capacity without the extra recurrent layer's depth/optimization burden.
- **Assumptions / breakpoints:** the B300 result is not a pure width ablation; H512 already lost to TL3 on a reported cold comparison, so more capacity may overfit the same small set of regimes. H640 increases hidden-to-hidden work by about `(640/512)^2 = 1.56×` as a rough arithmetic proxy, and activation state by about 1.25×; these are not measured runtimes or memory bounds.
- **Predicted observable:** modest positive paired official-WP deltas are plausible if H512 is capacity-limited; a flat or negative result is equally plausible if data reuse/optimization, rather than width, is limiting. Track both targets and predeclared sequence-position/activity bins.
- **Cheapest informative test:** an admitted <=300-second real-shape strict-FP32 profile, then two H640 fits paired against exact H512 controls on F0/1001 and F2/1002. Freeze both H640 standalone and one complete endpoint, 90% TL3 + 10% H640, before reading scores; use the blend only if source-pinned TL3 arrays cover the same rows. This coefficient is a fixed exploratory choice motivated by a prior complementary blend, not an expected gain. Include full-fold prediction export, official scoring, temperature/VRAM checks, and runtime/size projection in the 7,200-second inclusive gate. If controls cannot be reused, admit the four-fit panel only if the full projected cost fits with thermal margin.
- **Falsifier / stopping interpretation:** if both paired cells fail to improve H512 and neither frozen complete endpoint improves its matched comparator, do not spend this allocation on a larger width rung. This is a screen on the H640 recipe, not a verdict against scaling or the wider-model family.
- **Reusable artifact:** source-pinned H640 checkpoint, exact command/config and ancestry manifest, ordered full-fold NPY outputs, paired score receipt, and the native callback/package/runtime check.
- **Evidence limit:** two exposed cells estimate direction only; they do not provide fresh-CV or public-transfer evidence.

### 2. Small segment memory, not full-context attention

- **Mechanism family:** multi-rate causal memory.
- **Source component:** Transformer-XL's cached previous-segment hidden states with stop-gradient across segments; the B300 design's fast local stream plus completed-block slow summaries is a second structural precedent.
- **Intervention:** retain raw rows and the current recurrent state, while adding a small memory of the last 64–128 completed block representations (for example, mean and last-row projections per 32-row block). A current row may read only summaries whose block ended at or before that row. Reset memory at each sequence boundary. This tests whether explicit recent episodic memory adds value beyond the GRU's single compressed state without materializing 20k² attention.
- **Assumptions / breakpoints:** event boundaries are defined only from observable inputs; feature-group identities do not certify physical timestamps or book levels. Mean pooling can erase rare events. A Python-only scorer would not qualify as a native endpoint; state update, reset, and strict prefix visibility must be implemented and checked in the actual callback.
- **Predicted observable:** benefit, if any, should be concentrated in positions where earlier input patterns recur and the current H512 residual has same-sequence temporal structure. No gain is expected if the GRU state already retains the useful information.
- **Cheapest informative test:** first implement a deterministic causal-memory oracle and exact reset/prefix tests; then profile one real sequence. Only admit a matched two-cell-per-arm screen if a native-compatible implementation and full inclusive cost fit the remaining cap. No candidate score should be inferred from auxiliary reconstruction loss.
- **Falsifier:** any future-row visibility, reset contamination, non-determinism, or no endpoint gain after the fixed paired screen closes this implementation. A noisy negative does not disprove all forms of long memory.
- **Reusable artifact:** bounded-memory state schema, causal/reset oracle cases, native callback implementation, memory/runtime report, and full-fold prediction arrays.
- **Priority:** next-wave candidate; the engineering and serving surface is larger than the width rung.

### 3. Signed-log magnitude basis for anonymous inputs

- **Mechanism family:** pointwise nonlinear re-encoding.
- **Source component:** the seeded local passage above uses an absolute-value transform followed by z-scoring in a trading signal workflow. Transfer only the representation idea.
- **Intervention:** keep all 112 raw inputs and 99 canonical event features, but add a small residual projection of a fit-normalized `sign(x)·log(1+|x|/s)` basis for the raw channels; estimate `s` on fitting groups only. Preserve the original channels. This exposes direction and compressed magnitude without inventing physical feature semantics.
- **Assumptions / breakpoints:** large absolute values may be noise, and the 211-input model may already approximate this transform. It is distinct from the previously tested broad geometry bundle, but the bundle's partial negative lowers its priority. The canonical input assembler and native serving path must share the exact fitting-only scale and transform.
- **Predicted observable:** a gain would appear in nonlinear tail/regime performance while overall paired WP stays positive; if tails are irrelevant or existing GRU units already express the transform, quality stays flat.
- **Cheapest informative test:** before training, a target-blind fit-only scale/finite-value audit; if valid, compare one fixed transformed arm against the exact 211-input H512 control on two cells, counting two new treatment fits if controls match or four total otherwise. Keep the 99 event features, membership and all other recipe fields fixed.
- **Falsifier:** no repeatable endpoint contribution in both cells, or improvement only from a change in sequence coverage/normalization, closes this transform recipe.
- **Reusable artifact:** fitting-only scale manifest, transform parity/reset fixture, transformed H512 checkpoint and paired full-fold arrays.
- **Priority:** low; transform and serving parity work may cost more than the likely benefit.

## Allocation recommendation

Screen proposal 1 first. The existing B300 width result gives it a closer empirical bridge than the two new representation arms, and it preserves the native recurrent serving family. Cap it at **two new H640 fits** only if exact F0/1001 and F2/1002 H512 controls are reusable; otherwise allow **four total fits** (two per arm) only if the measured inclusive projection fits the unchanged 7,200 seconds. Evaluate the frozen 90/10 TL3/H640 endpoint without coefficient search if matched TL3 arrays pass preflight. If it cannot fit with the thermal margin, defer the width rung rather than spending the production reserve. Keep proposal 2 for a later implementation-focused branch and proposal 3 as a target-blind diagnostic before any fit.

Use no numerical prior for any untested proposal. Report every paired delta, both targets, exposure, component and training variation separately, and any fixed complete endpoint. All current folds are exposed, so this screen remains exploratory. No public-score forecast follows from a local delta.

**Access record:** no OpenCode, Freebuff, Grok, Kaggle, or paid-provider request was attempted; this source-only assignment did not allocate work to those providers. Actual availability/quota remains unverified and must be checked by the campaign owner before any later admission.
