# Shared experiment contract

Status: **proposal-only portfolio**. Owner of preparation: Ivan / Codex. Receiving
operator: unassigned (the team may claim a selected experiment). No fit, profile,
test, inference, model construction, submission or running-job mutation accompanies
this PR. The user asked for more ideas and review of recent PRs. The quoted team
request informs the handoff; it does not authorize us to operate their servers.

Objective mode: competition. Development evaluation is deterministic but repeatedly
exposed; it is not an untouched test of competition transfer. No predicted LB gain.

## 1. Source and data

- Integration base: `research-team` at
  `7e4074314a62d0907c9608f9220cf72958708a2d`. Most recent experiment code is in
  separate draft branches; this old integration base does not contain it all.
- Read-only reference configs are copied byte-for-byte under `sources/`, with
  origin URLs and SHA256 in `recipe.json`. They are **historical source examples**,
  not executable configs for these new experiments.
- GRU parent: PR79, head `67c3bd818ea243177eeb1dc789cd3d145eba0e81`, K16 config.
- Transformer parent: PR75, head
  `8626000a4b965763a70011a0291a891aa6535d7e`, **plain** config (not QK-RMS).
- Official file split, `train` only for fitting: 10,325 whole 20,000-row sequences;
  `valid`: 1,873 sequences, 37,460,000 raw rows and 3,528,273 scored rows.
- Split receipt SHA256:
  `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.
  The receiving host must bind actual data hashes, feature order and membership to
  this receipt; this document does not certify local dataset bytes.
- Keep all 112 ordered raw features. Fit normalizers, quantiles, auxiliary class
  counts and all trainable preprocessing on eligible train sequences only.
  `normalization_scope=all_fit` takes precedence over a legacy sampling-count key.
- IDs are shuffled identifiers, not timestamps. Audit source/component overlap
  and duplicates using the canonical data contract. Whole-sequence separation
  alone does not prove absence of related components or chronological shift.
  Report any discovered overlap; do not silently relabel this exposed official
  development split as clean component-held-out confirmation.

## 2. Online prediction and training semantics

At row t the model can use observed inputs through t and carried state, never
future inputs or any true lagged target. Advance the stream on every row, including
warmup and unscored rows; reset at whole-sequence boundaries. Preserve recurrent
state / per-layer KV state across chunks and detach gradients at the configured
horizon. No new chunk-level reset, changed sequence shuffle, train/valid carry or
cross-sequence pooling is introduced by a card.

The reference training mask is `all_after_warmup`; the evaluation mask is
`is_scored & need_prediction`, with rows 0..98 excluded. These are distinct masks.
Auxiliary losses define their extra eligibility explicitly. Future **training
labels** may be loss targets within the same fitting sequence; they never become
inputs or teacher forcing. No parameter updates on valid/test streams are proposed.
Here “online” refers to streaming state and live reporting, not permission for
test-time supervised learning.

Receiver-only checks before fits: input/split and ancestry identity, causal-prefix
invariance, sequence reset, chunk/callback equivalence, loss masks, finite gradients,
resume state and fixed-model metric replay. Restore optimizer/scheduler/RNG/sampler
and stream position together, or restart a safe documented sequence boundary.
Local execution of these checks remains prohibited.

## 3. Matched templates and precision

| Setting | T: Transformer template | G: K16 GRU template |
|---|---|---|
| Model | H768, 24 layers, 12 heads, FF multiplier 4 | H256, 2 layers, K16 input stem |
| Endpoint | fixed final epoch 4 | fixed final epoch 6 |
| Context / gradient horizon | 1024 / 256 rows | full carried state / 1000 rows |
| Batch / accumulated chunks | 128 / 4; world size 2 | 32 / 1; world size 1 |
| Optimizer | Muon .95 momentum, 5 NS steps; AdamW roles as source | AdamW |
| LR / WD / grad clip | 3e-4 / .01 / 1 | 3e-4 / .01 / 1 |
| Schedule | warmup_cosine_v2, 5% warmup, floor .1 | same |
| Precision | BF16/BF16x9; preserve source FP32 master/loss/state semantics | FP32 as measured parent |
| Objective | standalone_labels_wmse_v2; KD/hints zero | same |

These are fixed starting recipes, **not asserted optimal hyperparameters**.
For G the historical source requires L4. A future B300 implementation must make
an explicit device-only amendment for both arms and use fresh paired B300 controls;
never compare a new B300 treatment to the old L4 score as the primary contrast.
Do not silently convert G precision or copy T's Muon into G. Conversely, do not
reintroduce strict FP32 model compute for the B300 Transformer continuation.
No quantization, new queue manager, kernel tuning or throughput project is included.
Hardware placement and optimization remain with the receiving team.

Both arms start from scratch in every proposed screen; no ancestor checkpoint is loaded.
Card05 specifies a matched special block-copy initialization in both arms.
Common tensors must receive identical initialization, input order and augmentation
RNG streams within each pair. Allocate new-module RNG separately so merely adding
an auxiliary head does not change the shared backbone initialization. Keep LR,
loss normalization, update cadence and successful-update endpoint equal. Count
extra operations and parameters honestly: equal epochs do not imply equal FLOPs.

## 4. Fit accounting and endpoint selection

Each card: seeds **204101 and 204102**, same official file split, two controls +
two treatments = **4 new fits**. No seed/worker reservation or launch is made here.
The proposed ten-screen inventory is **40 new fits**, not a dispatched queue or
an authorized compute budget. No new teacher fits or teacher caches are required.
Select one mechanism first. Reuse is zero by default; exact control reuse needs a
separate source/config/runtime/ancestry/seed receipt and revised accounting.

The selected experiment gets its own implementation draft PR, exact final source
and config hashes, per-arm run IDs, owner, numeric timeout/resource budget and
artifact destination **before launch**. Those values are intentionally unassigned
here; this catalog cannot itself be submitted to a trainer. Four fits are an
initial scientific panel, not four automatic retries. Failed attempts remain in
the ledger; retry only after diagnosis under a new attempt ID and bounded budget.

Use full train and full valid; fixed E4/E6 is primary. Best-valid checkpoints and
intermediate curves are diagnostics, not replacements for the endpoint. If an
arm fails or misses the endpoint, preserve it as incomplete. Freeze the comparison
before outcomes; no immediate tuning of auxiliary coefficients on full valid.

## 5. Metric, uncertainty and promotion

Use canonical `global_wp_clip2_v1`: clip target/prediction to [-2,2], weight by
absolute clipped target, compute weighted centered Pearson over all eligible
rows separately for t0/t1, then their prescribed mean. Do not average per-sequence
correlations or change metric weights to those of an auxiliary loss. Preserve the
grader's degenerate-denominator handling. Record full counts and row identities.

Return both seed deltas, per-target metrics and their descriptive mean. Working
practical screen: mean delta >= **0.0005 WP**, both seed deltas positive, and no
per-target loss worse than 0.001 without explicit tradeoff review. This is an
allocation heuristic, not a significance test or automatic submission rule.
Mixed/smaller results remain uncertain; a zero-crossing interval alone closes no
direction. A clean negative narrows the exact recipe.

There are ten proposed primary contrasts; report how many were actually selected
and tried, including failures. Whole-component paired resampling of fixed
predictions (10,000 draws, seed 204111) may describe evaluation sensitivity only;
if a trustworthy component map is unavailable, label sequence bootstrap as weaker
and do not imply independence. Two seeds on one exposed split cannot establish
training variance, fold robustness or statistical significance. Do not bootstrap
two fit scores to manufacture it. Strong leads need separately budgeted,
source-qualified component/fold confirmation and a frozen serving endpoint. No
repartition of inspected data becomes a fresh holdout.

## 6. Progress, failures and artifacts

Adopt the event contract visible in Semyon's PR77–80. On launch, every 30 minutes,
each epoch and terminal exit record: owner/admission/run/arm IDs; UTC event and
**producer** timestamps; pinned source/config/data/split/normalizer/ancestry hashes;
phase; attempted/successful updates, examples and epoch; primary/auxiliary losses;
all LR groups, preclip grad norm, clipping/nonfinite/skipped counts; latest complete
full-valid WP/t0/t1 with the checkpoint/update it belongs to; elapsed time, peak
RAM/VRAM and actual device/runtime. Log card-specific diagnostics too.

A fresh mirror timestamp does not mean fresh training progress. Preserve the
actual step/log timestamp and freshness lag; stale means **status unknown**, not
running or failed. Do not mix metric timestamps across cells. Publish paired
endpoint readouts only after both models, exact row coverage and source identities
pass the canonical receiver. Keep failed exports separately from failed fits.

Return immutable evidence and weights archives separately with SHA256, file sizes
and storage locations: config/source/environment; progress and terminal receipts;
data/split/ancestry manifests; exact scored predictions, metric and all contrasts;
fixed final weights plus reproducible resume state under a declared retention
budget. Large weights/datasets stay out of Git. No deletion of old user artifacts.

## 7. Submission boundary

Transformer gains first justify a separate teacher-transfer or serving study;
none of these large models is declared CPU-ready. Training-only auxiliaries can
be removed from an exported predictor, but exact-predictor parity is still required.
For any eventual ZIP: Linux x86-64/Python 3.11, one vCPU, 16 GB RAM, 60 minutes for
the full test set, <=20 MB uploaded ZIP, no network/GPU. Profile on an available
server before full validation; compare against canonical TL3 and check existing
submission hashes. PR76 already supplies a different TL3+LSTM ZIP: this portfolio
does not rebuild, replace or upload it. No CPU acceptance has been measured here.
