# Shared experiment contract

All ten cards are **implemented; receiver acceptance pending**.
Owner of preparation: Ivan / Codex. Receiving operator and B300
allocation: unassigned. The current plan contains 40 fresh fits (four per
implemented card), with zero executed and no reused controls. No receiver check,
profile, test, inference, model construction, submission or running-job mutation
has been recorded. Publication does not automatically launch jobs. The exact
server-side sequence is documented in [B300_COMMANDS.md](B300_COMMANDS.md) and
[the cards07–10 family guide](B300_CARDS07_10.md).

Objective mode: competition. Development evaluation is deterministic but repeatedly
exposed; it is not an untouched test of competition transfer. No predicted LB gain.

## 1. Source and data

- Integration base: `research-team` at
  `7e4074314a62d0907c9608f9220cf72958708a2d`.
- Cards01–07 use the exact canonical plain Transformer runtime
  from PR75, commit `8626000a4b965763a70011a0291a891aa6535d7e`, loaded on the
  receiver through a separate clean, detached read-only worktree. PR81's runtime
  admission rejects a wrong or dirty checkout and computes source identity from
  both runtime and packet Python files. The existing `src/`, `scripts/`, training
  configs, optimizer, training loop, validation and checkpoint code are unchanged.
- Cards08–10 use the similarly pinned PR79 GRU runtime. Family selection occurs
  before importing model code; it cannot silently reuse the Transformer checkout.
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
- Keep all 112 ordered raw features. Fit normalizers and trainable preprocessing
  on eligible train sequences only. `normalization_scope=all_fit` takes
  precedence over a legacy sampling-count key. The canonical normalizer still
  uses every fit row.
- Card01's estimator amendment: select 64 evenly spaced row positions from each
  of all 10,325 train sequences (660,800 rows total), normalize these values
  using the canonical all-fit-row moments, and calculate 16 linear empirical
  quantiles per feature. The sample is fixed and label-independent; the knots
  are not exact quantiles over every training row. Valid/test data do not enter
  preprocessing. The receiving host creates one shared artifact bound to train
  membership, split, source and canonical moments. All arms consume that artifact.
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

Preparation and fit admission bind inputs, split and preprocessing ancestry.
[receiver_checks.py](runtime/impact81/receiver_checks.py) checks initial-function
agreement, finite new-parameter gradients, causal-prefix invariance with active
residual branches, fresh-sequence reset, chunk carry and strict model-state reload.
It runs through [worker.py](runtime/worker.py) after runtime admission. The separate
canonical full-size profile checks model/optimizer/scheduler/RNG reload; final
export and scoring establish complete endpoint coverage and WP. Loss masks are
inherited unchanged. These stages do not certify CPU callback equivalence or
submission readiness. Restore the complete canonical resume state at its documented
safe boundary. Receiver acceptance is pending; local model execution remains prohibited.

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
For G the historical source requires L4. Cards08–10 make the explicit device-only
amendment for both arms and use fresh paired B300 controls;
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

Each implemented card uses seeds **204101 and 204102**, the same official file
split, two fresh controls and two fresh treatments: **4 new fits per card**.
Cards01–10 therefore have **40 planned fresh fits**, zero executed and zero reused
old controls. There are no teacher fits or teacher caches. Cards07–10 add16 new fits under their implementation amendment. All40 remain
prepared and unexecuted. This is not a dispatched queue
or an automatic compute allocation.

The implementations are in this PR. Before each cell's launch, the receiving
owner must bind actual PR81 source/config/data/runtime identities, run IDs, owner,
GPU allocation, numeric timeout/resource bounds and artifact destination. No
profile or fit is authorized automatically by the packet. Four fits per card are
an initial scientific panel, not four retries. Failed attempts remain in the
ledger; retry only after diagnosis under a new attempt ID and bounded budget.

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
and storage locations: config/source/environment; receiver check and approved
profile receipts; progress and terminal receipts; data/split/ancestry manifests;
exact scored predictions, metric and all contrasts; fixed final weights plus
reproducible resume state under a declared retention budget. See
[MIDAS handoff instructions](midas/README.md) and the packet's
[evidence exporter](midas_evidence.py). MIDAS route registration is metadata, not
execution; the current [route preview](midas/route-preview.json) reports no
remotely launchable worker. The evidence archive may omit weight bytes, in which
case producer hashes are not byte verification. Large weights/datasets stay out
of Git. No deletion of old user artifacts.

## 7. Submission boundary

Transformer gains first justify a separate teacher-transfer or serving study;
none of these large models is declared CPU-ready. Training-only auxiliaries can
be removed from an exported predictor, but exact-predictor parity is still required.
For any eventual ZIP: Linux x86-64/Python 3.11, one vCPU, 16 GB RAM, 60 minutes for
the full test set, <=20 MB uploaded ZIP, no network/GPU. Profile on an available
server before full validation; compare against canonical TL3 and check existing
submission hashes. PR76 already supplies a different TL3+LSTM ZIP: this portfolio
does not rebuild, replace or upload it. No CPU acceptance has been measured here.

## Cards04–06 implementation amendment

[IMPLEMENTATION_04_06.md](IMPLEMENTATION_04_06.md) records the explicit experiment-local engine for auxiliary losses, card05's value-copied control and12 unique shared parameter sets with24 cache histories, and MIDAS evidence support. The existing canonical source and native recipe are unchanged. Fresh receiver receipts are required at the new source revision; earlier running work must retain its original checkout.


## Cards07–10 implementation amendment

[IMPLEMENTATION_07_10.md](IMPLEMENTATION_07_10.md) records the remaining16 configurations, separate PR75/PR79 runtime admission and exact device-only GRU amendment. The GRU recipe remains FP32/AdamW on one B300; Transformer remains BF16/BF16x9 on two B300s. Source-equivalent horizon WMSE is pooled over targets per horizon. Card08 uses matched clean traversal and checkpointed private noise RNG. Card10 raw-label prevalence is a train-only prerequisite for either arm. All40 fits and actual MIDAS evidence import remain pending.
