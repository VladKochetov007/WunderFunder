# Structured causal perturbations on B300

**Prepared; no profiles, fits, model tests or inference executed by this task.**
H-B300-STRUCTURED-PERTURBATIONS-20261114 / E-B300-STRUCTURED-PERTURBATIONS-20261114.
Owner of preparation: Ivanhacker13/Codex. Execution needs a named receiver and a
fresh allocation. Published PR URL is recorded in `publication.json` after creation.

The question is whether **correlated missing information during training** improves
clean-data Transformer quality more than independent omissions of the same size.
This packet places that intervention at two points: auxiliary raw input and old
attention history. They share fresh unperturbed anchors and the identical runtime.
It does not change the repository's shared training pipeline or add optimizations.

## Frozen panel

| Arm | Training intervention | Fits |
| --- | --- | ---: |
| `clean` | No perturbation | 2 |
| `stale_iid` | Independent positions of stale i1 | 2 |
| `stale_burst` | 128-row blocks of stale i1 | 2 |
| `history_iid` | Independent old attention keys hidden before softmax | 2 |
| `history_burst` | Circular block of old keys hidden before softmax | 2 |

Each arm uses seeds **208401, 208402**. Input screen: six new fits. History screen:
four new fits plus the two exact clean anchors from the input screen. Total ten
Transformer fits, zero GRU/student/teacher-target fits. There is no automatic
follow-up or combined-corruption arm. Do not substitute older PR75/97 controls.

All recipes copy PR75 `8626000a4b965763a70011a0291a891aa6535d7e`:
24 layers, width 768, 12 heads, context 1024, chunk/gradient horizon 256,
batch 128 per rank, accumulation four chunks, **two B300 per fit**, BF16/BF16x9,
FP32 master parameters, Muon plus AdamW, LR 0.0003, weight decay 0.01,
warmup 5%, cosine floor 0.1, grad clip 1.0, four epochs / 3280 updates.
The inherited compiled efficient-attention path remains enabled. This is an E4
screen, not a new 30-epoch scale run. Preserve all epoch snapshots but compare
**fixed E4**, not the best validation epoch. JSON files retain all exact options.

## Exact interventions

**Input staleness.** Raw112 i1 is columns `[52:104]`; i0 `[0:52]` and auxiliaries
`[104:112]` remain unchanged. Of 156 complete blocks starting at row 1,
choose 39 blocks of 128 rows: **4992/20000 = 24.96%** missing positions per
sequence. The IID control selects exactly 4992 of those same 19968 eligible
positions. Row zero and the last 31 rows are observed in both arms.
For missing row t, replace all 52 i1 fields with the latest observed i1 row at
or before t in the **same sequence**. Consecutive chosen blocks can merge into
longer outages. The sequence is transformed before chunking, so carry persists
through TBPTT boundaries. Seeds depend on fit seed, epoch, rank and batch start;
the perturbation does not consume the global model RNG.

This is a missing-update robustness hypothesis, not a proven label symmetry.
Same missing fraction does **not** mean same information age; burst−IID tests
the joint effect of temporal clustering and resulting staleness. Auxiliary
features may contain redundant information: this does not remove every possible
trace of i1 from the dataset. Normalization is fitted on original fitting data.
Labels, masks, target magnitudes and loss denominators are unchanged.

**Attention history.** For each layer, batch item and training call, protect all
keys from the current chunk and the most recent 32 cached keys. Let e be the
number of older cached keys. Remove exactly floor(e/4) keys either as a uniform
subset or a uniformly positioned circular block. The circular block can wrap
into two chronological intervals; it gives each eligible key the same marginal
removal probability. Apply this mask before softmax together with the original
causal context mask. Keep the cache itself intact and retain original RoPE positions.
The current key always remains, so no query has an empty attention row.

The two variants match cached-key count and per-key marginal exposure, not lost
attention mass. Sliding context can exclude additional keys separately. This is
transient access corruption; it is not a permanent KV outage. With multiple
layers, downstream representations naturally reflect lower-layer perturbations.
History randomness uses the native saved CUDA RNG; compiled replay is an admission
check. No claim of equal stochastic draws across different chunk sizes is made.

Both perturbations are training-only. Evaluation uses original inputs and the
unchanged native attention path; parameter count and checkpoint shapes are unchanged.
A large Transformer remains a teacher candidate, not a ready CPU submission.

## Evaluation and decision

Fit: 10325 train-source sequences, excluding 282 train sequences linked to evaluation components. Zero valid
sequences in fitting; clean normalizer ancestry. Split SHA256:
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.
Evaluate all 1873 official-valid sequences / 3528273 scored rows using
`global_wp_clip2_v1`, independent export plus official scorer. This valid is
**repeatedly exposed development data**, not new independent confirmation.
Anonymous chronology limits future-regime claims despite linked-group exclusions.

Primary comparisons: each burst arm against clean, equal mean of two paired seed
deltas. Allocation threshold: mean at least **+0.001 WP and both deltas positive**.
Burst−IID diagnoses whether the structured recipe earns its cost; if that contrast
fails, a burst benefit cannot be attributed to its structure. IID−clean is also
reported. All six contrasts and all endpoints are retained, with no best-seed
selection. No Holm or formal significance claim is made in this exploratory screen.

The readout uses the native component-paired bootstrap, shared draws across seed
endpoints and contrasts. It reports per-seed intervals and a descriptive two-seed
mean, without an interval for that mean. Native `p_le_0` is a bootstrap sign fraction,
not a calibrated p-value. Intervals describe evaluation-component sensitivity for
these trained models; two seeds cannot establish training-variance significance
or public transfer. A zero-crossing interval alone does not terminate a direction.
Negative clean-validation deltas narrow this strong recipe. Mixed seeds or a
small gain are inconclusive. Separate allocation is needed for new source-qualified
folds, student distillation and eventual measured CPU serving.

## Contents and evidence chain

- [PR.md](PR.md): reviewable contract, rationale, risks and fitting inventory.
- [recipe.json](recipe.json), [configs](configs/): frozen scientific cells and admission identities.
- [GPU_COMMANDS.md](GPU_COMMANDS.md): explicit Linux/B300 stages, budgets and collection.
- `prepare_runtime.py`, `runtime.patch`, `payload/`: isolated source preparation from
  PR75; no edits to the existing shared pipeline. `runtime_manifest.json` freezes
  exact resulting source files and is independently checked at admission.
- `run.sh`, `receiver.py`, `readout.py`: native fit/export/scorer handoff, source and
  resource gates, retained failure logs and the complete fixed-E4 comparison.
- `perturbation_probe.py`, `probe_invariants.py`: **unexecuted server checks** for
  causal carry, masks, cache protection, neutral evaluation, efficient/math
  attention, gradients, compiled RNG replay and streaming behavior.
- [sources/PR_AUDIT.md](sources/PR_AUDIT.md): duplicate search, concrete negative
  results, pinned sources and independent statistical criticism.
- [sources/INDEPENDENT_REVIEW.md](sources/INDEPENDENT_REVIEW.md): independent
  source/CLI/metric review, repaired probe gaps and remaining GPU evidence needs.
- [sources/LITERATURE.md](sources/LITERATURE.md): primary-paper motivation and limits.
- [RUNTIME_DESIGN.md](RUNTIME_DESIGN.md): implementation hooks and source mapping.
- [midas/README.md](midas/README.md): metadata-only handoff of independently graded
  endpoints. MIDAS is not evidence that any GPU is scheduled or running.

Receiver CPU/GPU compatibility, fit completion, clean score, CPU cost, public gain
and distillation benefit are unknown. No new model weights exist for this packet.
