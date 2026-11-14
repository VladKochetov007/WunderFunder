# Runtime recipe and source audit

Prepared source only. No model execution, training, local test, inference or
profile was performed while writing this packet. CUDA numerical admission and
the complete two-rank profile are required on the receiving B300 host.

The receiver derives its runtime from PR75 commit
`8626000a4b965763a70011a0291a891aa6535d7e`, applies `runtime.patch`, and copies
`payload/wnn/training/sequence_perturbations.py` into `src/wnn/training/`.
Existing repository `src/`, `scripts/`, optimizer, scorer and fit entrypoints
are not changed by this PR. All arms use the same derived runtime identity.
The added module is included by native `source_identity`, which enumerates
`src/**/*.py`.

## Deliberate differences from the frozen plain Transformer

| Stage | Required match | Deliberate change |
|---|---|---|
| Fitting membership and order | Original train-only source-qualified split and rank sampler | None |
| Raw schema | `wnn.data.FEATURE_COLUMNS`, 112 fields | Only i1 indices 52 through 103 can be replaced in stale arms |
| Normalization | Native all-fit clean-input statistics | None; augmentation occurs after `move_and_validate_batch` |
| Target, weighting, denominator | Original labels and standalone WMSE, clean target masks | None |
| Teacher, initializer | Label-only, paired seeds and native parameter initialization | No teacher and no new parameters |
| Input stream | Full sequence loaded before native chunks | Once per training batch, iid or block sample-hold of i1 |
| Attention | Original Q/K/V, RoPE, causal rolling context and cache writes | Train-only additive `-inf` for selected old keys before softmax |
| Optimizer and schedule | Native Muon/AdamW, updates, clipping, schedule | None |
| Checkpoint | Fixed configured E4 endpoint and native RNG provenance | None |
| Export and score | Clean native model `.eval()` and full original validation | Both perturbations are disabled |

The optional raw keys are `input_perturbation` and
`model.history_perturbation`, each exactly `none`, `iid`, or `burst`.
Explicit `none` survives config normalization; absent optional fields remain
absent for backward-compatible control construction. Both perturbations active
together are rejected. The registered fit panel requires Transformer models.

## i1 burst staleness

The canonical `FEATURE_COLUMNS` comprehension and `docs/data_overview.md`
identify the 52 i1 fields at `52:104`; the module verifies this mapping at import.
The 52 i0 fields and eight `a0..a7` fields remain original. This is an intervention
on the explicit i1 channels, not proof that all information about i1 has been
removed: auxiliary channels may retain correlated information.

For a sequence with `T` rows, row 0 and the incomplete final 128-row block are
always observed. Let `B=floor((T-1)/128)`. Burst selects `floor(B/4)` distinct
complete blocks uniformly; iid selects exactly `128*floor(B/4)` distinct eligible
rows uniformly. For `T=20000`, both select 4,992 rows out of 19,968 eligible rows,
giving each eligible row marginal dropout probability 0.25. Adjacent sampled
blocks can join into longer bursts. The tail of 31 rows is protected in both.

For dropout indicator `d[t]`, compute `s[t]=max{j<=t:d[j]=0}`. Replacement is
`x'[t,52:104]=x[s[t],52:104]`. Row zero guarantees this set is never empty.
The mask is drawn without reading features, labels, validation or future values.
Consequently replacement is causal, including across chunk boundaries. There is
no interpolation, zero fill, target transform or lost-row loss reweighting.

Corruption uses a private device generator with SHA-256 of
`(version, seed, epoch, rank, batch_start)` as seed. It does not consume global
model RNG or alter batch order. Native whole-sequence boundaries remain intact.
Paired iid and burst arms match coverage, not age of the held observation: the
longer age under bursts is the intended scientific intervention. This corruption
can destroy genuine predictive signal, so clean endpoint quality is decisive.

## Historical-key dropout

Only keys already in the incoming cache are eligible. All keys in the current
chunk and the most recent 32 cached keys are protected. Native rolling context
trimming occurs first. For `P` retained cached keys, eligible count is
`E=max(P-32,0)` and the exact selected count is `floor(E/4)`. Iid samples that
many distinct keys; burst samples a uniform starting position on a circular
array of eligible ages and selects an interval of that size. A wrapping interval
appears as two spans at the eligible-array boundary. Each eligible key has the
same marginal dropout probability `floor(E/4)/E` in both arms.

The selection is independent per sequence and layer invocation, and shared
across heads and all current-chunk queries. As a query advances, native context
expiry can remove additional selected or retained keys; the **cached-key** count
is matched, not an exact count per query after rolling expiry. No future or
self/current key becomes eligible. Chunk size remains fixed at 256 in training:
this intervention is intentionally chunk-boundary-dependent, and does not claim
equivalence under arbitrary training chunk schedules.

The implementation builds an additive causal attention bias with `-inf` on
selected keys, preserving the native 8-element aligned backing stride required
by efficient SDPA. The chosen backend is explicit (`math` or `efficient`) with
no fallback. Softmax renormalizes retained keys; there is no inverted-dropout
scaling. It does not mutate supplied K/V tensors or the stored cache. Deeper
layers can still receive activations influenced by the lower-layer perturbation,
as in normal stochastic training.

History masks consume native device RNG, which the original checkpoint and
engineering-reload machinery already records. Compiled stochastic kernels may
use a different random sequence from eager execution; equality between eager
and compiled training draws is not claimed. The prepared GPU probe and native
profile must establish replay under the actual compiled recipe. In eval mode,
or with mode `none`, the block calls the original `LocalCausalAttention` path.

## Static checks actually performed

- Parsed the added module and four patched native sources with Python AST;
  no package or model was imported.
- `ruff format --check` and `ruff check` passed for the added module.
- `ruff check` passed for patched source snapshots with `wnn` explicitly marked
  first-party, avoiding partial-snapshot import classification artifacts.

These checks do not establish CUDA kernel eligibility, throughput, VRAM usage,
gradient correctness, fitted quality or serving compatibility of future weights.
