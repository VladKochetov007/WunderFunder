# S38 mirror-symmetry diagnostic proposals R1

**Frozen before inspecting sampled feature values.** This records the complete candidate set and analysis rules for the diagnostic. It is not a claim that any map is a lawful symmetry.

## Question and scope

For the documented 112 feature coordinates, do any small schema-anchored bid/ask permutations or price-axis reflection maps preserve the input feature distributions on a deterministic set of complete fitting sequences? These are necessary input checks only. Anonymous coordinate names and target-law uncertainty prevent this diagnostic from establishing a valid label transform or authorizing augmentation.

The source description in `docs/data_overview.md` identifies bid and ask halves and the `p`, `v`, `dp`, and `dv` groups. It explicitly warns that level indices are identifiers, not an ordering by price or depth. `a0`–`a7` are additional fields with no mirror semantics in the supplied schema. All maps below preserve `dv` and `a0`–`a7`; no rank reversal is proposed.

## Candidate maps

Represent each map as `x'[j] = sign[j] * x[perm[j]]` for each 112-wide input row.

| Map | Exact transform | Interpretation and limits |
|---|---|---|
| `side_swap` | Within each instrument, exchange p indices 0–10 with 11–21 and v indices 0–10 with 11–21; leave dp, dv and a unchanged. | Same-index bid/ask permutation. It is an involution but has no target transformation justified by the docs. |
| `price_axis_reflection` | Apply `side_swap`; negate every p coordinate and all four dp coordinates for both instruments; keep v, dv and a values unchanged. | Conditional reflection candidate. If p/dp are signed price coordinates and labels are exactly `t0 = F - ask`, `t1 = bid - F`, the implied target action is `[t0,t1] -> [t1,t0]`. The target definitions are undisclosed, so this action is an algebraic conditional, not a label rule. |
| `price_sign_only_negative_control` | Negate all p and dp coordinates for both instruments, without exchanging bid/ask; preserve v, dv and a. | Deliberately tests the generic sign-only shortcut warned against in section 3. This is not presumed to be a valid order-book reflection and is not eligible for augmentation. |

For each map, require the signed permutation to be exactly involutive and every transformed sampled value to remain finite. The map is stateless and rowwise: transform every row of each selected sequence, preserve row order and sequence boundaries, and leave metadata, `need_prediction`, targets, masks, warmup length and reset logic untouched. No temporal state is carried across a reset.

## Frozen sample and summary

- Use only `train.parquet` groups in `research/campaign_2026-09-29/artifacts/splits/fold0.json` `fit.train`.
- Select 32 distinct feature-only connected components uniformly from the sorted available component IDs with `numpy.random.default_rng(2026101338)`. Select one eligible `("train", row_group)` uniformly from each selected component using that same generator. Read the complete row group/sequence; do not subsample or reorder rows.
- Read only the 112 feature columns via `SequenceFile.read(group, targets=False, mask=False)`. Do not inspect train labels, validation inputs, evaluation groups, model predictions or score outputs.
- For each map and each selected sequence, report finite counts; the fraction outside that sequence's same-coordinate empirical 0.1%–99.9% interval; the fraction outside its observed per-coordinate min/max; and the maximum absolute difference across quantiles `[0.01,0.05,0.50,0.95,0.99]`, normalized by `max(q99-q01, |q50|*1e-6, 1e-8)`. Summarize per-feature and per-component distributions with medians, p90/p95 and maxima. Give each component equal weight in component-level summaries.
- A predeclared strong marginal mismatch for this exact map is: in at least 16 of 32 components, more than half of the 112 coordinates have normalized maximum quantile displacement greater than 1.0. This is a screen for this recipe and sample, not a population-level test or a universal cutoff. Report all continuous measurements even when that condition is not met.

Matching marginal summaries, finite values, or an existing model's low equivariance residual cannot prove joint data symmetry or target correctness. No model inference is part of this CPU diagnostic. Because the label law and the meaning of anonymous coordinates remain unknown, no outcome from this screen alone can authorize augmentation.

## Sources, identity and cost

The proposal is bound to section 3 of `research/unified/ideas/FIRST_PRINCIPLES_EXPERIMENTS_20261013.md`, the feature schema in `docs/data_overview.md`, and the frozen F0 component split. Their SHA256 values and the split/source identity are in [PROPOSALS_R1.json](PROPOSALS_R1.json). The planned command uses `scripts/limited` with 1 CPU, 1.5 GB maximum memory, zero swap, and 120 seconds. No training, GPU, inference, provider or paid action is permitted.
