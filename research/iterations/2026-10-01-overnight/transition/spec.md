# TRANSHEAD-01 implementation and allocation specification

## Frozen treatment definition

Raw columns i0_dv0..3, indices48..51 only. Row0 emits no mark. An event requires current and immediate preceding four-vector each finite and abs<5.19, any exact numeric coordinate change, and current max−min>1e−6. Every raw row advances detector memory, including invalid and neutral rows. Emitted mark is four bits of raw>frozen center, little-endian bit weights1,2,4,8. **No event cap during forecasting.** Reset all causal state at each seq_ix boundary, as existing callback does; do not join adjacent row groups using offline chain knowledge.

68 channels, fixed scaling and ordering:

1. [0:16] one-hot most recent emitted mark (zero before first event).
2. [16:32] one-hot preceding emitted mark (zero until second event).
3. [32] log1p(total marks including current)/log(20001), clipped at20000.
4. [33] log1p(rows since last event)/log(10001), clipped at10000. Before first event age is row index.
5. [34] current-row detector event indicator.
6. [35:51] last event's pre-observation probability vector p; initial value pi.
7. [51:67] last event's one_hot(current mark)−p; initially zero.
8. [67] last event's min(−log(p[current mark]),10)/10; initially zero.

All non-event rows hold channels0:33,35:68 except age33; event flag34 is always current-row. At an event calculate p from counts strictly before the new mark, then expose the current mark and derived features for forecasting, then update counts. kappa16. F uses (counts+16*pi)/(total+16). T first event uses pi; later uses (transition[previous,:]+16*R[previous,:])/(outgoing_count[previous]+16). The common channels are identical between arms. All probability arithmetic float64, stored float32; neural head uses existing AMP convention. Per-sequence total is ≤20000 so float32 native count state remains exact for integers.

## Nuisance fitting

For each forecast fold0/1, use only validation chains outside that fold according to frozen cache/chain_folds.npz. Select minimum validation row-group per allowed chain, deterministic independent of labels; record every chosen row-group and chain. Read only four raw columns, no targets, masks or class labels. All detected marks of representative sequence are eligible. Centers are sequence-equal weighted marginal medians of up to512 evenly spaced detected values per representative sequence, identical center algorithm to old QMARK. Priors are sequence-equal frequency and transition tables with old add-one-distribution smoothing. There is no evaluated-sequence self-fit, and final deployment fits on all valid chains using the same deterministic method. Save raw-column hashes, source stat/footer hash, chain-map hash, selected identifiers, code hash and fitted values.

## Neural adapter

Same n_in211 baseline normalization/projection and GRU. Keep all original weights exactly. Add `head_extra.weight` shape(hidden,68), initialized to zero, to the first head preactivation: GELU(head.0([GRU,z])+head_extra(features)), followed by existing head.2. Reuse all existing auxiliary outputs/losses. Freeze inp and gru; head and head_extra train. Additional projection consumes no RNG after initialization (construct under fork_rng), preserving paired original-head and sampling randomness. No duplicated112feature disk cache: sequence loader computes vectorized extra features from four raw columns and appends temporary68channels; backbone input ignores appended channels.

Isolated trainer wrapper imports original scripts/train_rnn.py and replaces only Sequences and model factory. Original files unchanged. Checkpoint metadata embeds exact nuisance values, arm and feature schema; prediction wrapper recognizes metadata. Parent checkpoints resolve explicitly into original server read-only path; outputs into creative workcopy only. Fixed selection checkpoint model_ep1.pt, not best model.pt. Native support requires matching online count state and adding extra preactivation into head; no unsupported existing exporter call.

## Proposed allocation, pending root freeze

Parents ref_f{0,1}@ep3 seed3 and refs4_f{0,1}@ep3 seed4, immutable hashes supplied by checkpoint inventory. Arms F/T each2epochs, lr5e−4 AdamW wd.01, batch16 chunk2000, gcorr, mask_weight, aux_struct=.3 aux_next=.3, frozen inp/gru, fast existing recipe. No schedule/threshold sweep. Eight head fits; predict fixed epoch1. Root must separately allocate matching two-epoch plain anchors and V1-like d256/c500 partners. Primary fixed equal-quarter serving blend: partner d256H2 + candidate h128seed3 + candidate h128seed4 + partner c500H2, each .25. Compare T vs F and plain anchor. Same per-fold pure validation chains, pooled WP and both targets; seed slot deltas reported with other member fixed to its matched control. Whole-chain paired bootstrap conditions on fits; all evaluation exposed. Default +.0003 pooled and nonnegative both folds to spend final-fit/native budget. Eight fits / no more than eight candidate prediction/score readouts for this first contrast; root owns global ledger.

## Correctness and stop conditions

Independent scalar stream vs vectorized batch, prefix causality with changed future, chunk continuation, exact reset; pathological invalid/neutral/repeated/same-category vectors; matching dimensions/finite/bounds; identical initial model outputs and recurrent state; frozen-backbone gradients absent while both added and original head receive gradients; exact nuisance chain exclusion. CPU/native parity tolerance must be measured before packaging. Reject any future access, target-derived inference input, nonfinite output or unverifiable native implementation. GPU allocation remains root-managed ≤5.5GiB/job, two jobs max. This specification itself does not launch fitting.

## Pre-execution controller/critic resolutions

Promotion additionally requires T versus plain two-epoch reference ≥0 in **each** fold and pooled; F regression alone cannot promote T. No per-epoch labels/scores are evaluated. To preserve the original trainer's next-epoch shuffle, suppressed evaluation consumes its single sequential DataLoader CPU base-seed RNG draw. The trainer's required `weighted_pearson=0` return is explicitly marked an unevaluated placeholder, never an experimental score. Always read fixed `model_ep1.pt`; baseline `model.pt` checkpoint-selection behavior is irrelevant.

Nuisance doubles are embedded in `args.transition_features` and exported through separate JSON, never the native generic float32 weight map. Native detection must compare float32 raw against float32 thresholds 5.19f and1e−6f and must advance its raw detector state even when the backbone sanitizes invalid input. Original parent normalization dimension verified211.
