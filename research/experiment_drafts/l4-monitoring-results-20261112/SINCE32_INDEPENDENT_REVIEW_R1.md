# Since-event weighting: independent evidence review (R1)

## Decision

The two-seed E6 screen is strongly positive and merits a narrow confirmation wave. Both paired deltas exceed the PR82 prioritization rule (mean gain at least 0.0005 and both seeds positive). The current comparison is descriptive: the fitted since32 models use source 390c846cbdacd8e0e06e6079dd3b04312fa6bcc5 (identity 29fb45164667e69a71d4be2c696731d9824884d2d51674f37d2a633bd17d9511), while the archived K16 controls use the earlier source identity b9eddeb9a0733b8efa5a11bed24d03de53f94b83b03d72b5db09ab1adcd63188. The base K16 path has a bitwise parity fixture and fresh raw K16 replays reproduce the historical scores, but there is still no exact 390c846 no-weight fit as a source-matched control.

## E6 comparison

Candidate values below are from the completed producer E6 full-validation monitor events (global_wp_clip2_v1), before the pending canonical R3 prediction re-export.

| Seed | Since32 E6 WP | Historical K16 WP | Paired delta | Fresh raw K16 replay |
|---|---:|---:|---:|---:|
| 203301 | 0.6863060127 | 0.6830194976 | +0.0032865151 | 0.6830194983 |
| 203302 | 0.6859035983 | 0.6826293716 | +0.0032742267 | 0.6826293715 |
| Mean delta |  |  | **+0.0032803709** |  |

The fresh raw K16 replay scores differ from the historical canonical controls by approximately +7.5e-10 and -9.7e-11, respectively. That independently supports the numerical stability of the baseline. It does not make those replays an exact same-source control for the since32 fits.

## Fit, split, and ancestry audit

Both since32 fits completed the declared six epochs and 38,760 updates. They share source identity 29fb4516…, data SHA 4987ed5f263cf3c655714a7af3e792605716f6cb256c1c96474eb46ae9869323, split SHA 0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c, normalizer identity 39fc1b53a6b1a6e35cb5887e1a45232570ceedc8a57a3f429b688c5141ce2408, and moments SHA be4935168c8b4000b5b92e26cef38c8ab44b6d8844469bf874efebf759b8031f.

The run receipts record 10,325 fitting members, 1,873 full-valid evaluation members, zero valid fit members, and an all_fit normalizer over 10,325 sequences. The normalizer fingerprint names datasets/train.parquet; reference ancestry is none, teacher is null, and the cold labels-only config guard excludes pretraining and parent initialization. The fixed E6 checkpoint is the endpoint; full-validation events are report-only. The valid score covers 3,528,273 scored rows and the official membership hash.

The treatment config otherwise matches the K16 recipe: GRU 256×2 with K16 stem; six epochs; batch 32 and 1,000-row chunks; FP32; AdamW at 3e-4, weight decay .01, clip 1; 5% warmup and cosine-v2 floor .1; seeds 203301/203302. The only config additions are since_weight=32 and since_floor=0.25.

## Mechanism and interpretation

The implementation derives age from the current and prior raw input rows inside each sequence using causal_since_i0px_torch; it forms a cumulative last-event index and does not read labels, future rows, or seq_ix chronology. The row multiplier is max(0.25, 1 / (1 + age / 32)): weight 1 at age 0, 0.5 at age 32, and the 0.25 floor by age 96.

The code applies this multiplier to the main label-loss numerator only. Its denominator remains the original unweighted mass sum(abs(clipped_target) * mask), and the official validation metric remains unweighted global_wp_clip2_v1. Thus the treatment both favors more recent price-event rows and lowers total label-gradient mass relative to the unweighted objective; it is not a weighted mean renormalized by the weighted mass. That is the implemented contract and should be the interpretation of the experiment.

## Comparator and claim limits

The historical controls are explicitly frozen as source identity b9ed…, tied by the PR80 parity record to base commit 336b1c436cf9f7a7d2196fb29eb0f55ca50e5da2. That parity fixture reports bitwise-equal K16 initialization, state tensors, full forward, and chunked forward. The since adapter source is commit 390c846…, based on the PR80 successor source. Its since path is conditional and inactive at weight zero; its added GRU-Muon support is not used by these AdamW fits. These facts reduce the practical source-drift concern, but do not substitute for a fresh zero-weight fit under the exact 390c846 source.

Fresh raw K16 replays under a separate EMA-era source identity (33574f496a2640e1a4c14690c8858eeaad5075e21d5b96042b746d35ad55810a) have the same seeds, split, and normalizer identity, and reproduce the old scores to about 1e-9. Their data SHA differs because the report-only monitor contract differs. Use them as context and baseline-stability evidence only, not as the source-matched causal control. The EMA-weighted fits did not complete E6 and are not part of this comparison.

The official chronological valid set is the correct primary evaluation partition, but it has already been repeatedly exposed during development. Two seeds are a small screen; the similar positive deltas do not establish statistical significance, generalization to private data, or a public-score gain. The reviewed candidate scores are monitor values until the canonical R3 exported-prediction/scorer attachment confirms them.

## Recommended next step

Run a narrow matched confirmation under source 390c846…: first fit the two since_weight=0 K16 controls at seeds 203301 and 203302 with every other config, data, normalizer, runtime, and fixed E6 endpoint unchanged. Recompute each paired delta with the canonical scorer. If the positive screen survives, use a fresh seed pair for both zero-weight K16 and since32 arms; do not broaden the weight/floor grid before that comparison.

For an optional no-fit prediction check, a predeclared elementwise 50:50 blend of same-seed since32 and fresh raw K16 predictions is admissible as an exploratory ensemble diagnostic. Require exact E6/raw-weight manifests, source/model/checkpoint hashes, official split and full valid member order, 3,528,273 scored rows, and canonical scorer parity. Report per-seed blend scores against both members and prediction/residual correlations. Do not search blend weights or select a public claim on this exposed validation set; any result using the EMA-era raw models remains contextual because their source identity differs.

## Artifacts reviewed

- sequence-gru-followups-r2/MANIFEST.json and candidate config hashes.
- live-dashboard/native/l4-followups-since32_k16_s203301-fit/ and ...s203302-fit/ run metadata, terminal status, and E6 monitor events.
- l4-stem-successors-20261111/BASE336_PARITY.json, README.md, and CONFIG_MANIFEST.json.
- l4-live-and-old-results-r1.json historical canonical K16 readout.
- Fresh raw K16 E6 receipts in live-dashboard/native/l4-ema-k16_s203301-fit/ and ...s203302-fit/.
- Frozen runtime source at commit 390c846…: sequence_objectives.py, sequence_loop.py, features.py, and sequence_config.py.
