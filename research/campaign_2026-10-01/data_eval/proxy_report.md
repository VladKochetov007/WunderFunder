# Public-like local proxy (Task A, data/evaluation, 2026-10-01)

CPU only. Scripts: `scripts/de_proxy_cache.py` (per-sequence WP sums), `scripts/de_proxy_study.py`
(subsets, class mixes, nested calibration), `scripts/de_public_proxy.py` (tool). Outputs: `proxy.json`
(summary, tool proxy specs, ledger entries), `proxy_study.json` (full matrices), `seqsums_*.npz`,
`proxy_tool_example_xf_since_pair.json`.

Inputs: `cache/valid_targets.npz` (targets, `is_scored`, `market_class` = target-quantum class from
`wnn.data.market_class`), fold manifests `research/campaign_2026-09-29/artifacts/splits/fold{0..3}.json`
(component = undirected connected component of the feature-only successor graph; identical to
`cache/chains.npz` chains for valid), the reference members `cache/pred_c20260929_ft_since_s{3,4}_f{k}_valid.npy`
(fold k, equal mean, float64 mix rounded to float32, then clip), the 11 public packages from
`scripts/campaign_public_calibration.py` (scored on the 368 hash-holdout sequences only; every member was
fitted without them). Sanity checks passed: reference pooled WP 0.689523 (report: 0.6895232), fold WPs
match, and the old 21-variant nested LOSO reproduces 0.0017954 (pair) and family-held-out 0.0025767.

"Chain length" below = size of the sequence's component across train and valid. "Pure" = component has
no train member.

## 1. Class mix and chain structure of valid

| population | class 0 | class 1 | class 2 | class 3 |
|---|---:|---:|---:|---:|
| train sequences (10,606) | 12.3% | 11.9% | 32.9% | 43.0% |
| valid sequences (1,873) | 21.2% | 18.5% | 23.0% | 37.2% |
| valid single-sequence chains (168) | 8.3% | 11.3% | 14.3% | 66.1% |
| valid chains of length <= 3 (581) | 13.6% | 10.7% | 19.3% | 56.5% |
| share of metric weight on valid scored rows | 28.2% | 15.8% | 8.3% | 47.7% |

1,678 of 1,873 valid sequences are in pure components; components are almost always single-split. Class 3
has shorter chains (mean component size 4.6 against 6.6 to 8.0; 15.9% singletons against 3.5 to 5.6%).
Valid and train class mixes differ far beyond sampling noise, so the public mix is not implied by either.

## 2. Clean four-fold reference by subset (pooled Global WP, 1,873 sequences, 579 components)

| subset | n seq | WP | | subset | n seq | WP |
|---|---:|---:|---|---|---:|---:|
| all | 1873 | 0.6895 | | chain len 1 | 168 | 0.5783 |
| class 0 | 398 | 0.8069 | | len 2 | 199 | 0.6254 |
| class 1 | 347 | 0.7761 | | len 3 | 214 | 0.6597 |
| class 2 | 431 | 0.7387 | | len 4-5 | 391 | 0.6899 |
| class 3 | 697 | 0.4331 | | len 6-9 | 509 | 0.7127 |
| pure | 1678 | 0.6996 | | len >= 10 | 392 | 0.7479 |
| linked to train | 195 | 0.5941 | | | | |

Within class, chain length hardly matters: class 3 len 1 0.4234 against len >= 2 0.4349; class 0
0.7787 against 0.8079 (14 sequences); class 2 0.7583 against 0.7373. Reweighting len >= 2 sequences to the
single-chain class mix gives 0.5801, the same as the actual singles (0.5783). **The teammate claim
(single chains 64% class 3, score 0.59 against 0.74 for long chains) reproduces (66%, 0.578 against
0.748), but it is a class-composition (Simpson) effect, not a chain effect.** Short chains are harder only
because they are class 3.

## 3. What class mix would reproduce the public level

Reference WP if the scored population had another class mix (sequence shares, within-class rows unchanged):
valid mix 0.6895; class-3 weight x1.25 0.6729; x1.43 **0.6622**; x1.5 0.6583; x2 0.6343; train mix
0.6440; short-chain (<=3) mix 0.6190; single-chain mix 0.5792 (the isolated-like mix is far too harsh).

### Implied public class mix (`scripts/de_class_mix.py`, `class_mix.json`)

Reweighting rule matches `scripts/screen_score.py reweighted()`: each sequence of class c gets weight
mix_c / local_share_c (sequence shares of the scored panel); means unchanged. Families: 1p = valid
proportions for classes 0 to 2 with the class-3 share s free; 2p = class-2 and class-3 shares free (0:1 at
the valid ratio); 3p = free simplex.

| source | fitted mix (c0, c1, c2, c3) | class-3 share uncertainty | fit quality |
|---|---|---|---|
| (a) clean 4-fold reference = 0.6622, 1p | **0.183, 0.160, 0.198, 0.459** | component bootstrap SD 0.025, 95% [0.41, 0.51] | exact by construction |
| (a) same, incumbent +0.002 / +0.004 better than reference | c3 0.465 / 0.471 | same SD | |
| (a) same, incumbent 0.002 worse | c3 0.453 | same SD | |
| (b) 11 packages on holdout, 1p, pair | 0.177, 0.154, 0.192, 0.477 | bootstrap SD 0.042, 95% [0.39, 0.55]; LOSO-fold fits 0.476 to 0.479, lineage-held-out fits 0.474 to 0.484 | LOSO 29.4e-4, family 30.2e-4, lineage 38.8e-4 |
| (b) same, single scenario | 0.177, 0.155, 0.192, 0.476 | same | LOSO 26.8e-4, family 26.3e-4, lineage 31.5e-4 |
| (b) 2p | 0.02, 0.02, 0.80, 0.16 | bootstrap SD of c2 0.33, c3 0.18 | not identified |
| (b) 3p | 0.00, 0.56, 0.00, 0.44 | degenerate | not identified |

The two independent 1p estimates agree: the public level is consistent with valid class proportions and a
**class-3 sequence share of about 0.46 to 0.48** (95% roughly 0.40 to 0.55), instead of 0.37 locally.
Two caveats. First, the holdout panel overstates final packages through dev-valid chain neighbours,
which pushes (b) toward more class 3, so (a) at 0.459 is the preferred point. Second, as a calibrator
the level-matched mix predicts the package *ordering* worse than an affine map (LOSO 29.4e-4 against
17.95e-4), because no mix fixes the slope. The largest residual is the organizer baseline at -74e-4: it
does better on public than any mix allows. So the class mix explains the level and is stable across
folds and lineages, but the 2p and 3p shapes are not identified. Use it as the public-like weighting of
deltas, not as a stand-alone forecaster of absolute public scores.

## 4. Do subset or class-reweighted proxies predict public better?

Libraries: old21 = the existing 7 weightings x lambda {0, 0.02, 0.05}; new33 = 11 chain/class weightings
declared before looking at calibration results (single-chain, len <= 2, <= 3, >= 4, pure, 1/len,
1/sqrt(len), single-chain class mix, class-3 x1.5 and x3, classes 2+3 x2) x 3 lambdas; all72 adds 18
post hoc variants (see below). Proxy selection is repeated inside every outer fold. RMSE in 1e-4:

| calibrator, library | pair LOSO | pair family | pair lineage | single LOSO | single family | single lineage |
|---|---:|---:|---:|---:|---:|---:|
| affine, old21 (current) | **17.95** | **25.77** | 51.76 | **7.78** | 16.71 | 21.67 |
| affine, new33 | 14.94 | 25.28 | 46.37 | 15.42 | 15.35 | 29.10 |
| affine, old21+new33 | 14.94 | 26.18 | 46.37 | 15.33 | 15.56 | 26.73 |
| shift-only, old21 | 16.73 | 17.10 | 48.17 | 12.64 | 24.29 | 27.22 |
| shift-only, new33 | 15.30 | 15.84 | 42.46 | 17.30 | 27.21 | 31.08 |
| affine, all72 incl. post hoc | 12.61 | 20.84 | 45.43 | 5.29 | 4.90 | 21.12 |

Lineage = hold out one of the three literal-parent lineages (baseline/tv/s4/ema4, ab18, since/ema6).

**Verdict: no pre-declared subset or class-reweighted library robustly beats the current calibration.**
It improves the pair scenario (17.95 to 14.94 LOSO; family 25.8 to 25.3) and worsens the single scenario
(7.78 to 15.4). All libraries fail across lineages (21 to 52). The post hoc library looks best but was
designed after seeing these residuals, which nested selection cannot correct.

Best fixed variants (no selection, lambda 0), RMSE 1e-4, and the slope of public on local:

| variant | n seq | pair aff LOSO / fam / lin | pair shift LOSO / fam | single aff LOSO / fam | slope pair / single | mean public - local |
|---|---:|---|---|---|---|---:|
| holdout (all 368) | 368 | 29.5 / 23.1 / 44.6 | 33.5 / 34.5 | 32.5 / 19.9 | 0.63 / 0.66 | -0.032 |
| class-3 x1.5 | 368 | 27.9 / 23.2 / 40.7 | 30.2 / 31.1 | 30.9 / 18.5 | 0.66 / 0.70 | -0.002 |
| train class mix | 368 | 24.6 / 21.9 / 34.7 | 27.3 / 28.6 | 26.7 / 16.1 | 0.69 / 0.73 | +0.013 |
| single-chain class mix | 368 | 21.3 / 25.6 / 29.1 | 22.9 / 24.1 | 22.6 / 13.7 | 0.74 / 0.79 | +0.078 |
| chain len <= 3 | 106 | 15.9 / 17.5 / 20.9 | **14.3 / 14.8** | 16.0 / 12.1 | 0.86 / 0.92 | +0.044 |
| chain len <= 2 | 71 | 10.0 / 19.4 / 10.1 | 11.7 / 12.8 | 14.7 / 48.1 | 0.87 / 0.93 | +0.068 |
| single chains | 38 | 22.0 / 55.8 / 24.1 | 29.8 / 30.8 | 23.5 / 70.3 | 0.67 / 0.74 | +0.125 |
| train_unlike lambda 0.05 (historical pick) | 245 | 12.0 / 27.2 / 17.1 | 18.1 / 21.6 | 7.8 / 10.1 | 0.76 / 0.82 | +0.032 |
| post hoc: <= 1 dev-valid neighbour | 91 | 9.5 / 20.0 / 8.8 | 15.0 / 18.0 | 9.7 / 18.0 | 0.80 / 0.86 | +0.069 |
| post hoc: same, class-matched to holdout | 91 | 9.9 / 13.6 / 14.5 | 15.0 / 17.9 | 9.6 / 10.1 | 0.79 / 0.83 | -0.040 |
| post hoc: len <= 3 class-matched | 106 | 17.6 / 18.7 / 27.9 | 17.4 / 18.0 | 18.4 / 18.4 | 0.81 / 0.84 | -0.041 |
| post hoc: len >= 4 at short-chain mix | 262 | 23.3 / 28.4 / 32.7 | 29.0 / 30.8 | 25.1 / 18.4 | 0.67 / 0.71 | +0.073 |

Readings:
- Upweighting class 3 helps a little and monotonically (29.5, 27.9, 24.3 for x1, x1.5, x3; single-chain
  mix 21.3). Class mix explains part of the public ordering. It does not explain most of it.
- Short chains predict the ordering of packages better than class mix alone, and the advantage survives
  class matching: len <= 3 class-matched keeps a slope of 0.81 to 0.84, while long chains reweighted to
  the short-chain mix keep 0.67 to 0.71. The post hoc variants point to one mechanism: holdout sequences
  whose component has dev-valid members (sequences the final packages trained on) inflate local
  differences between packages. Slope is 0.57 for sequences with such neighbours and 0.80 to 0.86 without
  them. The same direction, noisier, appears for train-only ab18 against the organizer baseline
  (class-matched +0.0095 +- 0.0083 on sequences with >= 3 dev neighbours, +0.0013 +- 0.0046 with none;
  public -0.0015), so it may not be only leakage. It is best read as "long valid chains are less
  public-like".
- None of this changes the *level* story. Class-matched no-neighbour sequences score even higher than
  the full holdout (public minus local -0.040 against -0.032). The level gap is a class-mix/regime effect.

**Shared errors.** The 11 packages come from at most three literal-parent lineages. Nested residuals move
together within families: for example, the s4 family's errors are -8.6, -9.4, -12.9 and the EMA6 pair's
are +12.9, +23.3 (old21, pair, 1e-4). The effective number of independent observations is about 3 to 5.
RMSE differences under about 5e-4 between proxies are not resolvable, and every library's lineage-held-out
RMSE (0.002 to 0.005) is the honest transport error for a new lineage.

## 5. Implication for forecasting clean four-fold deltas

The chain-excluded four-fold manifests already remove the neighbour exposure that compresses the
historical slope (0.63 on the full holdout). Neighbour-free and short-chain subsets give slopes of 0.77
to 0.92, all fitted on level differences dominated by early-versus-final packages. Conditional working
rule: expected public is about 0.6622 + tau x Delta_public_like, with tau about 0.8 and a sensitivity
range of 0.63 to 0.92. Delta_public_like is the `public_like_class_mix` delta below. tau for a new mechanism is not
identified, and the lineage-held-out error of 0.002 to 0.005 still applies to absolute forecasts.

## 6. Tool: `scripts/de_public_proxy.py`

```
WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/de_public_proxy.py \
  --pair 0 cache/pred_c20260929_ft_since_s3_f0_valid.npy,cache/pred_c20260929_ft_since_s4_f0_valid.npy cache/pred_CAND_f0_valid.npy \
  --pair 1 ... [--boot 2000] [--output out.json]
```
REF and CAND are comma lists of full `[1873, 20000, 2]` exports, averaged in float64, rounded to float32
and clipped. Rows: step >= 99 and is_scored within the fold's evaluation sequences. The tool reports
pooled delta, component-bootstrap SD, 95% interval and P(delta <= 0) (seed 271828, whole components
resampled) for:
1. `all_eval_rows` (unchanged gate quantity; matches campaign_score pooled WP);
2. `public_like_class_mix`: the recommended public-like weighting, `{"class_mix": [0.1832, 0.1597,
   0.1984, 0.4588]}` relative to each panel's own class shares (on all valid this equals class-3 weight
   x1.43);
3. `short_chain_len_le3`: the best pre-declared fixed historical proxy. Its sample is small (581
   sequences), SD is about 2x;
4. `train_class_mix`, `single_chain_class_mix`: mix sensitivities (the latter is the isolated-like
   class-3 stress [0.08, 0.11, 0.14, 0.66]).

Weights depend only on class and component metadata. Example (`proxy_tool_example_xf_since_pair.json`,
cross-fitted s3/s4 blend weights against the equal pair, 4 folds): all rows -0.06e-4 +- 0.47e-4;
public-like mix +0.04e-4 +- 0.56e-4; len <= 3 -1.64e-4 +- 0.96e-4; train mix +0.06e-4 +- 0.61e-4;
single-chain mix +0.31e-4 +- 0.87e-4. Runtime about 12 s for 4 folds.

Second example (`proxy_tool_example_d256_head_f0.json`, fold 0, since pair against the equal-thirds
pair + head-FT distilled GRU256): all rows +4.31e-4 +- 5.25e-4 (screen report: +4.3e-4 +- 5.4e-4);
public-like mix +4.33e-4 +- 5.60e-4; len <= 3 +10.06e-4 +- 5.09e-4 (P <= 0 0.013); train mix
+5.59e-4 +- 5.37e-4; single-chain mix +7.12e-4 +- 6.61e-4. Under the implied mix, the class-3 gain and
the class-0 loss roughly cancel, so the public-like delta equals the all-rows delta. The +0.0007 seen
under the isolated mix depends on that too-harsh mix. Only the short-chain subset favours d256 clearly,
and that subset is noisy and post hoc.

Recommendation: keep gating on all rows. Also require the `public_like_class_mix` delta to be positive,
because that weighting is the public-like point estimate. Under that weighting, classes 0 to 2 carry 43% of the metric weight
instead of 52%, so a gain confined to them shrinks by roughly 15 to 20% in public terms. Read the len <= 3 delta as a noisy, independent check.
