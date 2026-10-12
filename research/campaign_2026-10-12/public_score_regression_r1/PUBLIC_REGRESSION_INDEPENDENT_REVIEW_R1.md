# Independent review of public-score regression R1/R2

Verdict: retain this as an exposed historical diagnostic only. It does not support a numerical public-score forecast for the current ZIP11 candidate, does not select a public-optimized blend, and does not change the primary local cross-validation ranking.

## Coverage and predictor audit

I independently parsed the frozen raw roster and endpoint map at base commit `e084c22d`. Both contain the same 42 IDs, with 38 scored results, three timeouts, one check error, and three reported-but-unshown events. Censored and unseen outcomes remain missing; none are zero-filled. `10tq248b` retains portal filename `submission.zip` and separately records alias `c20261004_combo_s3_i16`.

The endpoint map has 11 scored rows with some local metric, but these form four non-interchangeable evidence strata: three receipt-bound fold1 float predictions on one shared component, one manifest-reported fold1 float metric without checkpoint digest, five historical 368-group metrics across two declared components, and two conditional EMA member0 proxies on one component. Twenty-seven scored rows have no comparable local feature. Three visible censored rows have local sibling proxies, which are correctly excluded from the score target. All local features were exposed or retrospectively selected. The historical 368-group teacher/initializer/checkpoint ancestry is not receipt-verified.

The fold1 local predictions are FP32 research outputs, while the corresponding public packages use native Q8VNNI; they are exact-weight local predictions only before native quantization. The timeout H388 all-data package is paired in the map with a different fold1 sibling metric, so that value is not an exact target feature. No exact current ZIP11 local score is present. The mismatched H390 fold1 prototype cannot stand in for ZIP11. The two EMA6 entries are conditional member0 reconstructions, not exact served-ensemble scores.

The R1 fitter violated its frozen family-mapping rule by combining package alias text with present member identifiers. That can classify `0catwwxg` as chunked from its package alias despite its member source identifiers indicating legacy since/EMA. R2 corrects source precedence and applies alias fallback only when member identifiers are absent, including the H230 alias mapping. Independently parsing the endpoint map finds three alias-only H230 rows affected: `20o154uy`, `27t6d4ur`, and `1n7tmww1`. R2 is a post-result parser correction, so it is a sensitivity analysis, not new confirmation.

Source/member overlap correctly groups fold1 IDs `2ebpj2er`, `0srim121`, and `0hikjxpr`; the Q8 z1c, prior i16 z1c, z1c/z4 pair and z4 member also share one component. R1 and R2 report 18 declared source/member components, not 18 verified independent teacher/initializer ancestries. D256 exact-string repeats `2m6fzkn0`/`03wwfmuv` are grouped, while identifiers with and without epoch suffixes fall into separate components. Missing older package receipts prevent concluding whether those suffix variants are distinct weights; this is a residual possible dependence, not evidence of independence.

## Independent recalculation

I independently reconstructed the fixed splits, features, source groups and alpha-10 ridge from the endpoint CSV. R1 was recalculated in pure Python; R2 was recalculated with a separate NumPy implementation. Both match the stored metrics to numerical rounding.

| Frozen diagnostic | Ridge MAE / RMSE | Comparator MAE / RMSE |
|---|---:|---:|
| R1 grouped components: 38 rows, 18 groups | 0.00100003 / 0.00149920 | train mean 0.00174949 / 0.00231423 |
| R1 chronological purged: 17 rows, 9 groups | 0.00107763 / 0.00151733 | recent five 0.00078941 / 0.00117498 |
| R2 grouped components: 38 rows, 18 groups | 0.00108392 / 0.00168171 | train mean 0.00174949 / 0.00231423 |
| R2 chronological purged: 17 rows, 9 groups | 0.00097353 / 0.00139957 | recent five 0.00078941 / 0.00117498 |

The fixed regression improves on the arithmetic mean in these retrospective slices, but the simple recent-five baseline is better on the chronological rows. Choosing between the two after viewing these results is itself exposed selection. The 17 chronological rows are only nine declared source groups, and displayed portal times do not establish when scores became available. The result therefore cannot be used as a prospective transfer correction.

The earlier 368-group local study itself declares 11 dependent public submissions and 21 proxy variants, with nested-LOSO RMSE 0.0017954055. That number is selected retrospective local calibration evidence, not a fresh public-upload holdout; it does not validate a ZIP11 public forecast. The local-feature subsets here have only one, one, two, and one declared groups respectively, so no separate grouped local-to-public regression is supported.

## Evidence binding and remaining requirements

R1 result SHA-256 is `69f3efc3cfbb4ed445ea26ebfe3747b6d7d76c885baa0a83b4d8c16a7f3a229f`; it binds the endpoint CSV and R1 contract hashes, but its JSON does not bind the fitter hash or execution command. The recorded R1 snapshot is commit `85cfa5c8`, with code SHA-256 `cfc5c8068dbdf2470ec01b2e28d67deebc77c23d9805bd7bd9ea16abae6e7965`; preserve the missing self-contained execution binding as a reproducibility limitation. R2 result SHA-256 is `622fb07b31dd8fe00100a4e94087433348e55911e6275dcc5cb00048be3defe8`, binds the R2 contract and endpoint CSV, and has a pre-run frozen spec/code/input pin at commit `2e36d981`. No source, model, label, split, penalty, or outcome changes were introduced by R2.

A candidate-specific public forecast would need, at minimum, an exact ZIP/member/hash and native-Q8 serving receipt for ZIP11; a matching held-out local score from the exact current weights on complete source-qualified groups with teacher, initializer and propensity ancestry exclusions; and prospectively frozen public outcomes for assessing forecast error. Public timestamp/score-availability provenance is also needed for a true time-forward replay. Until those exist, report public performance as unknown for ZIP11 and continue ranking candidates on valid local evidence. This conclusion does not reject the model or research direction based on lower historical public scores.
