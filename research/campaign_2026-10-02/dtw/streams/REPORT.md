# dtw/streams — input streams and sequence similarity (agent final message, saved by controller)

Panel: pooled OOF of the D256 pair p = (parent + head-FT)/2, folds 0–3, scored rows ≥ 99. Baseline pooled 0.69124,
public-like 0.66413 (= public D256 0.6641), per class .807/.778/.740/.438. Candidates scored as full sums +
(n_scored/n_sampled)(candidate − baseline) on a 1/3 row sample; every correction cross-fitted (3 folds fit, 4th scored).
Scripts `scripts/cr_dtw_streams_{leadlag,i1table,build,linear,ceiling,probe,knn,warp,level,common}.py`.

H1 (DTW lead-lag i1→i0): decoded mids via TRAIN class move tables; open-end DTW over last 20 bins of 20 rows
(z-norm, band 6): dtw_pend, warp lag, DTW/Euclid ratio, window corr; plus fixed-lag i1 returns, i1 move since last i0
move, error-correction ec_h. Co-movement class 2: 0.38/0.72/0.85 at B20/B100/B500, i1 leads by 20 rows (0.187 vs
0.136); class 1 0.27/0.53/0.67; class 3 ~unrelated (0.02/0.08/0.13). After projecting on pD, partial corr with the
residual < 0.05; consistent remainders only in class 2 (dtw_pend +0.032, ec_200 −0.045, 4/4 folds). Cross-fitted ridge
(13 features, λ 0.5): pooled −0.00008, mix −0.00011 (class 2 +0.00059). In-sample ceiling +0.00060 / +0.00071.
HGB probes all lose (−0.0016…−0.0024). Verdict: falsified; the GRU already extracts the i1 lead.

H2 (DTW-kNN per-sequence calibration from rows < K, K 2000/5000, 8 binned channels; library same class, other folds):
oracle (uncentred per-sequence slope) +0.0061 pooled / +0.0067 mix (class 3 +0.0137); best of 36 retrieval configs
+0.000009; DTW = Euclid = channel means. Falsified (extends E-211).

H3 (class-3 quiet-run time warps, TRAIN, 150 seqs, 6,617 runs of 64–1024 rows): DTW/Euclid 0.16 vs 0.49 Poisson null —
runs are time-warped copies; trades back-load before the move; early speed does not predict run length (0.01) nor
direction (0.002) nor |yD| (0.004). Target-neutral.

H4 (GRU forgetting sequence drift; trailing means of its own pD/pS): ridge −0.00061 pooled, all 4 folds negative.

Recommendation: no GPU run from streams. If forced: 6 causal inputs (decoded i0/i1 tick returns 20/100 rows, ec_200,
dtw_pend) on the alpha-0.7 recipe, fold 0, gate +0.0005; forecast ≤ +0.0002.
