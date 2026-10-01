# cr_reframer — reframing cards and probes (final, saved by controller from the agent's message)

Reference: OOS ft_since s3+s4 mix, folds 0–3, pooled WP 0.68952. Cards committed 20:29 before any probe (cards.json);
R11/R12 added post hoc. Numbers in results.json; scripts scripts/cr_reframer_*.py.

| card | lens | outcome | status |
|---|---|---|---|
| R02 per-sequence level/scale oracles | Factor | in-sample +0.0289 / +0.0141; episode cross-fit −0.021 / −0.012; best shrinkage +0.0008 / +0.0005; first-half→second 0.0000 | supported (negative): oracle headroom ≈97% overfit |
| R01 i1 as second labelled stream | Change search unit | class vocabularies disjoint; i1 overlaps own-class i0 vocab 29/51/41/12%; no i1 state = another sequence's i0 state; 20.6% unknown i1 transitions in class 3 | screen_failed |
| R05 i1 as leader | Instrument | class 3: i1 0.089 code events/row vs i0 0.309 — slower separate instrument | screen_failed |
| R03 censored volume codes | Re-encode | ~10^5 distinct values/column, no max pile-up | screen_failed |
| R12 decoded level reversion / i0–i1 error correction (post hoc) | Instrument | partial corr ±0.04; class-3 LOFO correction +0.00001 ± 0.00017, folds disagree | screen_failed |
| R06 softer public mask | Relax | p²+0.03 weights already put 11.7% of mass on prop<0.01 rows (λ=0.05 asks 12.5%); LB gate probe would only settle the proxy | screen_failed (training lever) |
| R04 event-time multi-horizon labels | Change time | first event resolves 15–22% of D variance; next episode keeps 56–76% | screen_failed |
| R11 event-preserving row thinning (post hoc) | Change time | dropping 21% of rows with no i0 price-code change keeps every kept label exact; untrained incumbent 0.7445→0.7438 on 80 seqs (c2 −0.0046, c3 +0.0027), pred corr 0.995–0.998; view average +0.0002 | insufficient_evidence |
| R07–R10 | Mine surprises/Transfer/Invert/Domain op | prior evidence already below falsifiers | not probed |

Where not to dig: sequence-level latents (FiLM, context encoders, online recalibration); price-level or i1 information;
the scorer mask (weights already match). Binding constraint: estimation variance on ~28 effective episodes per class-3 sequence.
Proposed 2-GPU-h experiment: R11 row-thinning augmentation (loader-only: per sequence w.p. 0.5 drop step≥99 rows with
unchanged i0 price codes at rate q~U(0.1,0.4); thin x/y/mask/propensity; recompute event features and aux labels; pad with
zero-weight rows; predict unthinned). Arms seeds 3,4 × folds 0,1, 4 epochs; gate pair Δ ≥ +0.0008, both folds > 0,
class 3 ≥ −0.0003; kill Δ ≤ 0 or class 2 < −0.002. Expected +0.0002…+0.001, P(pass) ≈ 0.2.
