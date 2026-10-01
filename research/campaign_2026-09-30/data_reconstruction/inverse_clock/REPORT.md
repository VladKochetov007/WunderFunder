# Joint inverse clock: frozen fit-support gate failed

The single registered inverse-clock fit fails its fitting noncollapse gate. Squared reconstructed-increment mass occupies 3.73283 effective joint age-offset cells, below the frozen minimum of 4. The fitted a6 map also reaches its declared total-range bound. This result cannot identify a shared physical time coordinate or support a model/auxiliary change. No threshold was relaxed, no second fit was run, and the maps remain fixed.

Holdout agreement improves substantially against the two registered nonlinear coordinate controls, but it does not override that failure. On 432,796 eligible held-out transitions, symmetric relative RMS discrepancy is .653950 versus 1.159768 for scalar-aligned CDF and .993066 for scalar-aligned log-survival CDF. The candidate beats both controls on 28/32 components. Class 0 is counterevidence: its candidate discrepancy is .988533, worse than log-survival's .834347. These are contemporaneous feature consistency measurements, not forecasts, target scores, inferred seconds or public-transfer evidence.

The revised immutable `preregistration.json` incorporates `../../audit/inverse_clock_critique.md` before fitting/test feature access. It retains the existing 64 fit/32 component-disjoint TRAIN panel, which is already inspected exploratory data. Only `i0_p0..21`, `i0_dp0..3`, `i0_dv0..3`, `a5` and `a6` are read. Price/trade fields conservatively select quiet transitions; the inverse functions use only current/past a5/a6. No targets, masks, sequence IDs, external data, GPU or shared model/trainer sources are accessed or changed. Diagnostic classes come from the previous study's records solely for retrospective reporting.

A jointly quiet transition has all 22 current/past price-code fields unchanged and all eight dp/dv fields unchanged. Origins are excluded. Current lower-endpoint resets are excluded and counted; previous/current lower/upper clipping and negative changes are also excluded, while zero increments remain included. There are 432,797 observed joint-quiet held-out transitions: one is excluded for clipping, zero for negative changes and zero for endpoint reset. The rare nonorigin a6 reset exceptions from the earlier study do not enter this conservative joint-quiet set. The procedure cannot prove that no unrepresented physical reset occurs.

At most 512 evenly spaced eligible transitions per fit group yield 32,768 fit observations across 64 components. Their observed mean CDF increment supplies the scale, `s0 = 70.2482878883`. The constraint is `mean((Δf5+Δf6)/2)=1` on these observed fitting transitions. The old unsupported `f5(.5)=1` anchor is removed. Both maps start at zero, have a segment-slope floor of `.001*s0`, and have total range at most `10*s0 = 702.482878883`. These are declared arbitrary coordinate constraints, not a recovered physical unit.

The piecewise linear maps use the fixed 16 CDF knots `[0,.01,.02,.05,.1,.2,.35,.5,.65,.8,.9,.95,.97,.99,.995,1]`. The squared discrepancy objective uses uniform selected fitting rows. The curvature penalty is 1e-4 times squared adjacent slope changes, weighted by the square root of mean neighboring knot width in normalized map coordinates. SLSQP uses the analytic gradient, linear scale/monotonicity/range constraints, deterministic feasible initialization, `ftol=1e-10` and at most 200 iterations. The small Gram/Hessian matrices, curvature operator, constraint matrix/bounds, scale vector and initialization are in `qp_artifacts.npz`.

The one solver call converges in 68 iterations. Maximum linear-constraint violation is 1.11e-16. The objective is 3.8148709211: discrepancy contribution 3.7595107627 and curvature contribution .0553601584. Map ranges are 326.669419134 and 702.482878883, the second exactly at its range bound. The objective gradient and every fitted height are preserved in `fit_result.json`. A SciPy efficiency warning concerning mixed equality/inequality constraints appears in the raw log; the returned solution is converged and feasible. No solver warning was hidden or used to retry another method.

All support/scale/control-reference values are frozen using fitting features. `fit_support.json` and `fit_result.json` are written before the test-reading loop; the log records `fit_frozen_before_test` before `finished`. `file_timestamps.json` preserves filesystem modification times in this order. The script makes this order inspectable, although filesystem times and an author-written Boolean are corroboration rather than an independent sealed grader.

Support means at least 32 selected fitting rows per knot interval or joint quartile cell. Some extreme intervals are unsupported: a5 intervals below .05 and above .995, plus [.05,.1) having 10 rows; a6 below .05 and [.995,1] having 26 rows. All 16 joint offset cells have at least 725 fitting rows. The fixed two-dimensional cell edges are TRAIN quartiles of each current CDF clock and remain unchanged on test. All empty/unsupported bins and their mass are retained, not discarded from the normalized score.

The noncollapse diagnostics reveal why low agreement error alone is insufficient:

| Diagnostic | Fit | Holdout | Frozen requirement |
|---|---:|---:|---:|
| Paired positive increments | 98.66% | 98.18% | >=90% |
| Largest component squared-mass share | 11.57% | See raw report, passes | <=25% |
| Largest joint-cell squared-mass share | 44.01% | See raw report, passes | <=50% |
| Effective joint offset cells | **3.73283** | 4.02131 | >=4 |
| Unsupported-row fraction | .11597% | .02542% | <=5% |
| Unsupported squared-mass share | .55642% | .00189% | <=1% |

All remaining registered fit median/RMS/knot concentration checks pass, and all registered holdout noncollapse checks pass. Fitting positive-increment medians are .013141/.031086 in the chosen scale, while mean increments are .934812/1.065188 and RMS values 3.293509/3.320827. Held-out medians are .009422/.018432, means .772915/.949503, and RMS 2.900136/3.023589. The distributions remain uneven; all q10/q50/q90/q99/max values and component/knot/cell squared-mass fractions are preserved. The effective-cell failure is a frozen decision gate, not a post-hoc assessment of whether 3.73 is close enough to 4.

The symmetric error is `sqrt(mean((d5-d6)^2)/mean((d5^2+d6^2)/2))`, without an epsilon denominator. CDF/log-survival align the second clock with one positive scalar fitted on TRAIN only, then normalize both increments using their observed fit average. Log-survival clips CDF probabilities at `1-1e-7` before transformation; all methods use exactly the same eligible quiet rows. Publication-row age has identical unit increments by construction and discrepancy zero. That algebraic identity does not establish physical units and cannot validate the inverse maps.

| Diagnostic class | Eligible held-out transitions | Candidate | CDF | Log-survival |
|---|---:|---:|---:|---:|
| 0 | 39,198 | .988533 | .996680 | **.834347** |
| 1 | 24,253 | **.572034** | 1.131217 | 1.004645 |
| 2 | 155,596 | **.597755** | 1.097110 | .776379 |
| 3 | 213,749 | **.602308** | 1.222385 | 1.081427 |

Every component and each of the 16 fixed joint-offset cells is reported. Components 1378, 1855, 1981 and 2614 do not beat both controls. In cells 0, 1 and 4, respectively, CDF, log-survival or CDF is better than the candidate. Several cross-offset cells retain discrepancies near 1.35–1.40. Those cells show that aggregate improvement does not establish one accurate common additive generator over the entire age space.

The durable unit `wnn-inverse-clock-20260930-v2.service` ran exactly once in `wnn-campaign.slice`, with CPUQuota100%, MemoryMax2GiB, zero swap and RuntimeMax60s. It invoked `scripts/limited` with matching one-thread limits. The service ended successfully, ExecMainStatus0; exact properties/timestamps are in `service_exit.json`, the launch argument/resource record in `launch_record.json`, and stdout/stderr in `run.log`. Measured probe wall time is 4.4501s, CPU time 5.2700s (includes Python import work preceding the probe wall timer), peak RSS 1,175,936 KiB. No process exceeded the declared budget. Ruff formatting/checking passed before execution.

Independent raw replay was requested from the campaign audit role using source/result hashes before any supported claim. That remains distinct from the actor's fit and report; no model or physical-time promotion is made here. Repeating the same actor script is an implementation replay on this inspected panel, not fresh-component validation. A future experiment could freeze a larger, independently selected fitting component panel and justify stronger joint-offset coverage from feature-only support before fitting. It would require a new contract and independently fixed test components; the present gate or panel should not be adjusted to admit this result. No further execution is requested or launched.

Key SHA256 identities:

- `probe.py`: `468866b5138a25f7e40967e87fb020f27f1b5b717bb146debbb464120cfd52f2`
- `preregistration.json`: `59211cbe0fa7b7ecfd17637235df2eee754d26ac70fe455a29fca0f83a0813d0`
- `results.json`: `b84947a7dc8b11647152079d45b5a7e3967558f3e2b122622a773f9905666dec`
- `fit_support.json`: `a1c7b541dc414f7ba7c4fe881d623f3dfd29445eb9e25b188e9e7678dcc8bd13`
- `fit_result.json`: `edcf35e946bef35e75b284cc00c4b3da83f7a552c0bb8ee1fdb291f62b3a86e1`
- `qp_artifacts.npz`: `d2a9ea4200db28accbc00458cd53030bb8a19fce1572cb6ae1ab7dbe2af4edf8`

For an authorized independent replay, copy the frozen script and preregistration to a new sibling folder under `data_reconstruction` so the original outputs are preserved, then run the same script using one CPU thread/2GiB/60s campaign-slice limits. The relative parent references keep the original panel intact. Numerical outputs should be compared directly; runtime/resource fields and NPZ container timestamps need not hash identically across independent execution.
