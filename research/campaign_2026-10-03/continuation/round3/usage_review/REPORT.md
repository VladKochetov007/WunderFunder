Independent verdict: PASS the frozen report's scoped CPU instrument and sampled
arithmetic. No supported material implementation/report error was found. FAIL
any stronger interpretation as official full WP, fresh confirmation, beneficial
corruption, missing information, clean retention beyond32 identical model-input
rows, or GPU parity. The author already disclaims those interpretations.

I independently read the frozen REPORT/PREREG/results/verification and measurement
source, then executed array-only reconstruction and causal feature recomputation.
No model forward was executed; no future washout artifact was read. All128
registered actor artifacts matched their frozen hashes before and after the
audit. Live source hashes, baseline checkpoint identity and the actual prediction
cache/sidecar identity were independently checked. The stored cache rows match
the associated prediction file. Sources and actor files were read only.

The frozen acceptance matrix is in PREREG.json; executable checks are audit.py
and domain_diagnostic.py. Each query's six moments are reconstructed from its
stored target/prediction with weight abs(clip(target,-2,2))/pi, where pi=12/n.
Both target correlations are averaged after pooling. The 288 queries reproduce
the frozen uniform-without-replacement draw exactly from actual parquet masks,
sequence IDs and steps;24 distinct components and46,572 eligible after511 rows
form the declared domain. Estimated design row mass equals46,572 by construction,
not because every outcome was scored. Component selection is a fixed metadata
hash panel, not a population probability sample.

Independent moment maximum absolute difference is7.28e-12; floating reduction
order explains it. The WP scores are:

| Arm | Reconstructed sampled-domain WP | Standalone grader helper WP |
|---|---:|---:|
| Identity | .6404556770959638 | .6404556770959637 |
| Raw recomputed | .6597818241968942 | .6597818241968942 |
| Packed i0 | .6599083717651774 | .6599083717651772 |

The actual grader helper is root [utils.py](../../../../../utils.py),
weighted_pearson; it uses clipped target magnitudes and centered float64 moments.
It has no inclusion-probability argument. Because EVERY sequence selected12
queries, repeating each query n_eligible times gives the required n/12 weights
up to the same positive factor12, which cancels in correlation. This executable
check calls the actual helper on558,864 repeated rows. Those repetitions are a
mathematical weighting device, not additional observed data or official-domain
coverage. The active pipeline is raw112 plus99 causal event features, matching
actual build_inputs and the inspected n_in211/online_norm=false/raw_only=false
configuration. The feature computation executed event_features_torch, not the
actor's windows helper or its model.

The frozen1,000 component draws match seed614205 exactly. Applying them to
independently reconstructed moment totals reproduces the paired raw WP interval
[-.0125740644,.0581805960] and packed interval[-.0120236921,.0572495883]
within1e-12. These are conditional component-bootstrap intervals over this
exposed panel and fixed query draw. They neither quantify the complete within-
component query sampling variance nor establish a full-test effect.

An additional exact-domain diagnostic uses root utils.GlobalAccumulator on ALL
stored baseline rows, without any new forward:

| Baseline domain | Selected rows | Exact fixed-panel WP |
|---|---:|---:|
| CPU, entire selected panel scored domain | 47,650 | .6613566075790701 |
| CPU, same after511 domain estimated above | 46,572 | .6619506180037982 |
| Cached backend, same after511 domain | 46,572 | .6615038162600098 |

Thus sampled identity WP underestimates its known CPU domain value by.021495.
This is a useful sampling diagnostic, not an arithmetic bug or proof that the
paired intervention estimate has the same bias. The raw/packed altered outputs
exist only at query windows; there is no reconstructed exact full-domain
intervention score. Even these complete baseline panel scores are NOT official
full-validation/test scores. All panel targets were already exposed; independent
code and an independent query draw do not create fresh scientific confirmation.

Input and donor checks, reconstructed rather than trusting summary flags:

| Property | Verified coverage | Meaning |
|---|---:|---|
| Original current raw input | 288/288 | Raw suffix protected |
| Original complete current model input | 172/288 |116 current engineered inputs differ |
| Original complete protected33-row model-input suffix | 0/288 | Clean32-row retention test absent |
| Packed window agrees with recomputed causal inputs | 0/288 | Packed history is instrument-inconsistent |
| Chosen donor reproduced from raw/std only | 288/288 | Target-independent matching implemented |

For all288 interventions I reconstructed features from row0 using the actual
torch feature function, asserted identical pre-splice prefix, i1 and auxiliary
channels, and compared ALL211 model-input channels on[t-32,t]. The raw input
suffix is preserved, but every engineered suffix differs somewhere. Return-
seam deltas and retained clocks/held values give a concrete mechanism. At172
queries with the complete CURRENT input unchanged, raw-arm response RMS is
.0579274667 (versus.1182893991 at the116 changed-current queries). This supports
dependence on preceding causal MODEL-input history in this callable CPU model;
it does not isolate how far back recurrent influence was carried.

Donor AST arguments are exactly raw,t,std; its body and RMS helper contain no
target, mask, class or response lookup. Source serializes donor choices before
loading targets/predictions, and independent minimum-distance computations match
all288 serialized choices. Offline mask use is confined to the preregistered
query draw. Matching is nearest among available lags, not conditional
exchangeability or an empirical support test. Large seams and unchanged i1 can
still make recomputed raw perturbations economically OOD. Zero unmatched mass
only establishes computational availability. A response, even with current
input equality, cannot prove useful missing information.

The seven reported failures are preserved cross-backend cache bounds, NOT CPU
reference/identity failures: sequence indices1,2,13,14,16,19,20 exceed maxabs.02;
13 and20 also exceed RMS.005; none fails correlation.999. Worst maxabs.0298483,
worst RMS.00581282. All CPU baseline/reference comparisons pass unchanged
atol2e-5+rtol2e-5; maximum difference7.15256e-7. All stored predictions/states/
arrays are finite. The report's existing first-query future canaries and72-window
verification are frozen author evidence, not independently rerun model forwards
in this review. This audit checks their artifact identity and CPU arrays; it
does not newly establish deployed submission/GPU parity. Same-panel CPU/cache
WP differs by about.000447, which is compatible with a small aggregate score
difference despite real pointwise gate failures; it does not cure those failures.

No live correction is warranted from these checks. Next admissible experiments
are a newly frozen uniform query draw on these SAME exposed components to test
sampling robustness, and a separate clock-reset/washout instrument requiring
exact equality of every model input over the protected suffix, reporting any
unmatched mass. Seam/support matching and predictive-value controls remain
necessary before representation-changing training. I have not inspected any
future washout results or approved that future experiment's implementation.

Reproduce from repository root:
`bash research/campaign_2026-10-03/continuation/round3/usage_review/run.sh`.
For the exact-domain diagnostic, invoke the same run_cpu.sh round2-critic wrapper
with env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python and this directory's
domain_diagnostic.py. Both use onecore/2GiB/900s; results/logs and STATE contain
measured cost. The main audit takes about11 seconds, peak RSS below900MiB;
the array-only domain diagnostic takes about.12 seconds. No GPU, services,
training data fit, model forward, grader/source edit or queue modification occurred.
