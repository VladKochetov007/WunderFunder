# Representative small-data hypothesis screens

User scope on 2026-10-08: CPU is for analysis/EDA and support, not large model fitting. Use GPU/TPU for fits. Freebuff/GLM receives a fresh independent EDA assignment; verify the actual provider/model, and preserve any access failure without attributing another model's output to GLM.

## Question

Can cheaper training subsets identify useful mechanisms without losing directions whose benefits emerge with data or model scale? This is exploratory, lower-fidelity allocation evidence, not a substitute for the competition endpoint or a universal rejection rule.

## Sampling and comparison

- Sample only fitting membership. Preserve connected components and complete sequences, reset boundaries, causal prefixes and all original scored/prediction masks. Do not randomly discard rows from recurrent histories.
- Begin with nested roughly 10%, 30% and full-data training sets; exact realized counts depend on component sizes. Freeze membership lists, selection seed and hashes before grading. Record inclusion probabilities and any stratification. Preserve an unstratified reference; rare-regime enrichment is a separate treatment.
- Use streaming fitting-data summaries to check feature ranges, quantization, missingness, sequence lengths and regime coverage. Training-label stratification, if needed, is explicitly declared and never chosen from validation outcomes. Visual resemblance alone does not establish representative conditional targets.
- Retain the same fixed, component-separated evaluation membership and official metric at every fidelity. Existing validation remains exposed; smaller training sets do not make it fresh. Avoid shrinking validation merely because training is cheaper.
- Pair candidate and control on the same membership, initialization convention and sampling schedule. Declare epochs, processed rows and optimizer updates: equal epochs and equal update counts answer different questions. A subset continuation from a full-data checkpoint tests adaptation, not small-data training from scratch.
- Teacher/source/propensity ancestry must exclude evaluated components. If teacher weights saw the complete fitting set, record this: the resulting student screen does not measure the benefit of limiting all training data.

## Promotion without premature family rejection

First compare several different mechanism families at two fidelities, measuring paired effects, throughput and rank changes. Allocate larger tests to promising screens and reserve some larger tests for plausible scale-dependent mechanisms, conditional contributors and rank reversals. A weak small-data result can defer an exact recipe; it cannot establish that a whole family fails at scale.

Changing data size changes the training regime; it does not violate a scaling law. Learning curves and rankings may cross because capacity, regularization, rare regimes and optimization budgets interact. Do not fit an extrapolated public-score promise from a few subset points.

For uncertainty, bootstrap paired predictions by evaluation component. That interval conditions on the fitted models; it does not include training-subset variation, optimizer randomness, adaptive selection or public transfer. Record these separately. Confidence intervals crossing zero remain an uncertainty summary, not an automatic kill switch.

## EDA deliverable

The independent GLM lens is anonymous encoding, algebraic redundancy, latent geometry and sampling coverage. Earlier EDA covered target persistence, target coupling, event positions, calibration and residuals; provide a fresh minimal task context rather than its conclusions. Execute through marimo-pair and save standalone plots under the E534 lineage's `plots/`, with notebook, bounded scan parameters, source identities, coverage limitations and public final output. No reactive cell launches a fit.

Primary methodological basis: [FABOLAS](https://proceedings.mlr.press/v54/klein17a.html) treats dataset size as a fidelity variable and models loss and cost. This motivates a cheaper screen; it does not guarantee stable model rankings for our data.

Status: protocol prepared; no subset fit, representativeness result or new score is claimed. Exact GPU recipes and resource admission follow the EDA findings.
