# Empirical selection and confirmation

Use this protocol when repeated empirical feedback selects models, recipes, proxies, or hypotheses. Exact proofs and deterministic correctness checks do not need seed experiments or a statistical holdout. Scale the protocol to the claim and budget; a cheap screen is useful without being confirmatory.

## Before reading candidate results

- Declare the primary outcome, comparator, practical effect threshold, evaluation unit, number of paired training seeds, and stopping rule. Match the comparison to the deliverable: for an ensemble, measure the candidate's contribution to the fixed deployed blend.
- Record each evaluation slice's source/group identifiers, split hash, prior uses, and exposure status: `unexposed`, `exposed`, or `unknown`. Include earlier feature design, proxy tuning, diagnostics, and model selection, not just direct fitting. Treat unknown exposure as discovery-only until established otherwise.
- Preserve exposure through lineage. Repartitioning, renaming folds, changing seeds, or hiding old results from a new agent cannot make inspected data untouched. Group related observations (for example continuous chains) across train/evaluation boundaries. Record residual dependence and domain shift separately.
- Set a selection budget: candidate configurations, proxy variants, posthoc readouts/slices used to choose a candidate, and evaluation or public-submission queries. Count failed and revised attempts. A budget bounds search; it does not itself correct selection bias.

## Discovery is allowed to adapt

Use exposed development data to diagnose failures and rank candidates. Log the complete tried set and label estimates conditional on the selected procedure. A fixed-size threshold or one bootstrap SD is a spending gate, not a significance guarantee. Failed gates mean `screen_failed` or `insufficient_evidence`, with the recipe and tested effect size attached.

If validation chooses a proxy, blend, cutoff, or other evaluation procedure, its own evaluation must include that choice. For example, choosing the best of many proxies by leave-one-out error and quoting the winning error is still adaptive selection. Use an outer evaluation containing the entire selection procedure when feasible, or freeze the selected procedure for prospective testing. Reused outer data and correlated public submissions do not supply independent confirmation. Do not infer effective sample size from the number of comparisons sharing those submissions.

## Freeze before confirmation

Commit a timestamped specification with hashes for the candidate/configuration, reference, evaluator, and evaluation split before accessing confirmation outcomes. Freeze the primary metric/mask, epoch/checkpoint rule, blend weights, seed pairs, comparison and stopping rule. Include exact commands and the selection history. Amendments after results are new exploratory attempts; keep the original specification and result.

Use genuinely unexposed units or a prospective external evaluation. Record which artifact was submitted, which outcome was declared in advance, and what the result actually tests. Once used to choose or alter a candidate, that feedback is development evidence for descendants. If all data have been inspected, say so: additional cross-validation can assess robustness but does not restore pristine confirmation. A competition may still choose a submission under uncertainty; that operational choice does not upgrade a scientific claim.

For formal statistical claims, state the error target and use a design valid for the planned selection and repeated looks. Do not impose one universal p-value, Sharpe ratio, or minimum delta across tasks. If the available budget cannot resolve the stated effect, report the limitation rather than declaring equivalence or impossibility.

## Separate sources of uncertainty

- **Evaluation sampling:** resample the independent unit, such as a whole chain, paired across candidate and control. A chain bootstrap of fixed predictions conditions on the fitted models; it does not include training variability.
- **Training variability:** preregister multiple paired candidate/control seeds for a recipe claim. Hold data, training budget, epoch rule and serving comparison fixed except for the declared treatment; couple initialization and sampling randomness where meaningful, documenting exceptions. Report every paired delta and their variability. Do not select the best seed or count ensemble members as independent replications. A single-seed screen remains provisional.
- **Transfer and selection:** neither seed repetitions nor a small bootstrap interval establishes performance under domain shift or removes adaptive selection. Report those uncertainties separately. Reserve truly fresh confirmation when possible.

A rerun of the same code checks computational reproducibility. An independently written implementation checks implementation dependence. Paired fresh training seeds check recipe variability. Unexposed or prospective evaluation checks the frozen claim on new evidence. Agent/model-family diversity helps find mistakes but substitutes for none of these.

The minimal machine-readable fields are in [research-artifacts.md](research-artifacts.md). Use a manifest plus append-only results; a new workflow engine is not required.
