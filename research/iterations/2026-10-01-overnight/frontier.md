# Overnight frontier

Updated 2026-10-01 01:05 UTC. Campaign ends approximately08:23 UTC. Local and SSH work only; no GitHub publication or public submission.

The historical public V1 score remains .6641. Reconstructed two-epoch H2 four-member controls score .7023672255138177 on exposed pure-chain folds0/1. These are different evaluation domains; the latter is not a public score estimate. No new forecasting recipe has passed its promotion gate.

| Attempt | Fixed primary pooled delta | Fold0 / fold1 | Decision |
| --- | ---: | --- | --- |
| Auxiliary losses off | −.000105740 | −.000099715 / −.000131771 | Stop; keep auxiliaries |
| Conditional transition T versus adaptive-frequency F | +.000036168 | −.000003463 / −.000000418 | Stop; below .0003 and folds negative |
| T versus plain H2 | +.000039310 | −.000015561 / +.000024744 | Does not establish a useful gain |
| F versus plain H2 | +.000003141 | −.000012097 / +.000025163 | Diagnostic null |
| Ensemble-aware heads / two-scale control | Not read | Not read | Preparing training-partner caches |
| SAM parent versus matched ordinary parent | Not run | Not run | Implementation preparation only |

All20 completed forecasting fits have passed artifact checks: the declared frozen backbones are bit-identical to their parents, heads changed, tensors are finite, and fixed epoch1/fold metadata agree. All20 held-fold prediction passes succeeded. The thirteen completed evaluator readouts include one health reproduction, five AUX comparisons and seven transition comparisons. Each candidate result retains prediction/checkpoint hashes and whole-chain sufficient statistics. Intervals condition on fitted models and do not cover seed variation, adaptive selection or distribution shift.

The transition pooled delta being positive while both fold deltas are negative is possible for pooled Pearson: scale/centering across folds changes the pooled result. It is another reason not to interpret a tiny pooled increase as consistent predictive improvement. Positive code-sequence log-loss evidence from the earlier quant controls did not translate into a useful price-forecast improvement through this fixed68-channel head recipe. This is a failure of the tested delivery mechanism, not proof that all event-order information is irrelevant.

Runtime is a material deployment constraint. Historical V1 prunes members based on elapsed wall time and can therefore change forecasts with host load. Fixed4 is reproducible but projected at roughly112minutes on beleriand, above the60minute limit. Fixed3h128 is near55minutes before optimization and fixed2 near35minutes. An isolated deterministic member-execution-order optimization has early bit-exact parity and approximately10% speedup for fixed3; independent review and full-package measurements remain pending. The deployment plan freezes quality readouts and a50minute projected runtime gate before selecting a composition.

Remaining work: finish the ensemble-head study, review and run the paired SAM screen, validate the exact-output native optimization, then train/verify any promoted final candidate and build fresh submission ZIPs. No top1 claim is supported by these local observations.
