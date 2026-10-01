# Overnight frontier

Updated 2026-10-01 01:32 UTC. Campaign ends approximately08:23 UTC. Local and SSH work only; no GitHub publication or public submission.

The historical public V1 score remains .6641. Reconstructed two-epoch H2 four-member controls score .7023672255138177 on exposed pure-chain folds0/1. These are different evaluation domains; the latter is not a public score estimate. No new forecasting recipe has passed its promotion gate.

| Attempt | Fixed primary pooled delta | Fold0 / fold1 | Decision |
| --- | ---: | --- | --- |
| Auxiliary losses off | −.000105740 | −.000099715 / −.000131771 | Stop; keep auxiliaries |
| Conditional transition T versus adaptive-frequency F | +.000036168 | −.000003463 / −.000000418 | Stop; below .0003 and folds negative |
| T versus plain H2 | +.000039310 | −.000015561 / +.000024744 | Does not establish a useful gain |
| F versus plain H2 | +.000003141 | −.000012097 / +.000025163 | Diagnostic null |
| Ensemble-aware heads | −.000596892 | −.000185676 / −.000893625 | Stop |
| Two-scale ensemble control | +.000121319 | +.000260040 / −.000042112 | Stop; below .0003 and inconsistent folds |
| SAM parent versus matched ordinary parent | Not read | Not read | Seed3 paired parent/head study running |

All28 completed forecasting fits have passed artifact checks: the declared frozen backbones are bit-identical to their parents, heads changed, tensors are finite, and fixed epoch1/fold metadata agree. All28 held-fold prediction passes succeeded. The26 completed evaluator readouts include one health reproduction, five AUX comparisons, seven transition comparisons, two deployment-composition comparisons and eleven ensemble-objective comparisons. Each candidate result retains prediction/checkpoint hashes and whole-chain sufficient statistics. Intervals condition on fitted models and do not cover seed variation, adaptive selection or distribution shift.

The scale-export checker initially assumed CPU and CUDA exp produced identical float32 values. They differ by one ULP on two fitted scale vectors; maximum resulting weight difference was7.45e-9. Repeating the declared export computation with CUDA exp reproduced all four exports bit-for-bit. The failed checker, numerical diagnostic and corrected checker are preserved; no trained weight or prediction was changed.

The transition pooled delta being positive while both fold deltas are negative is possible for pooled Pearson: scale/centering across folds changes the pooled result. It is another reason not to interpret a tiny pooled increase as consistent predictive improvement. Positive code-sequence log-loss evidence from the earlier quant controls did not translate into a useful price-forecast improvement through this fixed68-channel head recipe. This is a failure of the tested delivery mechanism, not proof that all event-order information is irrelevant.

Runtime is a material deployment constraint. Historical V1 prunes members based on elapsed wall time and can therefore change forecasts with host load. Fixed4 is reproducible but projected at roughly112minutes on beleriand, above the60minute limit. Fixed3h128 costs .000624796 WP versus fixed4; fixed2 loses another .000523673. The member-order optimization's first version passed equal-coefficient checks but failed arbitrary-coefficient exact parity because the compiler changed contraction in the new reduction loop. A separately versioned explicit-arithmetic correction passed independent checks on all3 CPU ISA builds. Fixed3 whole-callback samples project47–49minutes when the SMT sibling is idle. Contaminated samples exceed60minutes and remain in the record; CPU/wall equality alone does not exclude shared-core contention. The deployment plan freezes a50minute proxy gate and requires longer complete-package verification.

A separate dynamic-INT8 numerical serving branch cuts fixed4 cost substantially but has not passed that runtime gate. Quantized outputs remain finite with approximately .0043 relative RMS deviation on a fixed raw trace. This is not a quality score or a bound on WP loss. The initial89.93us/row variant and later exact-output kernel variants are preserved; no quantized forecasting score has been read. Further kernel work keeps the quantizer and predictions fixed.

After three failed mechanisms, the research direction was explicitly reconsidered. The independent SAM proposal changes parent representation training without adding inference operations; CPU, CUDA AMP, fused AdamW, overflow and recurrence checks passed. Its seed3 study started01:25:55UTC with two matched controls followed by two SAM parents and their fixed two-epoch heads. GPU scope monitoring stops only our registered jobs if usage exceeds5632MiB. The second seed is conditional on the frozen first-seed gate.

The other research direction is a cheap input-only falsifier: test whether anonymous a5 resets reveal repeated mark occurrences omitted by payload-change detectors. Its common-clock and state-divergence requirements must pass before any target association or new forecasting allocation. Implementation/review is pending; no empirical clock result exists yet.

Remaining work: finish the paired SAM screen, evaluate the input-clock falsifier, establish numerical-kernel quality/runtime if feasible, then train/verify any promoted finalist and build fresh submission ZIPs. No top1 claim is supported by these local observations.

## 02:03UTC frontier update

SAM seed3 closed after8fits: pairedandincumbent delta−.000088073,bothfoldsnegative,CI[−.00032696,+.00017060]. Controls reproduce incumbenttensors andpredictions exactly; twoqueriesare sameevidence. No seed4/finalist/conditionalfixed3 queries. QRENEWAL closed: zero missedmarkedresets/295833; all256chains failclockmedianerror. Outcome-awareselection scoutclosedasE103/E186duplicate.

Concretefixed3ZIP9471821e… validatesall3ISA; v3fullcallback1M67.7116s→44.46min39.4Mrows,88.75MiBRSS,quietSMT. Startup~19ms excludesalready-importedNumPy. CompositionOOFstill−.000624796vsfixedfour; notnewqualitySOTA.

Rootprospectiveamendment01 permitsCTXMEMcurrent+eventEMA andQ3[d256,s3,c500] preparation, scientificconfigs8→11; all64fit/WP/resource/deadlinecapsunchanged. No newempiricalruns yet. Selectedpostfailureonexposedhistory; fixedgatesbeforeoutcomes.
