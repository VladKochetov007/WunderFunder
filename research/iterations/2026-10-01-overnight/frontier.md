# Overnight research frontier

Updated 2026-10-01 03:29 UTC. Campaign deadline: 08:23 UTC. Local and beleriand only; no GitHub publication or public submission.

The historical public V1 score remains .6641. Reconstructed fixed-four H2 controls score .7023672255138177 on the exposed pure-chain folds 0/1. These are different domains; local WP does not estimate a public rank. No new forecasting recipe has passed its promotion gate.

## Completed forecasting experiments

| Mechanism and fixed primary comparator | Pooled WP difference | Fold 0 / fold 1 | Decision |
| --- | ---: | --- | --- |
| Auxiliary losses off versus ordinary H2 | −.000105740 | −.000099715 / −.000131771 | Closed; retain auxiliaries |
| Conditional transition features T versus frequency control F | +.000036168 | −.000003463 / −.000000418 | Closed; below useful-gain gate and both folds negative |
| T versus plain H2, diagnostic | +.000039310 | −.000015561 / +.000024744 | No useful consistent improvement |
| F versus plain H2, diagnostic | +.000003141 | −.000012097 / +.000025163 | Diagnostic null |
| Ensemble-aware nonlinear heads versus ordinary heads | −.000596892 | −.000185676 / −.000893625 | Closed |
| Ensemble scale control versus ordinary heads | +.000121319 | +.000260040 / −.000042112 | Closed; insufficient and inconsistent |
| SAM parent, followed by ordinary H2, versus matched ordinary parent | −.000088073 | −.000033374 / −.000117150 | Closed at seed-3 screen; no seed 4 or finalist |
| Event-clock memory heads versus duplicated-current-state heads | +.000055788 | −.000048406 / −.000003062 | Closed; below gate, both folds negative |
| Event-clock memory heads versus plain fixed-three, required | −.000006320 | −.000106248 / +.000016052 | Closed |
| Q3 INT8 versus same-composition native FP16 | −.000011454 | −.000009248 / −.000011516 | Numerical loss gate passed |
| Q3 INT8 versus plain fixed-three GPU anchor | +.000258385 | +.000375393 / +.000044130 | Closed; below the fixed +.0003 promotion gate |

Completed: 44 forecasting fits, 44 held-fold prediction jobs (40 GPU and four native CPU), four parent-integrity checks and 34 guarded WP readouts. All required completed artifact checks pass. SAM controls reproduce the incumbent checkpoint tensors and full prediction files exactly; its paired and incumbent queries are the same evidence, not independent replication. The SAM paired whole-chain interval is [−.00032696, +.00017060]. Full failures, controls and immutable sufficient statistics remain in the branch directories and evaluation ledger.

A small positive pooled Pearson difference with negative differences in both folds is possible because pooling changes means and scales. It is not evidence of a gain in each domain. Chain-bootstrap intervals condition on fitted models and observed chains; they do not cover seed variation, adaptive selection or hidden-test shift. All folds have been exposed.

## Input-only falsifier and historical triage

QRENEWAL is closed. The predetermined 256 train-only chains contain 295,833 marked resets; the existing change detector misses zero. The proposed common-clock median errors are .583 and .618, and every chain fails its clock gate. Numerical solver checks pass, so this is an empirical mechanism failure rather than a solver failure. No targets, forecasting fits or WP queries followed.

The outcome-conditional selection-weight scout was also closed without new empirical spending: E-103 already tested its central premise, E-170 failed sharper propensity weighting, and E-186 failed a related conditional residual map. Historical attention, dual-clock, stronger teachers, mirror-code assumptions and online reconstructed-label experiments are not automatically new hypotheses. In particular, the large historical matured-label gain was leakage and is excluded.

## Serving artifacts and current branches

**Ready fixed-three fallback:** `submissions/overnight_baseline3/baseline3_fixed.zip`, SHA256 `9471821e81e7c3f80a088147512183b72e0ab52bc593a1136d2becc229d0d098`, 3,796,019 bytes. Equal weights on plain h128 seeds 3/4 and the chunk-500 h128 member; every array matches the corresponding historical V1 weight. No wall-clock pruning. All three ISA builds pass correctness checks. The v3 one-million-callback test takes 67.7116 seconds, projecting to 44.46 minutes at 39.4M rows, with 88.75 MiB peak RSS and a quiet SMT sibling. Its local WP is .7017424296043642, which is .000624796 below the fixed-four ensemble. This is a reproducible runtime fallback, not a new quality SOTA.

**Q3, experimental artifact, not promoted:** fixed distilled-h256 + plain-h128 seed 3 + chunk-500-h128, equal thirds, unchanged dynamic-INT8 quantizer. ZIP SHA256 `48cf940bd1c7e812fe8005a90e3f11fb3199a8940c55e0adbfef00ad1de1f10f`, 6,501,798 bytes. Independent runtime review passes for v3/VNNI: all three 100k traces and one 1M trace project to 45.38–46.10 minutes; the long trace is 45.85 minutes. Every timed prediction matches the frozen old-engine reference. The separately forced AVX2 floating/integer path fails cleanly at 125.18 minutes and is stopped. Natural v3+AVX2 and v2 runtimes are unmeasured; unknown public CPU is a material limitation. All four fold/mode prediction jobs and independent export/output/input provenance checks passed. Native INT8 WP is .7020008144532741, versus .7020122685424693 for the same FP16 composition. The +.000258385 advantage over the plain fixed-three GPU anchor has a conditional whole-chain interval [−.000527936,+.001045509] and misses the fixed +.0003 gate. Both allocated queries are complete; no rescue or extra ISA audit is allocated. This is neither confirmed hidden improvement nor a native-versus-native comparison to the plain fallback.

**CTXMEM, closed:** frozen ordinary h128 hidden state projected into 16 fixed coordinates, with two event-clock EMA memories of half-lives 4 and 32. Compare against an equal-parameter head receiving duplicated current projected state. Two plain members of the fixed-three fallback are replaced together. All baseline parameters remain frozen; only ordinary H2 heads and zero-initialized extra head weights train. Independent state, gradient, AMP/fused-optimizer and Python/native checks pass. Native preflight overhead is +2.304%, under the fixed 3% gate; zero-extra actual-input callbacks are bit-exact to baseline. All eight fits and eight predictions completed, with 2,258 MiB sampled GPU peak per fit. Independent actual-handoff review passes. All four quality gates fail: joint comparisons are in the table; seed-3 replacement is +.000050415 with negative fold 0, and seed-4 replacement is +.000001702 with both folds negative. No conditional native OOF audit or final heads will run. The initially incorrect H1 comparator aliases were caught and corrected to H2 before training; the rejected plan and invalidated review remain preserved.

The original fixed-four INT8 branch remains closed at 56.57 minutes against its unchanged 50-minute gate. Q3 is a separately declared composition under prospective amendment 01, not a retuned continuation of that failure. The amendment adds CTX current-state, CTX event-memory and Q3 configurations while preserving the 64-fit, 64-readout and wall/resource budgets.

## Interpretation and next decisions

Q3 preserves quality under quantization but misses its composition utility gate. CTX meets serving-cost requirements but adds no useful consistent forecasting improvement. Their artifacts remain available; neither becomes the champion and neither is retuned after outcomes.

**SCALARPLE01 is the only active mechanism.** It retains the original input projection and adds a rank-16 piecewise-linear representation of 44 raw price-code coordinates before the first GELU. Knots will be fitted only on the already fixed 256 train-only chains if preparation gates pass. The planned comparison is a fresh ordinary four-epoch EMA parent plus two-epoch H2 versus the full adapter architecture under the same recipe. Generated CPU checks pass; independent source review, generated full-parent GPU resource tests and a frozen native cost pilot remain prerequisites. No empirical knot fit or scalar forecasting fit has started. Amendment 02 preserves the global 64-fit/64-readout limits and authorizes preparation only; at most 18 scalar fits are reserved including conditional second-seed replication and a final parent/head. No additional mechanism-family search is allocated.

Keep the 80-minute final-verification reserve and the historical public ZIP. Report startup timing accurately: current import/model timings exclude NumPy already imported by the harness. Long runtime traces repeat five 20k-row sequences with new IDs, not one uninterrupted million-row state. No top-1 claim is supported by the current observations.
