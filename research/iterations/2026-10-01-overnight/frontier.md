# Overnight research frontier

Updated 2026-10-01 02:48 UTC. Campaign deadline: 08:23 UTC. Local and beleriand only; no GitHub publication or public submission.

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

Completed: 36 forecasting fits, 32 held-fold prediction jobs, four parent-integrity checks and 28 guarded WP readouts. All required completed artifact checks pass. SAM controls reproduce the incumbent checkpoint tensors and full prediction files exactly; its paired and incumbent queries are the same evidence, not independent replication. The SAM paired whole-chain interval is [−.00032696, +.00017060]. Full failures, controls and immutable sufficient statistics remain in the branch directories and evaluation ledger.

A small positive pooled Pearson difference with negative differences in both folds is possible because pooling changes means and scales. It is not evidence of a gain in each domain. Chain-bootstrap intervals condition on fitted models and observed chains; they do not cover seed variation, adaptive selection or hidden-test shift. All folds have been exposed.

## Input-only falsifier and historical triage

QRENEWAL is closed. The predetermined 256 train-only chains contain 295,833 marked resets; the existing change detector misses zero. The proposed common-clock median errors are .583 and .618, and every chain fails its clock gate. Numerical solver checks pass, so this is an empirical mechanism failure rather than a solver failure. No targets, forecasting fits or WP queries followed.

The outcome-conditional selection-weight scout was also closed without new empirical spending: E-103 already tested its central premise, E-170 failed sharper propensity weighting, and E-186 failed a related conditional residual map. Historical attention, dual-clock, stronger teachers, mirror-code assumptions and online reconstructed-label experiments are not automatically new hypotheses. In particular, the large historical matured-label gain was leakage and is excluded.

## Serving artifacts and current branches

**Ready fixed-three fallback:** `submissions/overnight_baseline3/baseline3_fixed.zip`, SHA256 `9471821e81e7c3f80a088147512183b72e0ab52bc593a1136d2becc229d0d098`, 3,796,019 bytes. Equal weights on plain h128 seeds 3/4 and the chunk-500 h128 member; every array matches the corresponding historical V1 weight. No wall-clock pruning. All three ISA builds pass correctness checks. The v3 one-million-callback test takes 67.7116 seconds, projecting to 44.46 minutes at 39.4M rows, with 88.75 MiB peak RSS and a quiet SMT sibling. Its local WP is .7017424296043642, which is .000624796 below the fixed-four ensemble. This is a reproducible runtime fallback, not a new quality SOTA.

**Q3:** fixed distilled-h256 + plain-h128 seed 3 + chunk-500-h128, equal thirds, unchanged dynamic-INT8 quantizer. ZIP SHA256 `48cf940bd1c7e812fe8005a90e3f11fb3199a8940c55e0adbfef00ad1de1f10f`, 6,501,798 bytes. Independent runtime review passes for v3/VNNI: all three 100k traces and one 1M trace project to 45.38–46.10 minutes; the long trace is 45.85 minutes. Every timed prediction matches the frozen old-engine reference. The separately forced AVX2 floating/integer path fails cleanly at 125.18 minutes and is stopped. Natural v3+AVX2 and v2 runtimes are unmeasured; unknown public CPU is a material limitation. Two folds of quantized and same-composition FP16 predictions are now allocated, with exactly two later root-owned WP comparisons. No quantized quality result exists yet and no AVX2 full-fold audit is eligible.

**CTXMEM:** frozen ordinary h128 hidden state projected into 16 fixed coordinates, with two event-clock EMA memories of half-lives 4 and 32. Compare against an equal-parameter head receiving duplicated current projected state. Two plain members of the fixed-three fallback are replaced together. All baseline parameters remain frozen; only ordinary H2 heads and zero-initialized extra head weights train. Independent state, gradient, AMP/fused-optimizer and Python/native checks pass. The fixed-three actual-input native preflight passes: first three quiet paired overhead ratios 1.026603, 1.014414 and 1.023039, median +2.304% under the fixed 3% gate. Zero-extra full callback predictions are bit-exact to the archived baseline. Exact eight-fit training sources and execution plan are being prepared; no CTX fit has started.

The original fixed-four INT8 branch remains closed at 56.57 minutes against its unchanged 50-minute gate. Q3 is a separately declared composition under prospective amendment 01, not a retuned continuation of that failure. The amendment adds CTX current-state, CTX event-memory and Q3 configurations while preserving the 64-fit, 64-readout and wall/resource budgets.

## Interpretation and next decisions

Q3 must pass both frozen quality gates: quantization loss at least −.0001 pooled and on each fold against same-composition FP16; improvement at least +.0003 pooled and nonnegative on each fold against the fixed-three h128 fallback. Passing would qualify only the evaluated numerical path and would not establish public CPU compatibility.

CTX must pass four predetermined joint and per-seed comparisons after its matched two-seed/two-fold training. No rank, half-life, projection, clock, seed or blend rescue is allocated. A positive native cost result is only permission to prepare that experiment, not evidence of forecasting value.

Keep the 80-minute final-verification reserve and the historical public ZIP. Report startup timing accurately: current import/model timings exclude NumPy already imported by the harness. Long runtime traces repeat five 20k-row sequences with new IDs, not one uninterrupted million-row state. No top-1 claim is supported by the current observations.
