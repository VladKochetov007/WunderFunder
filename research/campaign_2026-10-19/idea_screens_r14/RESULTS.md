# R14 same-span EMA coordinate screen: results

The fixed screen did **not** support the proposed optimization advantage for the orthonormalized EMA coordinates. After 128 full-fit exact steepest-descent steps per target, arm B had a larger relative objective gap than arm A for both targets; the predeclared B/A ≤ 0.8 spending gate failed for both.

| Target | Arm A gap at 128 | Arm B gap at 128 |
| --- | ---: | ---: |
| 0 | 0.04419051 | 0.05106590 |
| 1 | 0.04278438 | 0.04888715 |
| Mean | 0.04348744 | 0.04997652 |

Arm A is the normalized raw112 plus three unit-norm EMA blocks at half-lives 8, 32, and 128. Arm B applies the fixed analytic infinite-impulse Gram whitener to those same three blocks. Both use the same 449-coefficient predictor span, 64 fit components/31,808 scored stride-40 rows, λ=100 with an unpenalized intercept, absolute-clipped-label weights, and equivalent mapped regularization. Each arm completed 256 target iterations (128 per target), all nonzero; across both arms the total was 512 iterations and 512 nonzero steps, with no no-ops. The finite-step endpoint predictions can differ between coordinate systems.

The empirical Hessian condition numbers were also higher in B: 8,003.16 and 8,348.51, compared with A's 3,589.91 and 3,738.10. The analytic infinite-impulse Gram identity passed (maximum absolute error 6.8×10⁻¹⁶); mapped-Hessian errors were at most 5.7×10⁻¹³, direct-versus-mapped fit-prediction errors at most 1.1×10⁻¹⁴, and mapped optimum coefficient errors at most 1.6×10⁻¹⁴. These checks support that the comparison preserved the fixed same-span objective. The analytic whitener is not empirical covariance whitening.

Secondary canonical scores on the one exposed 32-component evaluation panel were A WP 0.54094749, B WP 0.54259526, and common-optimum WP 0.54173284 (59,436 selected rows each). These scores are secondary to the failed optimization gate, were not used to alter the fit, and do not establish a broader validation ranking.

The run used the accepted R3 panel's 64 fit and 32 eval components. It finished in 61.5 seconds wall time and 30.7 seconds CPU time. The output records report the run's source SHA256 as `65aed497461bbd44a060f44495ec0826a6e462ce8eb0271c74f815d83c8712b0`, result SHA256 as `022ad04edcacd4403485ffc64b772599f5fbcfb52dabeab4b226817fe0f7f5d0`, and a B-only parameter archive of 8,113,012 bytes with producer SHA256 `1eebd144cd8b9e7b37d568dbcc1af466edbcbd60a6a4756ee866be7bb631108c`. This document does not open or copy that archive. The exact unit invocation was `76acba17661a4f49b356e2bacfa39ce4`; root validation occurred at 07:00:24 UTC and the exact stop at 07:00:34 UTC. Capture, validation, and stop are separately recorded.

This closes the exact fixed recipe without pole, penalty, iteration, or objective retuning. It is evidence about finite-step optimization for this analytic convex problem on one exposed panel only. It does not establish behavior for a GRU, strict FP32 campaign training, other optimizers, new information, generalization, or public performance. Component and drop-one summaries are sensitivity views, not independent replications. No original full stdout from the local synthetic-test run was saved; [TEST_RECEIPT.json](TEST_RECEIPT.json) records 69 NumPy matmul warnings observed there, with cause undetermined.

The raw compact capture is linked at [REMOTE_RESULT.json](records/terminal_capture/REMOTE_RESULT.json), [REMOTE_SOURCE.json](records/terminal_capture/REMOTE_SOURCE.json), and [REMOTE_DONE.json](records/terminal_capture/REMOTE_DONE.json). The independent compact result audit passed at [RESULT_AUDIT.json](RESULT_AUDIT.json). See [DESIGN.md](DESIGN.md), [DESIGN_REVIEW.json](DESIGN_REVIEW.json), [PREPARATION_MANIFEST.json](PREPARATION_MANIFEST.json), and [TEST_RECEIPT.json](TEST_RECEIPT.json) for the fixed design, source binding, and verification boundary.
