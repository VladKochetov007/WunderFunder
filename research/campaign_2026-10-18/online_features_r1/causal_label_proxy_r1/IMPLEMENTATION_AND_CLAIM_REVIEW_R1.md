# Independent implementation and claim review R1

## Review scope

An independent source-only review inspected the proxy recipe, hypothesis card, runner, data reader, split checks, official metric and submission callback documentation. The reviewer did not read Parquet payloads, execute code, or inspect experiment outcomes.

## Findings and response

- **Pass:** the predictor uses only `x[t]` and `x[t] - x[t-1]`; shifted labels and scoring masks are used only by the offline grader. Fit and evaluation row groups are source-qualified, component-disjoint and checked against the F0 manifest. The official metric and shifted scoring mask are applied consistently.
- **Claim correction:** target definitions are described only as indicators of future price movement; their horizon and timestamps are undisclosed. Therefore, `y[t-1]` cannot be called a realized or available feedback label at time `t`. The hypothesis and recipe now describe this solely as a previous-row-label prediction / lagged pseudo-label feasibility screen. Any pseudo-label-driven expert update must be evaluated as its own frozen endpoint.
- **Numerical correction:** the initial runner formed covariance from `E[x x^T] - E[x]E[x]^T`, which could lose precision for high-offset inputs. Before any fit, it was changed to a stable centered weighted batch merge across fit sequences. The weighted ridge objective, component membership, sample stride, target alignment and evaluation procedure remain the same. The fit receipt records feature-scale summaries and floored-feature counts.
- **Follow-up implementation fix:** source review caught a row-shape assignment error in the revised accumulator (`mean_x` versus `mean_x[target_index]`) before execution. The assignment was corrected; the reviewer confirmed the fixed shape and found no remaining blocking source issue.
- **Interpretation limit:** the metric is evaluated on target rows 99–19,998 because each target row is paired with the following feature row; the final target row has no following input row in the same sequence. The 32 evaluation components are exposed development data. One fit and a component bootstrap do not establish training-seed variability, target maturity, useful online adaptation or public transfer.
- **Resource note:** the fit/grade runner does not impose its own cgroup limits. Execution must use the durable controller and the reviewed 1.5 GB, one-CPU reservation. The separately indexed undersized metadata preflight OOM remains part of the operational record; this source review adds no execution and observed no new OOM.

## Decision

**Proceed with a bounded lagged-proxy feasibility screen after a one-row-group input profile and fresh resource admission.** Treat any result as exposed-F0 proxy evidence only. No online expert reweighting, competition endpoint, B300 training recipe or submission gain is established by this screen.
