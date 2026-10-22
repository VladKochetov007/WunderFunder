# Strict-FP32 Numba online relationship stream code/math review R1

This is an independent source and recurrence review of the optional compiled
online relationship transform. I found no semantic difference from the Python
reference by inspection. This review does not establish runtime speed or a
training result.

## Source and scope

- Reviewed actor commit: `46b7ff106875acc30259a3be1abb33f3313d90ee`.
- The same compiled implementation is present at root integration commit
  `c0c1c639bca2babbf03981e42f878619817ab6a0`.
- The Python reference in `src/wnn/online_relationship.py` is unchanged by the
  actor commit except for the new `transform_compiled` adapter. Its parent blob
  is `e2cc59cb58c89f05442dcec46ee7d7024c961106`.
- Reviewed the [Numba kernel](../../../src/wnn/online_relationship_numba.py),
  [adapter and Python transform](../../../src/wnn/online_relationship.py), and
  [scalar-oracle and chunk-continuation tests](../../../tests/test_online_relationship_numba.py).
  No tests or benchmarks were run during this independent review because the
  root agent reserved the CPU slot.

## Recurrence check

The kernel is decorated with `njit(cache=True, fastmath=False, parallel=False)`
and loops over rows and channels serially. Its online state and output arrays
are FP32. Arithmetic that feeds the recurrence has explicit FP32 conversions;
the update equations follow the Python path's operation order.

At row `t`, the kernel forms `u[t] = i1[t] - i1[t-lag]` when available and
stores that standardized, clipped value in the 16-row ring. The matured target
at `t >= 16` is `v[t] = i0[t] - i0[t-16]`, paired with the stored `u[t-16]`.
Unavailable `u` values are zero. For lag 16, this means matured rows 16 through
31 have the specified zero regressor; row 32 is the first matured pair with a
real regressor.

The error uses the coefficient state before the current pair update. The
forecast and emitted slope use the state after the update and the current
`u[t]`. Arm A keeps slope zero while maintaining the shared mean/statistic
state; arm B updates on matured rows 16 through 98 inclusive and freezes at row
99; arm C continues updating. The ring slots, reset count, and state mutation
also match the Python implementation for chunked calls. Each emitted component
is clipped to ±8, as are the standardized differences.

The new tests compare every output and final state array against a scalar FP32
oracle for all six arm/lag combinations. They also cover chunk restore and
mixing compiled and Python chunks. Root reports that the bounded test run
passed 35 tests bitwise with peak RSS 229,548 KiB; I did not independently
rerun that report.

## Limits

This is a code/math pass only. The kernel's cold compilation and warm throughput
were not measured here. The earlier feature-only profile calls the Python
`transform` path, so its timing does not characterize this compiled method.
The low-level Numba function assumes its caller supplies a valid arm code, lag,
and correctly shaped state arrays; the only in-repository caller maps from a
validated `OnlineRelationshipState`. Direct callers should use that state API.
No full training runner, parent prediction cache, GPU fit, grading result, or
performance claim is reviewed here.

**Disposition:** no code/math blocker found by inspection; compiled-path
performance remains unmeasured.
