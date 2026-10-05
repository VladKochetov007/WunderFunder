# Independent source review: E484 export arithmetic

Reviewed the root-owned E484 reproduction helper without opening the result
JSON or paired export and without running its `main()` function. Reviewed
source SHA256:

`reproduce_context_exports.py`: `b7a5919b5ddc5d046dda8c73ee7fb9f76a26ec6508a1487b3614d2787532eedd`

## Arithmetic disposition: GO

For source-only masking validation, the helper indexes full `[16, 20000, 2]`
target and per-arm prediction arrays with the shared boolean `[16, 20000]`
selection mask before splitting selected rows by per-component counts. It
reorders component statistics by numeric component ID, regenerates the shared
2,000-draw component multiplicities with seed 271834, and compares those draws
with the saved multiplicities. The global and two target weighted-Pearson
values are each checked per arm. Every named contrast is recomputed by paired
candidate-minus-reference bootstrap subtraction; global scores are the mean of
the two target scores. The helper checks global and target-wise point
differences, bootstrap intervals, sample standard deviations (`ddof=1`), and
nonpositive-draw fractions.

A synthetic ragged-mask fixture tested the raw-sum method against official
`wnn.gmetric.global_wp` for 12 arms and all 2,000 component draws; maximum
absolute discrepancy was `1.89e-15`. The 8 registered contrasts for each of
two seeds imply 16 global and 32 target-wise comparisons. This synthetic test
did not load any E484 exports or labels and did not call the helper's output
writer.

The final source adds exact full-array shape and boolean-mask checks, confirms
the target-contrast key inventory matches the 16 exported global contrasts,
and requires exactly `t0` and `t1` target summaries for every contrast. These
are fail-closed structural checks; they do not alter the arithmetic. No source
change is requested by this review. No real E484 export has been read or
scored here, and this report makes no scientific-result claim.
