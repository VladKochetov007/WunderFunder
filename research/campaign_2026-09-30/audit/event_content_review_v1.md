# D1 independent CPU grade: FAIL, frozen v3 source

Model SHA256 `c591649e38fb787006a8714107ab71e60028f671cf9a0d854795e2256027b9c8`;
preregistration SHA256 `7fb27cdd25e01a67cd686c8ef256e344f2b9cabbaf24d9349a01f859ed4ff378`.
Independent contract and own numerical source were frozen before execution.
Actor proof/reference helpers and results were not used by the numerical grader.

Durable `c20260930_independent_d1_grade_v1` finished with structured FAIL:
19 of22cases pass. Analysis51.31s, peak716520KiB (699.73MiB), no CUDA or
competition data. Actual nested scope properties prove1CPU, MemoryMax2GiB,
MemorySwapMax0 and shared wnn-campaign.slice. The program exits0 to preserve
structured evidence; its result status is FAIL and is the promotion gate.
`event_content_grade_v1.json`, own `grade_event_content.py` and
`event_content_grade_contract.json` preserve all cases/source hashes.

## Exact blocking case

All three arms fail20k quiet serialized-state parity under the preregistered
absolute state tolerance0.003. Endpoint-content and raw-code-content differ
by0.00612640380859375; endpoint-position differs by0.012939453125. Test source:
model(211inputs,23outputs), seed534, zero1x20000x211 input; a6 exact endpoint
only at row1, firstprice0.2 from row1 onward, no artificial event origin, one
genuine event. Direct real S/Z recurrence and row-by-row GRU are independently
implemented. Forward output comparison passed3e-5 before state assertions.
The subsequent20k eventful subcase was not reached after each quiet failure;
it remains ungraded. Actor short/chunk checks cannot establish this long serial
state identity. No tolerance was relaxed and original source remains unchanged.

Parallel code exponentiates cumulative serialized log_decay, whereas direct
serialized FP32 real steps use rounded exp(log_decay) at each update. The
coefficient/power difference can accumulate, especially in row-head half-life512
with large unscaled S/Z. This is a candidate cause, not a proven sole cause.
Builder must preserve original files, amend arithmetic/representation before
new probes, then retest the actual serialized recurrence with the same numeric
tolerances. Root proposed bounded scaled S/Z with correspondingly scaled floor;
if adopted, add an exact real attention-ratio equivalence test and explicit
carried-state conversion semantics. No state tolerance relaxation or GPU fit.

## Cases that pass

Independent FP64 forward/state/parameter/input/state gradients pass (worst
gradient4.98e-14); FP32 gradients pass (worst3.815e-5). Nonzero state, chunks
1/17/64/2000, prefixes against future perturbations, exact reset and batch
isolation pass. Raw sourcefloat32 predicate is tested at one-ULP adjacent values
after nonidentity normalization: only exact endpoints write. Raw-code origin is
false, endpoint origin writes only when actually equal, empty eventhistory stays
zero, first/boundary writes and counters agree with the independent reference.

Walsh128x96 Gram/rank are exact, and all trained parameters plus every q/k input
column have gradient exposure on synthetic sequences. This does not establish
actual TRAIN exposure. Values vary through time: normalized retrieval lies in
the per-channel observed prefix convex hull includingzero when floor is active.
Positive Z, documented memory bounds, extreme clamped logits and outer CPU
bfloat16 autocast preserving FP32 S/Z pass. Serialized counters/previous prices
agree exactly. No native/GPU/parity or package-runtime conclusion follows.

Position addressing still aliases modulo128. Endpoint-versus-code arms alter
write and decay together. Frozen model has no explicit carried-state dtype/shape
validator or stream_step convenience method; integration should establish those
boundaries or provide a wrapper before deployment. These limits are distinct
from the blocking numerical failure. Fresh TRAIN component exposure and equal
successful-update cost gates also remain outstanding.
