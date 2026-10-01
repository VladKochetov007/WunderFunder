# D1 amended independent CPU grade: PASS

Model SHA256 `75beff387c4f9312dc3bc7e065d76e7df2ceeb666fe1bb3e93cc245dfca445cd`;
preregistration SHA256 `9f3325206d41a9fe4118aada9b967b6fc574e24a97af81a86d48cd77e5ce5a61`.
Independent v2 contract SHA256
`153cc7833006a19df05385c715e157eb2b2664946ae916dc93fdbf71356050df`.
Original failed source/contract/results remain preserved separately.

Root authorized a prefit numerical amendment: store S/Z multiplied by the fixed
per-head factor1-base_decay, multiply every write injection and denominator floor
by that same factor, and use powers of the actual serialized FP32 base_decay.
Both serial and parallel definitions now use that coefficient. States are bounded
by9 from zero valid history. The scaled checkpoint/buffer/state grammar is new;
old unscaled state/checkpoint bytes cannot be carried without explicit conversion.
No width, addressing, event clock, readout or trainable parameter change.
Absolute numerical tolerances were unchanged. Their state units are explicitly
the new bounded representation, not the old unscaled accumulator units.

Fresh independent own recurrence/gradient grader passed25/25cases under durable
`c20260930_independent_d1_grade_v2`:96.86s, peak735500KiB (718.26MiB), verified
one-core2GiB/swap0 scope in wnn-campaign.slice, CUDA uninitialized. It imports
only the frozen model, never actor reference/proof helpers or competition data.
`event_content_grade_v2.json`, own `grade_event_content_v2.py` and v2 contract
preserve hashes and all measurements. Scorer remains unchanged.

All three arms pass20k quiet AND20k eventful full-versus-literal-serialized
recurrence at original forward3e-5/state0.003 bounds. Raw sourcefloat32 one-ULP
endpoint discrimination before normalization, raw-code originfalse, no artificial
event seed, empty eventhistory, genuine first ordinal0, boundaries/counters,
nonzero state, chunks1/17/64/2000, causal future-prefix/reset/batch isolation,
FP64/FP32 input/state/parameter gradients and synthetic Walsh96-column exposure
pass. Variable values obey the positive normalized retrieval prefix convex hull;
clamped extremes and FP32 memory under outer bfloat16 autocast remain finite.

An additional independent unscaled real recurrence with the same serialized
coefficients verifies model output equivalence after converting carried S/Z.
Per-head denominator tests above AND below the floor verify algebraic scaled
ratio equivalence to about1e-16. This confirms the floor was scaled together
with numerator/denominator; silently keeping the old floor would be a different
retrieval rule. These equivalence tests are separate from relaxed tolerance.

PASS is standalone CPU engineering parity only. Actual TRAIN endpoint/exposure
gate, integration instrumentation/closure, GPU/native backend parity, equal
successful optimizer updates, resource cost and forecast quality remain unproved.
Modulo128 position aliases and joint event write/decay contrast remain explicit.
No candidate fit/score/public quality promotion or automatic GPU launch.
