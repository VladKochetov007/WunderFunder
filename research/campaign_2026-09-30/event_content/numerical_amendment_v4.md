# Frozen numerical amendment v4

Independent grade v1 failed all three long20k quiet final-state checks with unchanged
atol0.003: content/rawcode max0.0061264 and position0.0129395. Actor short checks
passed but did not cover these long sequential states. Original sources, proofs,
integration patch and independent failure are preserved under revisions/; no
threshold relaxation, period/clock/map/dimension change or selected replacement.

Serialize FP32 base_alpha=exp(originalFP32log_decay) once in the model state_dict.
Parallel microblocks and literal stream both use that same stored coefficient.
Powers have nonnegative integer exponents<=64, avoiding unrounded-log exponentials
and inverse powers. The old nominal continuous half-life remains a design intent;
the stored real FP32 coefficient defines actual discrete recurrence/native parity.

Define c_j=1-base_alpha_j, fixed and serialized, and scaled state S'_j=c_j*S_j,
Z'_j=c_j*Z_j. Recurrence is S'_t=alpha^gate*S'_(t-1)+c*gate*phi(k)*v and likewise Z'.
Retrieval is q*S'/max(q*Z',c*1e-6). Since c>0, exact-real output equals original
normalized attention including denominator-floor behavior, not only where active.
Valid zero-history scaled Z/S bounds become9 rather than9/(1-alpha). Tolerances
and all parameter shapes/seeds/maps/gates/headhalf-lives stay fixed. FP32 arithmetic
still differs from ideal reals, and a fresh independent long grade is mandatory.

New checkpoint grammar has base_decay/norm_scale buffers. Flat state positions and
byte sizes remain same, but their S/Z contents now represent scaled memories.
Old checkpoint/state payloads must not be silently loaded under the revised model.
Native implementation must serialize and use identical base and scale, scaledfloor,
and explicit origin/ordinal/priceprevious-known semantics. No Rust/native patch.

Before v4 checks, add stronger literal serial full20k cases for each arm at fixed
seed534/input211/output23: quiet after actual row1event and eventeachnonoriginrow.
Original actor cases and tolerances are retained; timeout is inconclusive, never
license to skip failing long cases. No competition data/targets/scores/GPU used.
