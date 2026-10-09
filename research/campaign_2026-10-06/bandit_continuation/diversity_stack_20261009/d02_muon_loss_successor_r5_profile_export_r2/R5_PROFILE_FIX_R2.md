# R5 profile export producer repair, sealed candidate R2

This candidate keeps the same R5 first-batch profile export correction and the
same WMSE/Huber training recipe as the R1 draft. The R1 offline bootstrap check
failed before source extraction because its template searched for the R4 code
dataset slug while the sealed private asset manifest identified R5. R2 binds
both the bootstrap and worker receipt to the R5 source dataset slug.

The actual R4 WMSE output contains `profile_inputs.npz`; the actual R4 Huber
output does not. The offline producer regression reconstructs an expected
Huber-consumer fixture by passing the captured WMSE first-batch arrays through
the same profile export function. That fixture tests the repaired producer
contract and is not represented as observed Huber input data.

No provider mutation, training, or scientific recipe change is included here.
The R2 packet must pass the sealed bootstrap and entrypoint checks and receive
root review before any private asset creation or kernel push.
