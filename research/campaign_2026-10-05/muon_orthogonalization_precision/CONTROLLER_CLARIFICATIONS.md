# E487 diagnostic preregistration clarifications

Registered before implementing or running this diagnostic. Accept the blind
critic's crossed-comparison interpretation and first-step-only limitation.

For deterministic shared synthetic gradient matrices, CPU versus CUDA FP32
orthogonalizer outputs must satisfy elementwise `abs(a-b) <= 1e-5 +
1e-4*abs(cpu_reference)`. Report each matrix's maximum absolute difference
and relative RMS. This arithmetic reference gate is distinct from the
unchanged 2% update-RMS / 2e-5 update-peak native/custom admission gates.
CPU helper versus frozen CPU helper remains byte-exact on CPU. Same-gradient
model clones using the same helper/device must remain byte-exact including
optimizer state; CUDA BF16 helper must match the frozen CUDA E474 helper.

Expected BF16 native/custom admission failure is an observed outcome of this
diagnostic. Record every finite crossed replay and all admission booleans;
do not abort on the first native/custom update-gate failure, since that would
prevent the registered precision comparison. Preserve results even if both
variants fail. Missing tensors, nonfinite values, wrong formulas, failed
same-gradient controls, changed bytes or exceeded resource limits remain
execution/proof failures. Never relax a numerical bound after observation.

Only local scoped diagnostic arithmetic may change; shared production
optimizer modules and the E485 frozen packet remain untouched. An FP32
native/custom gate pass would permit consideration of a newly registered
training experiment, not automatic admission or retroactive success of E485.
