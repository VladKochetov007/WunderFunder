# S4 cleanfold profile AMP-scale diagnostic (R2)

This source-only successor diagnoses the single `nonfinite_steps` event in
profile R1. It runs the same 60-update FP16 profile on the unchanged `fa615`
trainer, data, parent checkpoint, split, and training arguments. A wrapper logs
the GradScaler scale before and after each original `update()` call and delegates
to that method exactly once, returning its original result. It does not alter
model parameters, gradients, optimizer state, learning-rate schedule, RNG state,
or training arguments.

The wrapper's `get_scale()` calls synchronize CUDA. R2 is only a skip diagnosis;
its wall-time and step timings must not be compared with R1. It is not a fit
authorization. The exact wrapper, source/runtime hashes, command, and resource
caps are in `diagnostic_recipe.json`.

R1 remains immutable. Its 60-step profile completed with one `GRAD_STATS`
nonfinite step and no OOM/resource failure, so it did not meet the R1 profile
acceptance contract. The user has since cleared the continuous goal and root
closed new admission. R2 is not launched; a future explicit resume, root review,
and fresh identified profile-only admission are all required before launch. No
fit, export, grade, or precision change is authorized by this packet.
