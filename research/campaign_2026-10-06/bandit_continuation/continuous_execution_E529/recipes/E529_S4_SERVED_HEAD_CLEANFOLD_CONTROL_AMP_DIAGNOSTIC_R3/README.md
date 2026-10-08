# S4 cleanfold profile AMP-scale diagnostic (R3)

R3 preserves the R2 60-update FP16 diagnostic and corrects its runtime source
identity check. R2 stopped before trainer startup because the verifier asked
`inspect.getsourcefile(torch.optim.AdamW)`, which resolves `torch/optim/adamw.py`,
while its expected digest was for `torch/optim/adam.py`. R3 binds both modules:
the AdamW configuration class module and the unwrapped inherited AdamW step
implementation. It also retains the exact GradScaler module pin.

The diagnostic still uses the unchanged `fa615` trainer, data, parent
checkpoint, split, optimizer, precision, and trainer arguments. Only the
wrapper and unique output name change from R2. The wrapper logs the GradScaler
scale before and after every original `update()` call and delegates to that
method exactly once, returning its original result. It does not alter model
parameters, gradients, optimizer state, learning-rate schedule, RNG state, or
training arguments.

The wrapper's `get_scale()` calls synchronize CUDA. R3 is a skip diagnosis;
its timing must not be compared with R1 or R2. R1 remains immutable and met no
zero-nonfinite profile gate; R2 remains immutable and failed before trainer
startup on the instrumentation source-selection mismatch. R3 is source-prepared
only: root review and a fresh identified profile-only admission are required
before launch. No fit, export, grade, or precision change is authorized here.
