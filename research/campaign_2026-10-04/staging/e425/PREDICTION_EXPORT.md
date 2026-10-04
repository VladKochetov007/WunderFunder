# E425 soft-route prediction export

`scripts/predict_e425_soft.py` is a source-only fixed-checkpoint exporter for
the future H106 soft-route run. It mirrors the E424 recurrent VALID exporter
while forcing both trainer and model factory through the E425 staged overlay.
No prediction was launched from this source task.

The script resolves the repository root from its own location, so execution
does not depend on the caller's working directory. Before loading the trainer,
it orders `staging/e425/src`, repository `src`, repository `scripts`, and the
repository root on `sys.path`. The final root entry makes top-level helpers
such as `utils.py` available to imported root utilities, following the
explicit root-path handling used by the E462 child scorer. It also moves the
staged `wnn` package directory to the front of any cached package path,
evicts cached root `wnn.specialist` and `wnn.specialist_diagnostics` modules
and stale package attributes, then imports the staged trainer under a unique
module name. Runtime assertions verify the imported specialist path and class
identity used by `make_model`.

The source closure covers root WNN Python dependencies, staged WNN overrides,
all staged scripts, the exporter, and root `utils.py` when present. It is
hashed before and after trainer loading and checked again after inference.
The checkpoint is similarly hashed before load and after prediction. The
exporter accepts only a soft specialist checkpoint with the fixed epoch-3
policy, 211 inputs, 23 outputs, and unquantized/unfactorized state. It writes
all VALID target-cache row groups in FP32, rejects nonfinite outputs and
existing output/sidecar/partial files, and records checkpoint, prediction,
model-source, row-group, and code-closure identities in the sidecar.

The native soft-model package/export path and prediction parity remain later
deployment checks. They are not blockers for the scientific H106 fit. A
root-owned static/import smoke is still needed before using this exporter.
