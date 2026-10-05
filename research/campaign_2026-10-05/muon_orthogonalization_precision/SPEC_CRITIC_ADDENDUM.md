# E487 critique addendum: registered numeric and completion gates

This addendum supplements `SPEC_CRITIC.md` for design SHA-256
`c937a9d6772f70cd56d83e7c9d6bfdde47a52c0e4bb9c66b0ca05a4e29fe8bd6`.
The base critique remains unchanged.

Before implementation begins, the controller registered CPU-versus-CUDA
FP32 orthogonalizer parity at `atol=1e-5`, `rtol=1e-4`, plus per-matrix raw
maximum and relative-RMS reporting. The frozen CPU helper comparison remains
exact. Same-gradient duplicate GPU replays, including optimizer state, must be
byte-identical, and the BF16 shared-gradient result must match the frozen E474
helper exactly. These checks answer distinct questions and must not be
substituted for one another.

The implementation should retain every finite crossed replay and report all
four combinations (native/custom captured gradient × BF16/FP32 orthogonalizer)
even if the first expected native-versus-custom strict update gate fails.
That failure is the diagnostic outcome, not a reason to suppress the other
precision arm. Record each gate boolean and its raw metrics. Missing or
nonfinite values, a changed formula, mismatched clipped shared gradients,
resource-limit violations, or broken same-gradient controls remain immediate
failures and invalidate the diagnostic. No tolerance may be changed after
observing the GPU result.

This resolves the base design's blanket “any strict update gate failure must
stop” wording for expected parity failures only. It does not relax the
registered gates or authorize any training run. The interpretation remains
limited to whether orthogonalizer precision contributes to this one-step
discrepancy; it cannot retroactively admit E485 or establish a predictive
benefit.
