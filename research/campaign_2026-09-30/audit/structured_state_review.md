# Independent structured-state CPU review

PASS for corrected standalone CPU arithmetic and state interface before
integration. This does not approve a competitive gain, GPU resource usage,
native implementation, final package runtime, or public-score expectation.

The corrected model SHA256 is
`227a5b8fe8ada7b34d61463ad2f022a0d802011aa985b215ef880d5a469c1756`.
`structured_state_grade_v2.json` records eight passing cases: only decay
initialization differs between matched arms; both arms match independently
written scalar/2x2 forward recurrence and every FP64 parameter/input/initial-state
gradient; FP32 full/chunk1/17/64/2000/streaming results agree on a 2049-row
timeline with nonzero initial state; prefix causality, fresh reset, separate
versus batched sequence independence, and CPU autocast FP32 state invariants pass.
A 20,000-row half-life8192 impulse passes against analytical serialized dynamics.

The original model failed long-horizon stream/FFT state parity, saved in
`structured_state_grade_v1.json`. At 20,000 rows its maximum discrepancy was
0.00013569, with two coordinates exceeding the frozen mixed tolerance by up to
2.388x. FFT used unrounded log-radius powers while streaming repeatedly used
rounded real transition coefficients. The actor corrected both paths to the
same rounded scalar/rotation coefficients; this review did not edit actor code.

After correction the long-horizon discrepancy is 0.0000042617, maximum bound
ratio0.213, with no violations. FFT matches the analytical powers of the
serialized scalar/rotation matrix within 0.0000001193. Compared with an ideal
continuous-rho response, FP32 coefficient quantization still shifts the scalar
impulse by up to0.00012821; that is recorded as a diagnostic.

The second protocol explicitly bases the analytical impulse on the actual FP32
serialized matrix rather than ideal unrounded parameters. All forward/gradient
and mixed tolerances are unchanged. This correction occurred before any actor
GPU fits. The first protocol and original independent grader source are retained
as `architecture_cpu_grading_contract.json` and `grade_structured_state_v1.py`;
the corrected protocol is `architecture_cpu_grading_contract_v2.json`. A different
native coefficient ordering or precision would need new parity evidence.

The grade was a fresh durable CPU job
`c20260930_independent_structured_grade_v2`, one thread, 2GiB scope, no GPU
visibility or CUDA initialization. It took10.35 seconds, approximately691MiB
peak RSS. The independent reference uses direct real recurrence math and shared
model tensors, without calling actor recurrence helpers. No competition targets,
predictions or masks were read by either grade.

Integration must retain the checked flat tuple, state precision, coefficient
order, mean/std buffers, reset behavior, and original auxiliary output count.
Multi versus single remains an initialization/optimization comparison because
both arms learn the same range of decay scales. Freeze chronology and matched
recipe before model fitting. The scorer and its source hashes are separately
frozen in `campaign_score_freeze.json`.
