# E471 bounded recurrent candidate × optimizer

All six fixed scientific fits completed under the common full-FP32 policy. The official scorer validated source identity, wrapper precision flags, selected targets/masks, update/parameter budgets and all pooled/per-target scores. It retains seven primary within-study contrasts and two historical backend diagnostics.

| Alpha / optimizer | Pooled WP | t0 | t1 |
|---|---:|---:|---:|
| m10__adamw | 0.4887901 | 0.5010069 | 0.4765732 |
| m10__muon_adamw | 0.4971020 | 0.4991056 | 0.4950985 |
| zero__adamw | 0.4822645 | 0.4987989 | 0.4657301 |
| zero__muon_adamw | 0.5054955 | 0.5055565 | 0.5054344 |
| p10__adamw | 0.4793591 | 0.4945869 | 0.4641314 |
| p10__muon_adamw | 0.5116375 | 0.5086565 | 0.5146186 |
| cudnn_tanh__adamw | 0.4822594 | 0.4987939 | 0.4657249 |
| cudnn_tanh__muon_adamw | 0.5054984 | 0.5055664 | 0.5054304 |

| Contrast | Delta WP | Conditional 95% interval |
|---|---:|---:|
| m10__adamw_minus_zero__adamw | +0.0065256 | [+0.0003891, +0.0109914] |
| m10__muon_adamw_minus_zero__muon_adamw | -0.0083934 | [-0.0128293, -0.0047516] |
| p10__adamw_minus_zero__adamw | -0.0029053 | [-0.0064516, +0.0019704] |
| p10__muon_adamw_minus_zero__muon_adamw | +0.0061420 | [+0.0028803, +0.0101905] |
| m10__muon_adamw_minus_m10__adamw | +0.0083120 | [-0.0108818, +0.0262744] |
| zero__muon_adamw_minus_zero__adamw | +0.0232310 | [+0.0041524, +0.0431232] |
| p10__muon_adamw_minus_p10__adamw | +0.0322784 | [+0.0116651, +0.0535106] |
| zero__adamw_minus_cudnn_tanh__adamw | +0.0000050 | [+0.0000012, +0.0000086] |
| zero__muon_adamw_minus_cudnn_tanh__muon_adamw | -0.0000029 | [-0.0000153, +0.0000096] |

The negative candidate perturbation helps AdamW (+0.006526) but hurts Muon (−0.008393); the positive perturbation helps Muon (+0.006142) while its AdamW point delta is negative and its interval crosses zero. This challenges a universal candidate-activation preference and warrants retaining both signs and both optimizers at additional fixed seeds. It does not identify periodic data: the monotone bounded map changes recurrent candidate curvature and local derivative.

Each arm uses n = tanh(v) + alpha*sin(pi*tanh(v)), alpha −0.1/0/+0.1, unchanged sigmoid reset/update gates and the PyTorch reset-after-hidden-affine bias placement. The one-layer GRU32 still has 8,235 parameters and 474 updates on 24 fit / 16 historically exposed components. Conditional component intervals omit training-seed and research-selection uncertainty; there is no multiplicity correction or public prediction.

Default CUDA parity failed before updates; the independent policy diagnostic exposed the matmul/cuDNN precision asymmetry. A new wrapper disabled TF32 in both backends, and a fresh six-update probe passed the unchanged output/state/gradient thresholds. All six fits share that amended computational policy. The old failed probe, quarantine, later post-run diagnostic edit and exact executed-source recovery are retained. No tolerance or scientific equation was weakened.

The two alpha-zero comparisons with historical E470 cuDNN controls are diagnostics mixing backend/precision regimes. Their score differences are only +0.0000050 (AdamW) and −0.0000029 (Muon). Do not treat these tiny differences as a practical improvement, bit-exact training reproduction, or a fresh control for the full public model.

Full worker: 191.7076 guarded GPU-job seconds, observed GPU peak 0.3712GB, RAM 1.7927GB, exit0 and verified teardown. No worker OOM or increase in historical parent OOM counters. The unchanged original 80deviceGPUh ceiling, local24-hour block and parent12CPU/30GBhard/27GBsoft/zeroSwap limits remain. Cost also includes the preserved failed/profiling attempts; see the ledger.

Artifacts: `RESULT.json`, `FP32_POLICY_PREREG.md`, `PRECISION_DIAGNOSIS.md`, `GPU_PARITY_INCIDENT_V1.md`, `GPU_PROBE_FP32_RESULT.json`, fixed sealed specs and immutable runner receipts. Weights/selected exports remain locally under `runs/six_v1/` as bulk evidence.
