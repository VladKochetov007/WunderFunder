# Optimizer research register

Each family and useful small component is an eligible tree arm. The selection question is **when this update rule may help our model or combination**; implementation is a means to test that question. Untried arms have null effects/costs, not invented rankings.

[Primary-paper scout](GRU_OPTIMIZER_FAMILY_SCOUT_20261008.md) · [Machine-readable register](OPTIMIZER_RESEARCH_REGISTER_20261008_R1.json)

| Arm | When it may be useful | Bounded first experiment |
|---|---|---|
| AdamW | Reference final-student adaptation and matched optimizer control | Keep the faithful final AdamW+OneCycle stage; compare a named alternative at fixed parent/data/objective. |
| RAdam | Early adaptive-step instability or sensitive warmup | One matched GRU512 arm; keep OneCycle and disclose its existing warmup overlap. |
| AdaBelief | Gradient direction varies across heterogeneous targets | Match AdamW budget and define epsilon/decay; compare endpoint and component contribution. |
| Adafactor | Larger teachers make optimizer-state memory significant | Factor recurrent matrix moments with explicit relative-step and update-clipping treatment. |
| Muon hybrid | Recurrent representations benefit from matrix update geometry | Reconstruct strong-parent stage, then compare stage-specific use against AdamW. |
| NorMuon | Orthogonalized updates have unequal row/neuron scales | Add row-second-moment normalization only; same architecture and objective. |
| Polar Express | Polar approximation dominates optimizer time or fidelity | Replace the matrix subroutine only, keeping outer momentum/scaling/decay fixed. |
| Scion norm-constrained updates | Width scaling or bounded operator geometry may change training | One norm/radius recipe versus Muon; do not borrow transformer scaling without a test. |
| Shampoo | Cross-coordinate curvature limits progress | Bound block size/preconditioner frequency, preserve full state for checkpoints. |
| SOAP | Adam directions may improve in a matrix covariance eigenbasis | One recurrent-block SOAP arm versus AdamW/Muon, with explicit refresh frequency. |
| SPlus | Whitening stability and iterate averaging may improve the endpoint | Bound one whitening/averaging recipe; compare full served endpoint, not training loss alone. |
| DASH | Preconditioner implementation costs obstruct useful curvature tests | Test accelerated block preconditioning at fixed mathematical treatment where possible. |
| K-FAC | Recurrent gradient covariance suggests a structured curvature advantage | Small full-sequence component screen; charge curvature-estimator work explicitly. |
| Lion/sign momentum | A low-state direction or different magnitude sensitivity may provide diversity | Matched finite run with sign updates; measure standalone and teacher/blend contribution. |
| SGD/momentum/Nesterov | Coordinate adaptivity may obscure a useful alternative training path | Compare nonadaptive direction at fixed loss/order/clipping, with explicit schedule. |
| Schedule-Free | Schedule sensitivity or averaging may limit adaptation | AdamW-based schedule-free versus OneCycle, exporting its prescribed evaluation weights. |
| Prodigy | Effective step scale is difficult to choose for a new stage | A frozen adaptive-step protocol versus tuned fixed-LR anchor, not a silent OneCycle substitution. |
| D-Adaptation | Learning-rate scale uncertainty across teacher/students | One adaptive-scale control; compare with Prodigy without claiming parameter-free means no choices. |
| Sophia | Diagonal curvature/clipped steps may help noisy regression | Bound estimator frequency/clip constants and count estimator backward work. |
| Muon whole matrix versus per gate | Coupling GRU gates may change useful recurrent representations | Match outer optimizer/schedule and change only grouping; compare fixed endpoint. |
| Muon recurrent-only versus hidden matrices | Input/readout geometry may matter alongside the recurrent core | Compare recurrent-only with one declared wider scope; keep biases on an appropriate vector optimizer. |
| Muon RMS/shape/LR scaling | Width and update magnitude may distort optimizer comparisons | Test explicit LR multiplier/shape scaling while holding grouping and momentum fixed. |
| Stage-specific optimizer handoff | Representation learning and fine-tuning may need different geometry | Faithful handoff anchor versus a named final-stage Muon variant; specify whether state resets. |
| LR schedule and beta1 coupling | A cooldown result can arise from momentum as well as LR | Compare named OneCycle/WSD/SGDR packages first; isolate LR shape with matched beta1 in a descendant. |
| Finite plateau reheating/noise | A fitted trajectory may plateau in a narrow region | Use the Astra bounded trigger recipe; no evaluated-group checkpoint selection. |
| Muon matrix precision/device policy | A hardware numerical path may change feasible training cost | Explicit native-BF16-or-FP32 policy profile; execution evidence alone is not scientific optimizer superiority. |

Evaluate optimizer use separately for teachers, strong parents, final students and complete blends. A worse standalone predictor may remain useful through its representation, residual or disagreement signal. Match or explicitly separate data, initialization, objective, update budget, schedule, grouping, and evaluation exposure.

Select a few informative arms at a time. Registration preserves the rest for later allocation; it does not require a full factorial before producing a faithful submission endpoint.
