# Frozen common-macrostep credit control

PASS: 58 executable checks. Combo review REPORT/STATE and all 20 previously hashed artifacts are unchanged. This is a synthetic CPU control proof, without a full fit, real targets, empirical credit benefit, recipe attribution or public-score claim.

Both arms execute the actual `RNNModel(3, hidden=6, layers=2, dropout=0, n_out=23)` through the actual trainer `forward_chunk`: two 10-row calls under identical unchanged parameters, concatenate, then evaluate ONE 20-row loss. Only the raw midpoint `.detach()` differs. The encoder is linear; convolutional/stateful encoder alternatives are unsupported. Twenty active synthetic rows are a scaled macrostep interface, not a replay of 2000 real training rows or the warmup/propensity/fit-mask pipeline.

The trainer's four selected loss bodies are compiled from its actual AST, with hashes/line identities in results.json. No separate arm-specific loss implementation exists. Cached teacher inputs are explicitly synthetic detached trainer-facing tensors, not cached exports or reconstructed actual scientific EMA state. Three prior synthetic updates generate genuine lagged EMA weights and distinct detached raw/EMA incoming hidden states under their respective historical weights. These, warm Muon/AdamW state, scheduler, disabled scaler, labels and features are serialized and roundtrip validated. `synthetic_capture.pt` is explicitly synthetic.

The independent output derivative is: main `2w(f-B)/sum(w)` with `w=v*abs(clip(Y))`; EMA `2v(f-U)/sum(v)` broadcast to both targets (no extra divide by two); structural each of 12/3/4 heads `0.3v(softmax-onehot)/sum(v)` (sum three CEs, no divide by three); next `0.6nw(u-(log1p(distance)-3)/2)/sum(nw)` and `0.3nw(sigmoid-a)/sum(nw)`. Denominators use the source floor. Independently loop-reconstructed teacher targets and a VJP of this oracle agree with source autograd: maximum parameter-gradient error 5.96046448e-08, threshold 2e-6. This checks coefficient/broadcast/teacher-clipping identities, not a statistical optimum.

H0 and strict H8 cached-distillation semantics are source-bound. The fixture independently checks first future RAW PAIR change, strictly `next_change > t+8`, sequence-end exclusion, and pair changes that preserve the target difference (the auxiliary next-loss helper differs). This validates synthetic branch arithmetic only; it does not endorse the source comment's unbiased-stopping-law interpretation.

Every forward/EMA output, teacher target and common loss is finite and BITEXACT across detach. Planted second-half-only loss gives nonzero first-half input and midpoint credit in the connected arm, exactly zero in the detached arm, and different parameter/optimizer updates. First-half-only loss produces BITEXACT parameter gradients, optimizer state and updated weights. Mean-of-normalized-halves is measurably unequal to the common full-macrostep loss for this unequal-mass fixture and is rejected.

| Case | Common loss | max gradient difference | first-half input gradient norm: connected / detached | midpoint gradient norm: connected / detached |
| --- | ---: | ---: | ---: | ---: |
| H0 all | 2.77409291 | 0.0239184573 | 0.0122091742 / 0.00993492547 | 0.0380823649 / 0 |
| H0 second | 2.8385098 | 0.0359202027 | 0.00679534627 / 0 | 0.0569194071 / 0 |
| H0 first | 2.64677978 | 0 | 0.0299117286 / 0.0299117286 | 0 / 0 |
| H8 all | 2.74855232 | 0.0239184573 | 0.0119729806 / 0.0096518686 | 0.0380823649 / 0 |
| H8 second | 2.8385098 | 0.0359202027 | 0.00679534627 / 0 | 0.0569194071 / 0 |
| H8 first | 2.5693028 | 0 | 0.0290684029 / 0.0290684029 | 0 / 0 |

Each arm reloads exactly the same warm optimizer/scheduler/EMA/scaler state and advances once: scheduler epoch 3→4, EMA counter 3→4, Adam state steps 3→4; Muon momentum is retained, not reset. Optimizer group ordering/names are recorded. Actual source clip=1 executes before each actual MuonAdamW step, including cases with preclip norm above one. `synthetic_replays.pt` stores preclip/postclip gradients and post-update raw/EMA/optimizer/scheduler/scaler state for independent inspection. Same clock does not imply same updates: the positive and null controls discriminate this explicitly.

CPU GradScaler is disabled: state is empty, scale operation is identity, and one actual step/update call occurs per arm. There is NO active overflow/skip or fused/GPU parity evidence. Other trainer branches, actual teacher-cache binding, native inference, full data exposures and scientific checkpoint replay remain unsupported. No 32-fit launcher, GPU request, live trainer edit or queue/ledger mutation is included. This foundation is informative for checking a future common-clock gradient instrument; it cannot approve its own mechanistic candidate or authorize a fit.

Reproduce: `bash research/campaign_2026-10-03/continuation/round5/credit_control/run.sh`. It invokes `run_cpu.sh round2-critic` with one core/2GiB/900s. Numerical run: 0.961s wall, 0.956s CPU inside main, peak RSS 700336 KiB. See resource_cost.txt for whole wrapper command cost and checks.log for executed results. Initial fixture construction errors were corrected before this freeze; they were not trainer bugs. Source files were read/compiled in place, not copied or modified.
