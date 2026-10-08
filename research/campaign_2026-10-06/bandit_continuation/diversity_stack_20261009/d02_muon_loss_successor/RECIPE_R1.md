# D02: Muon × supervised loss, two-cell successor

Status: frozen science/argv, **not admitted and not yet a sealed runnable source**.
Root owns source review, allocation and launch. Sol owns delivery through evaluated
artifacts after admission. No fit, device profile or score is claimed here.

Add only `d02_muon_wmse_r1` and `d02_muon_huber_d1_r1`. Reuse the current GPU R2
AdamW WMSE/Huber cells as controls; do not retrain them. Compare each optimizer
within its loss and each loss within its optimizer, then report the difference
of loss contrasts as an exploratory interaction. All scores/contrasts are unknown.

Retain GPU source `8924deec1eaf1a50cb31d4c42e042d24093b4be6` as the trainer/cache/
export base. Both new cells must match the completed GPU contract's fresh initial
state and shared normalization, epoch orders and 220 pre-update LR/beta1 rows.
The shared initialization is our independently generated **pre-fit** state, never
an AdamW-trained checkpoint. Recreate it with seed `20261009` and require exact
tensor/normalization equality to the producer artifacts before update zero.

Keep raw112 + causal event99 inputs, linear projection, GELU, GRU256×2→2,
FP32/TF32 off, no dropout, batch32, chunk2000, two epochs, drop-last false and
220 attempts (11 batches × 10 chunks × 2). Carry detached sequence state across
chunks; reset between batches. Fit350, retain28 unused fit-valid and evaluate468
whole sequences, using membership SHA
`dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd`, cache manifest
`7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b` and cache key
`949587f77794935c08589bd5d5e81641062de547526b92fc1b01bc4109cd82a6`.
Fold0 remains exposed. No checkpoint selection on outer labels: fixed epoch1.

Targets clip±2; weights are absolute clipped labels; fit mask row≥99. Losses are
chunk-normalized WMSE and weighted Huberδ1 with the existing trainer's numerator
and denominator semantics. Official evaluation stays `global_wp_clip2_v1` with
`is_scored & row≥99`, fixed masks and clipping±2. No propensity, auxiliaries,
distillation, optimizer switching, special initialization or other new factors.

## Optimizer and schedule audit

D02 uses our independent `MuonAdamW` implementation: Newton–Schulz five steps,
coefficients `(3.4445,-4.7750,2.0315)`, epsilon1e-7, Nesterov-form momentum and
RMS scale `0.2*sqrt(max(block_rows,columns))`. Scope `rnn`, gatewise `all`:
the four `gru.weight_ih_l*`/`gru.weight_hh_l*` matrices each use three gate blocks.
Input projection, head and all biases use AdamW. No teammate code or weights.

Both groups have peak LR0.001, beta2=0.999, epsilon1e-8 and decoupled WD0.01.
Head and Muon LR multipliers are1.0. OneCycle has warmup0.05, initial divisor25,
final divisor100, default cosine annealing and cycled beta1/momentum0.95→0.85→0.95.
Starting Muon momentum0.9 is overwritten by the shared scheduler before the first
update; retain the actual pre-update trace rather than claiming fixed momentum.
Every D02 group must match the control's single group's LR/beta1 at every attempt.
Thus nominal optimizer LR differences are **zero**; Muon's transformed update is
the treatment. Remove `--fused-adam`: this combined optimizer implements its AdamW
group steps directly; do not claim identical optimizer kernel throughput.

Inspected existing strong-parent authorization
`continuous_execution_E529/H230_SEED9_LOCAL_ROOT_FIT_AUTHORIZATION_20261008_R1.json`
instead uses hidden scope, whole packed matrices (`none`), Muon LR multiplier3
(peak0.003 versus AdamW0.001), starting momentum0.95, fp16, B16/C1000, all-data and
additional teacher objectives. It is ancestry/method evidence, not a D02 control.
Those choices are separate future optimizer-policy arms, not silently included.
The audited H273 source README and `src/wnn/{optim,training_dynamics}.py` under
`continuous_execution_E529/h230_source_v5_h273_raw_weight_parity` document the
independent dtype and grouping controls. D02 explicitly chooses ortho policy
`fp32`; model FP32 alone does not control this matrix math.

## Delivery prerequisites and outputs

GPU R2's optimizer currently performs unconditional CUDA BF16 orthogonalization
and does not accept `--muon-ortho-dtype-policy`; H273 exposes that audited control
but lacks the parsed-cache/loss-pair exports. Before admission, create a narrowly
reviewed successor source combining the current cache/export path with the
existing explicit FP32 orthogonalization policy. Do not execute the argv against
either unmodified source or bypass this gap. The existing R2 source already
supports recurrent-only/per-gate grouping and equal group rates; no LR/grouping
extension or new runner framework is needed. Freeze the resulting source hashes.

`exact_argv_recipe_r1.json` contains only the two fit and two prediction argv arrays;
`${...}` tokens are literal runtime bindings, not shell substitution. Bind exact
mounted paths/receipt hashes and the reviewed successor project before dispatch.
Use the existing finite GPU packet/profile/resource supervision, root quota and
admission. Require an actual-device finite loss/unscaled gradient/update/reload
gate, shared initialization/order/normalization equality and both-group schedule
trace equality before useful fits. Hardware failure is execution evidence.

Save `model_ep1.pt`, full `training_state.pt`, normalization/provenance, ordered
`predictions_valid.npy` with source-qualified membership/NPY sidecar and checkpoint
hashes, per-update loss/LR/beta1/global and family unscaled grad norms/clip/skip,
phase timing and memory peaks. Buffer diagnostics as in R2. Preserve all attempted,
completed and skipped counts; no forced 220-completed claim. Root grades all four
endpoints and useful pair/equal-four mixtures with the fixed official scorer and
component-resampled uncertainty. No public score forecast follows from this
exposed-fold screen. Keep weights/predictions/resume evidence per
`ARTIFACT_RETENTION_R1.md`; no automatic deletion or quality promotion.
