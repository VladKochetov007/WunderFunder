The synthetic state-gate instrument passes. This is interface validation, not an
actual scientific replay, objective optimum result, WP gain or GPU approval.

`capture_toy` owns detached copies of raw student and lagged EMA weights, both
incoming hidden states, warm optimizer moments/current group settings/name order,
pre-forward RNG, clocks, clipping and detached chunk labels/teacher streams.
`validate_capture` rejects missing hidden state, attached EMA state, unsupported
optimizers and altered parameter ordering. `replay` reloads every arm independently
and returns four component gradients, total gradient, clipped gradient and the
one-step parameter delta. The adapter explicitly admits synthetic CPU FP32,
unfused AdamW and disabled scaler/scheduler only. It is not a generic live-trainer
capture plugin; an actual backend must implement and validate those semantics.

The tiny GRU has a shared trunk, two main outputs, three structural heads and two
next-change outputs. Four genuine warmup updates precede capture; detached EMA
weights evolve with decay .999, and hidden states carry the historical forwards.
Main uses abs(clipped original target), clipped average of scaled cached teachers,
strict bootstrap8 survival and alpha .5. EMA uses unclipped detached teacher output,
row-weight denominator with two-output broadcasting and coefficient1. Structural
coefficient .3 multiplies the sum of three CE losses, not their average. Next
coefficient .3 multiplies MSE+BCE on known rows. Final row weights and next labels
are supplied synthetic inputs, not a recreation of propensity/validation masks
or the target generator. A separately coded rowwise derivative oracle checks all
active coefficients, clipping and strict/end-of-sequence bootstrap cases.

For the three slots, aligned scalar mass/mean(mass) is approximately
[.160753, .688396, 2.150851]; the sham uses exactly that multiset in order
[2.150851, .160753, .688396]. The parameter-update differences are .130640,
.044362 and .102624 in L2 norm despite identical pre-step moments. All three clip
actively to approximately .12. The small clip is a deliberate stress setting;
the real export has clip1 and Muon/fast, and this toy makes no parity claim.
The equal-mass control has distinct inputs but scalars [1,1,1], giving exact
arm identity. Repeat and serialized replay identity errors are zero; identity
tolerance is max(1e-7, ten times measured repeat error).

Equal weights AND incoming hidden states yield EMA gradient maximum 6.90e-9.
Equal weights alone leave EMA gradient norm .078732. The actual lagged toy teacher
gives EMA gradient norm .783125. Clearing optimizer moments leaves the gradient
exactly identical but changes the step by .212709. These positive controls
demonstrate that the interface can falsify mistaken state equivalence. No arm
evolves a separate multi-step trajectory; this is independent local replay.

One pre-freeze run failed the overly tight arithmetic comparison in the added
sequence-end case. See AMENDMENT.md and retained attempt2 files. Identity/mismatch
thresholds were preserved; different FP32 summation orders now have an explicit
scale-dependent four-epsilon allowance. Final results and logs include every
measurement. Final run takes about1.4 seconds inside Python, with peak RSS below
700MiB. Admission/import overhead and all development runs are separately recorded
by the CPU wrapper; STATE.json records the exact final command and measured cost.

Actual export inspection is metadata-only under FakeTensorMode:
`artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt`, SHA256
`d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e`.
Its keys are model,args,n_in,epoch,readout,readout_tilted,provenance.
Actual eval_ema=true and the trainer selection clause choose evaluation EMA
weights. This is not absence of every EMA weight: the missing information is
the contemporaneous RAW student/lagged EMA pair, optimizer state, BOTH incoming
TBPTT hidden states and exact chunk/RNG/scaler/scheduler/update-phase inputs.
Aggregate provenance counters do not recover that state. Separate parity dumps
can contain raw+EMA weights, but weights alone remain insufficient. Full schema,
provenance identity and missing-input inventory are in export_schema.json.
No actual model forward or full training data was loaded.

Ranked next experiments:

1. Future controlled exact capture, authorized by the controller: capture a few
   fixed TRAIN and validation-in-fit slots with all state above, after scheduler
   selection/before forward and before EMA advancement. Include raw tensor/name
   identities, current optimizer groups/moments, detached raw/EMA hidden histories,
   cached teacher identities, final post-mask weights, future-label masks, scaler
   state and attempted/completed/EMA clocks. Independently replay default/aligned/
   sham from the SAME snapshot, compare every loss, gradient, clipping and step
   against the actual trainer. Preserve all main/auxiliary coefficients. This
   measures the missing full-gradient local comparison most directly. GPU/fused/
   AMP skipped-step handling needs its own observed parity; this toy doesn't cover it.
2. EMA-export surrogate: informative for shape/coefficient/mass diagnostics with
   explicitly chosen hidden state, but cannot recover the actual coupled teacher
   mismatch, optimizer update or training trajectory. Assigning exported weights
   to both student and EMA manufactures a zero-consistency control, not actual
   scientific replay. Do not interpret its null as exhaustion of objectives.
3. This synthetic instrument: useful to test capture/replay plumbing before either
   experiment; it establishes no predictive gain and cannot approve itself.

Runnable command from repository root:
`bash research/campaign_2026-10-03/continuation/round3/objective_state_gate/run.sh`.
This invokes the mandated round2-critic CPU wrapper with one core,2GiB,900s.
The frozen toy_capture.pt can independently be loaded and passed to
validate_capture/replay. Real-state evidence is intentionally rejected by this
adapter. No trainer, grader, queue, prior actor artifacts or live source was edited.
