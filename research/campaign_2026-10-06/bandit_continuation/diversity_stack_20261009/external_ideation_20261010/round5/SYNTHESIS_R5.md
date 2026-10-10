# Round 5 synthesis (grok-4.7 plan mode + opencode muse-spark-1.3; Freebuff not run). Brief: HPO + bests + sibling-competition Discord tricks.
Raw: grok_r5.md (10 ideas), opencode_muse_r5.md (10 ideas). Overlap = convergent signal.

## Convergent (both models)
1. **Adaptive/stacked blending of 2 served members** (causal disagreement/vol/age gates, regime lookup table, ridge stacker on [p1,p2,p1-p2,|p1-p2|,clock age]). Zero GPU, CPU probe on banked preds, fit on one fold, apply on another. Serving cost ~0.5us/row on top of 2 q8vnni members (~94us pair loaded - needs the kernel work).
2. **GBM teacher distilled into GRU student as 5th family** (LightGBM on windowed causal features and/or frozen GRU hidden state; stage-2 residual variant). Serve only student. Prior risk: E-092/E-093 earlier found GBM residual overfit; this is distillation, different role. Falsifier: GBM OOF WP < 0.60-0.63.
3. **Pretraining** (next-step/masked-block reconstruction on train+valid inputs only), **HPO t52 recipe through student chain**.
## Unique
- Privileged 8-row-future teacher (oracle 0.711 vs 0.666 causal) distilled as extra family (grok #5).
- Second-stage residual GRU distilled (grok #6); stack distillation per-target (muse #5); anti-curriculum, chrono+highway, time-warp aug (muse #8-10); early-sequence hard routing (grok #3).
- AutoGluon+GluonTS teacher (user idea, H-347): both models rank it LAST (0.0002-0.0004 EV/GPUh; sibling DeepAR/TFT lost to recurrent nets). Keep as parallel low-priority branch after LightGBM teacher shows OOF WP.
## Actions taken
- Backlog: hpo52 replicate s4,s5; hpo52 x prop0 (s3,s4); hpo52 at h256 (s3,s4); prop0 x gated+drop02 (s3,s4). Local queue: hpo52 s4, hpo35 s4, hpo52 s5.
- Agents: stacking/gating probe (CPU), GBM-teacher feasibility (CPU).
