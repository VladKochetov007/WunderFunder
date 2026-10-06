# Research contract — quiet-regime campaign (2026-10-02)

Question: can a representation change make class-3 predictions robust to the quiet (low trade-activity) regime that the hidden test over-represents, and lift public WP above 0.6641 toward top-1?

Deliverable: competition ZIP (1 vCPU, 60 min, deterministic, causal, reset per seq_ix), native C++ engine where possible.
Baseline: V1 public 0.6641 (also e6a 0.6641). Midpoint probe set: full 0.6639, c3zero 0.5473, c3only 0.4017.
Tier: B for local (chain-fold OOF, multi-seed needed), C for transfer to test (public LB is the only test proxy; ~5 subs/day, user-controlled).

Evaluator (frozen): research/iterations/2026-10-02-slow-fast/tailrank.py on fold 0+1 OOF, groups all/c012/c3/c3quiet(z<-1, 121 seqs)/c3active; quiet_z from make_weights.py (train-standardized i0_v12_nuniq, i0_dp0_std, -a5_mean, -a3_std).
Primary metric: c3quiet WP; invariant: "all" WP must not drop by more than 0.001 vs matched control. Matched control = same recipe, same seed, same sampler (qw0).
Falsifier for a branch: c3quiet gain < +0.004 on the 2-fold pool, or all-WP loss > 0.001.
Noise: single-seed subset noise ~±0.003 assumed; any promotion needs a second seed.

Branches:
 A reweighting (done): beta=2 c3quiet +0.0046, c3active -0.0063, all -0.0001 -> capacity trade, not new info. Kept as minor add-on.
 B time-stretch augmentation (labels change only on i0 price-part changes => inserting no-event rows is label-preserving).
 C event-clock GRU (state updates on i0 events, elapsed-time inputs) — after B.
Budget: GPUs 0,1,4,5,6 on beleriand (shared; check before launch), ~10-70 min per fold job.
Rules: no public submissions without the user; GitHub uploads paused; homelab untouched.
