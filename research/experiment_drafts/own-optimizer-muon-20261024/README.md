# Own optimizer research (Muon-inspired) — 2026-10-24

User-requested evening stream: "think about the Muon transcript; I think we can
research on making our own optimizer; spawn parallel subagents to think about it
and train few experiments." Analysis plus bounded synthetic experiments only; no
competition-data fits, no provider jobs, no uploads.

- [RESEARCH_PROPOSAL_R1.md](RESEARCH_PROPOSAL_R1.md) — scout proposal from the
  user-supplied Muon transcript, repo archaeology and an optimizer-literature
  scan. Key repo facts established: `src/wnn/optim.py` `MuonAdamW` is our own
  implementation with the Kimi/Moonlight RMS-matching scale (0.2·sqrt(max(m,n)))
  and decoupled weight decay already present, per-gate-block matrix split, and
  configurable scope (`rnn`/`hidden`); **its Newton-Schulz `orthogonalize()`
  casts to bfloat16 on CUDA, so every historical GPU Muon fit violated the
  strict-FP32 policy** (CPU branch is FP32). In-repo Muon evidence is mixed
  (strong unmatched parents; one matched 350-sequence screen where Muon-WMSE
  trailed AdamW by ~0.005). Eight candidate cards C1–C8 with falsifiers and a
  prioritized plan; honest prior: endpoint value small and uncertain
  (±0.001–0.005 WP class).
- [experiments/](experiments/) — the bounded mechanism probe:
  [synth_optimizer_screen.py](experiments/synth_optimizer_screen.py)
  (self-contained strict-FP32 CPU screen: synthetic causal task mimicking our
  per-batch-normalized weighted-MSE regime and clipped-target weighted-Pearson
  metric; tiny 2-layer GRU; arms adamw / muon / muon_hybrid (hh-only,
  DeepSeek-style) / muon_ns2stage (two-stage Newton-Schulz coefficients)),
  [run_grid.sh](experiments/run_grid.sh) (12 sequential fits, one core, ≤900 MB,
  20 s cool-downs under `scripts/limited`), results in
  [experiments/results/](experiments/results/), and
  [experiments/EXPERIMENT_R1.md](experiments/EXPERIMENT_R1.md) (honest report;
  synthetic screen only justifies or kills a follow-up on real cached data /
  Kaggle GPU).

Executed follow-ups (same evening): **C5 matched effective-rank diagnostic —
done** ([results/C5_RANK_D02_R1.json](experiments/results/C5_RANK_D02_R1.json)):
on the sha256-verified matched D02 checkpoints, Muon raises `weight_hh_l0`
stable rank 0.0685→0.4095 and `weight_ih_l0` 0.1417→0.4076 (entropy-erank
+0.009–0.015) — the rank mechanism is real on competition weights — **while the
same Muon arm lost that matched screen by −0.0054 WP**, so the Parallax-style
"optimizer unlocks quality via rank" inference does not transfer at
small-screen scale. **C1 strict-FP32 compliance patch — done**: `orthogonalize()`
in `src/wnn/optim.py` no longer casts to bfloat16 on CUDA (always fp32);
ruff/structure checks pass and all 14 `tests/test_optim.py` tests pass; this
changes future training numerics only (serving never used the optimizer).
Remaining best local candidate: the C4 batch-mass screen (untested
`loss_norm="running"` branch); the Kaggle C2 per-group scope panel is
post-competition unless C4 surprises before ~Nov 5.

Linked from the [ideas index](../../unified/ideas/README.md) and the
[value/blind-spot review](../../unified/reviews/independent_value_review_20261024_r1/SYNTHESIS_R1.md).
Ledger rows: see `research/hypotheses.jsonl` / `research/experiments.jsonl`
(H-OPT-OWN-MUON-20261024-R1, E-OPT-SYNTH-SCREEN-20261024-R1).
