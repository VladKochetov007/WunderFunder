# Future-input dynamics auxiliary panel — execution, 2026-10-25

Executes the reviewed, never-run four-fit panel `E-FUTURE-INPUT-DYNAMICS-20261021-R1`
([draft](../../experiment_drafts/dynamics-aux-20261021/README.md),
[packet](../../campaign_2026-10-21/next_cycle_r1/dynamics/README.md)): cold H512×1 GRU,
10% frozen fit memberships, 3 epochs, fixed CP2, aux weight 0 vs 0.05 on F0/seed1001 and
F2/seed1002. It was deferred on Oct 22 only because a 3.37 h projection exceeded a 3 h
stage cap; tonight's user allocation ("utilize gpu, experiment!") covers it. Mechanism is
the same family as FTT's 52 lead features, which carry most of TL3's public score mass.

- Source: corrected frozen worktree `781deb6a5` (FP32 window-sum fix; profile passed).
- [run_panel.sh](run_panel.sh) runs the four unmodified recipes (script paths resolved into
  the frozen tree), then the frozen paired grader, copied byte-identically into
  [grader/](grader/) so its `parents[4]` root resolves to this checkout (hashes match the
  source commit; `dynamics_auxiliary.py` hash identical in both trees).
- Units `wnn-dynpanel`, `wnn-dynpanel-thermal` (freeze ≥97 °C / thaw ≤88 °C); ≤14 GB RAM,
  no swap, ≤1.5 CPU, ≤4.5 GB CUDA allocator. Status: running.

## Execution notes

- Attempt 1: F0 control fit (4,620 updates, ~15 min) and export succeeded; the recipe's
  per-run `score_run.py` step failed because that script resolves its root from the frozen
  source tree (`.../dynamics-screen-corrected-20261022/cache/valid_targets...` missing).
  Path issue only. The runner now skips completed fit/export steps and runs score steps with
  the main checkout's `scripts/score_run.py` (same official metric); the paired grader is
  unaffected.
- F0 control (aux weight 0) official WP **0.65907** (10% membership, cold H512×1, CP2;
  per-run diagnostic, not the paired grade).
- F0 aux 0.05 official WP **0.65953** vs control 0.65907 (+0.00046, per-run diagnostics; paired interval pending the frozen grader after F2).
