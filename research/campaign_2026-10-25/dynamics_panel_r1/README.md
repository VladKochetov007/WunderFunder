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

## Paired grade — [dynamics_panel_grade_R1.json](dynamics_panel_grade_R1.json)

Frozen grader run unchanged through [grade_wrapper.py](grade_wrapper.py) (allowlists
`torch.torch_version.TorchVersion` for `weights_only` loading of our own checkpoints, the
same PyTorch 2.6 issue as the Oct-24 C5 diagnostic; grader dir added to `PYTHONPATH`).

| Cell | control | aux 0.05 | aux − control [95% component CI] | 90/10 mix − control | rows 99–500 aux Δ |
|---|---:|---:|---|---|---:|
| F0/s1001 | 0.659074 | 0.659535 | +0.000460 [−0.00156, +0.00223] | +0.000226 [+0.000024, +0.000407] | +0.0136 |
| F2/s1002 | 0.684108 | 0.683952 | −0.000157 [−0.00144, +0.00093] | +0.000144 [+0.000020, +0.000256] | +0.0042 |

**Interpretation.** Aux alone: equal-weight mean +0.00015, below the pre-registered
+0.0003 follow-up target and inside seed noise — `insufficient_evidence`, not negative.
The fixed 90/10 mixture is positive in both cells, but any second independently fitted
model would add ensemble value; without a control+control blend this does not isolate the
auxiliary mechanism. The consistent signal is early-sequence: rows 99–500 gain +0.0136 /
+0.0042, suggesting the future-input target speeds state formation after reset, a region
with small metric weight. Two fold/seed cells, exposed folds, 10% memberships: no public
claim. Decision: do not scale this recipe; an early-row-targeted or warm-start variant
would need a new mechanism-specific hypothesis.
