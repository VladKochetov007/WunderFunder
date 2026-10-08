# Plan to 0.700: external-model ideation round (2026-10-08)

Status: ideation only. Nothing was trained, pushed, rented or committed. All effect sizes below are ideator estimates, not measurements. Treat them as priors to be overwritten by fold-0 evidence.

## 0. What ran, and what did not

| Tool / route | Model | Lens | Outcome | Receipt |
|---|---|---|---|---|
| Grok native CLI 1.0.46 | grok-4.7 | microstructure, target law, quant brainteaser | Completed, exit 0, 8 numbered ideas, about 6.5 min. `grok models` printed "You are not authenticated" yet the run succeeded | `grok_receipt.json`, `grok_raw.md` |
| OpenCode 1.18.35 | opencode/muse-spark-1.3-contributor-free | ML architecture, optimization, ensembling, distillation under budget | Completed on the third attempt, exit 0, 8 ideas. Attempts 1 and 2 exited 1 with no output (flag-order issue, cause not isolated). Text output, not JSON | `muse_receipt.json`, `muse_raw.md` |
| OpenCode | opencode/nemotron-3-ultra-free | data/evaluation structure, causal adaptation, metric calibration | Timed out at 600 s, exit 124, no output. Not retried (budget) | `nemotron_task.md`, `nem_exit.txt` |
| Freebuff 0.2.15 | n/a | n/a | Not run. `--help` documents only interactive use plus `--cwd`, `--continue`, `--trust-agents`. No non-interactive prompt flag | none |

Passages (library pool, seeded `random.Random(20261008)`, random page and 1200-char window): MathStatLectures.pdf p300 (Grok P1), RLAlphas.pdf p4 (Grok P2, Muse P2), Markowitz_1952_Portfolio.pdf p5 (Muse P3). The file list excluded personal and credential-bearing folders. Selection data is in `passages_selected.json`. Passages were judged as weak sources: the random excerpts had no direct correspondence to the label law, and both models said so.

Working-set note: this round used 2 successful external providers, not 3. Muse ideas and Grok ideas are independent prompts, so agreement between them is weak evidence only.

## 1. Already-tried list used for dedup (from BRIEF section 4 and the bandit tree)

Items 1 to 18 in BRIEF.md. The most relevant for this plan: privileged-belief and privileged teacher (3), readout mixture and direct-label readout (12), residual memory (10), quiet-output hold (10, screen failed), Net2Net widening (7), cadence and denominator (14), optimizer families (15), and same-recipe seed ensembling (+0.0004 public).

## 2. Clusters (dedupe vs tried)

Cluster A, class-3 latent identity and algebraic readout. Grok 1, Muse 1. Two families proposed the same head independently. Novel vs readouts already tried (they were learned readouts, not analytic subtraction). Agreement here is weak evidence, but it makes the identity-audit cheap to check.

Cluster B, quiet-row / hold / state-clock. Grok 2 (hold F for H rows), Grok 8 (top-K threshold gate). Grok 2 is a reframing of the failed quiet-output hold; the held object is F, not the output, so it is not identical, but the risk is high.

Cluster C, seed formulas and residual learning. Grok 3 (frozen seeds, narrow residual GRU). Muse 6 (disagreement-selected heterogeneous teacher pool, drawn from the same RLAlphas passage). The seed-pool lesson, that diversity must come from an expanded operator set and not more seeds of one recipe, matches our local +0.0004 same-recipe finding. Strongest transfer in this round.

Cluster D, heterogeneous blends and low-rank capacity. Muse 2 (covariance-weighted blend of two branches), Muse 5 (low-rank wide student). Muse 5 is novel relative to dense Net2Net widening (item 7). `--recurrent-rank` already exists in `scripts/train_rnn.py`, so the flag may cover part of it; semantics must be verified before use.

Cluster E, cointegration, innovation basis and state geometry. Grok 4 (bottom-eigenmode cointegrating residual of i0 and i1 mids), Grok 5 (rank-4 innovation basis with independent scalar filters), Grok 6 (tick-unit book histogram), Grok 7 (trade sub-clock inside the row). Novel as stated. Grok 4 and 5 use cross-instrument structure we have not explicitly tested. Each is a feature change, so a CPU linear-readout screen is possible without GPU.

Cluster F, distillation, privileged targets and curriculum. Muse 3 (privileged-F teacher with variance weighting), Muse 4 (two-stage metric-weight curriculum), Muse 7 (optimizer switch late in training). Muse 3 extends item 3 with an uncertainty weight. Muse 4 and 7 are interactions of cadence and optimizer (items 14 and 15), explicitly listed as not_tested as a joint interaction.

Cluster G, memory. Muse 8 (EMA side state with erase-write). Close to residual-memory (item 10, prepared but unrun). Reuse that prepared run rather than building a new one.

Not clustered with anything tried: Grok 7 (trade sub-clock), Grok 6 (tick-unit histogram). Grok 7 has no defensible passage correspondence; it is a structural idea from the label clock. Keep it only if the fold-0 check shows four distinct trade prices.

## 3. Ranking and cheapest first test

Rank by expected value per GPU-hour, with CPU-only screens first. EV is a judgment, not a number.

1. Cluster A identity audit plus oracle ceiling (Grok 1, Muse 1). CPU only. On fold-0, measure the identity residual of current predictions on class-3 rows and the weighted Pearson ceiling if F is replaced by the label-implied value on those rows. If the current heads are already coherent and the ceiling is small, Cluster A is closed cheaply.
2. Blend weights from fold-0 error covariance (Muse 2). CPU only, on existing fold-0 predictions. Compare uniform, inverse-covariance with shrinkage, and the current best. Coordinate with the blend-lab-2 agent so the same panel is used.
3. Top-K threshold gate sweep (Grok 8). CPU only, post-hoc on predictions. Weighted and equal-weight correlation at 20, 50 and 100 percent of training weight. Pre-register on training weight only.
4. Causal cointegrating residual (Grok 4) and rank-4 innovation basis (Grok 5). CPU linear-readout screen on fold-0 rows, same fit membership. Diagnostic: residual cross-correlation of coefficients.
5. Tick-unit book histogram (Grok 6). CPU feature screen. A quick check first: are price-like columns already relative? If so, the idea is void.
6. Quiet-row hold on F (Grok 2). CPU post-hoc screen on quiet vs active rows. Escalate only if the quiet-slice gain is above the noise band. Given the failed quiet-output hold, do not spend GPU here before the CPU screen.
7. Disagreement-selected teacher pool (Muse 6). First a GPU-free check: among existing teachers, measure pairwise disagreement and whether the disagreement predicts student gain. Then a fit (see E4).
8. Low-rank wide student (Muse 5). Verify `--recurrent-rank` semantics first (rg the script), then a matched-CPU-minutes probe.
9. Late optimizer switch (Muse 7). Fork from a mid checkpoint with `--resume-state`, change `--optim`, `--muon-scope` or `--sched` at fixed membership.
10. Privileged-F variance-weighted teacher (Muse 3). Small code change to the target; `--privileged-mode` and `--privileged-horizon` exist, so check overlap first.
11. EMA side state (Muse 8). Reuse the prepared residual-memory run, add zero-ablation evaluation.
12. Trade sub-clock (Grok 7). Only after the trade-price dispersion check.

## 4. Proposed next experiments (8 to 12)

E1. Cluster A identity audit and oracle ceiling. CPU, no fit. Expected output: identity residual and class-3 ceiling on fold-0.
E2. Covariance-weighted blend vs uniform on fold-0 predictions. CPU, no fit. Coordinate with blend-lab-2.
E3. Top-K threshold sweep on fold-0 predictions (Grok 8). CPU, no fit.
E4. Teacher-pool selection: 4 same-recipe seeds vs 4 disagreement-selected heterogeneous small teachers, distilled into a GRU256x2 student with `--distill`, `--distill-family-weight`, `--distill-cache-variant`. Equal student budget. This is the bold-bet test.
E5. Cointegrating residual and rank-4 innovation basis as an auxiliary input channel (feature builder change). CPU linear screen first; GPU only if the screen clears the noise band.
E6. Low-rank recurrence (`--recurrent-rank`) at GRU384 matched to GRU256x2 CPU minutes on a 2000-row probe, then fold-0. Verify the flag first.
E7. Late optimizer switch from a mid checkpoint (`--resume-state`, `--optim`, `--muon-scope`, `--sched`), fixed membership, paired with a continue-as-is control.
E8. Privileged-F target with per-row variance weight (small code change to target construction). Control: current bootstrap-parent targets, same seeds.
E9. EMA side state (`--memory-arm` or prepared residual-memory run) with zero-ablation of the side state at eval. Reuse existing prepared run.
E10. Tick-unit histogram features (feature builder). CPU screen against raw 112-vector linear readout, same rows.
E11. Quiet-row hold on F (post-hoc first, GPU only if E-screen passes).
E12. Trade-price dispersion census (fraction of rows with more than one distinct trade price in the four slots). CPU, a gate for the trade sub-clock idea.

## 5. Bold bets (three)

Bold 1: teacher pool from disagreement, not seeds (Muse 6 with E4). Rationale: same-recipe seeds give about +0.0004 public, so diversity must come from different operators. Risk: the selection rule may pick noisy teachers; the control is the same-recipe pool.

Bold 2: latent-identity decomposition as a teacher, not only a head (Grok 1 plus Muse 1 plus Muse 3). Rationale: if class-3 saturation is the latent F itself (see E1), a privileged offline F teacher with variance weighting is the only proposal that targets it directly. Risk: E1 may show the heads are already coherent, which closes this bet.

Bold 3: cross-instrument coordinate (Grok 4 plus Grok 5). Rationale: i1 is an input we use only as features; a cointegrating residual is a structural claim that no tried arm makes. Risk: in-sample hedge may be unstable out of sample, and the CI can cover zero from a few dozen sequences.

## 6. Honest uncertainty

- No effect size above is measured. The ideators' numbers are plausible ranges for a 0.001 to 0.02 scale, and their confidence intervals are wide.
- The gap is 0.034 from 0.666 to 0.700. No single proposal credibly closes it. The plan assumes some combination, and a combination must be measured as a combination, not summed.
- The 0.700 public report is not verified. Before setting the target, confirm the metric, exposure and split under which it was reported. If it is not directly comparable to our public score, the gap is unknown.
- Model-family diversity is limited: 2 providers, one prompt each, one lens each. Nemotron produced nothing. Freebuff was not run.
- Passage correspondence was weak. Both models said so for most ideas. Only the seed-pool result (Muse 6, Grok 3) has a passage-grounded argument with a local data point behind it.
- Coordinate with blend-lab-2 (E2) and with the existing residual-memory and privileged-teacher arms before spending GPU time, to avoid repeating an interaction already queued.
