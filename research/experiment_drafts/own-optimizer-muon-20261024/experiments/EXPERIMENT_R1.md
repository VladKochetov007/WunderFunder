---
kind: experiment-record
status: complete
created_utc: 2026-10-24T21:30:00Z
experiment_id: E-OPT-SYNTH-SCREEN-20261024-R1
new_competition_fits: 0
synthetic_fits: 12
---

# Synthetic optimizer screen R1 — AdamW vs Muon variants on a tiny GRU

## Purpose

First bounded probe of the "own optimizer" question
([proposal C2/C3/C5](../RESEARCH_PROPOSAL_R1.md)): does matrix-aware
orthogonalization (Muon), a DeepSeek-style hybrid scope (recurrence matrices
only), or a two-stage Newton-Schulz schedule change convergence on a task that
mimics our competition's structure — per-batch-normalized weighted MSE, targets
clipped to ±2, magnitude weights, metric-style per-channel weighted Pearson?
**This is an exploratory synthetic mechanism probe, NOT competition-quality
evidence**; it only justifies (or not) follow-ups on real cached data / Kaggle
GPU.

## Design

- Task (fixed data seed 1234, identical for all arms): 48 sequences × 1500
  rows; 32 inputs = 24 smooth AR(1) fields in 4 groups + 4 decaying event-tick
  channels + row clock + 3 slow interactions; 4 target channels = bounded
  nonlinear lagged functions of group means and ticks + noise, clipped ±2;
  weights = |clip(y)|; causal split first 70% train / last 30% val.
- Model: 2-layer GRU hidden 64 → linear head (4 outputs); init seeds 0/1/2.
- Loss: weighted MSE with per-batch weight normalization (batch weights sum to
  1) — mirrors the trainer's chunked per-batch regime.
- Training: chunked BPTT (192-row chunks, state carried within sequences),
  batch = 8 sequence-chunks round-robin, 800 steps, linear 50-step warmup then
  cosine to 0.1×, base LR 3e-3 for ALL arms (same budget philosophy; LR not
  per-arm tuned — a validity caveat, not an accident), weight decay 1e-2
  everywhere, strict FP32, single thread.
- Arms: `adamw` (betas 0.9/0.95); `muon` (all 2D matrices: weight_ih/hh both
  layers + head; momentum 0.95, Nesterov view, Newton-Schulz 5× quintic
  (3.4445, −4.7750, 2.0315), Kimi/Moonlight RMS scale 0.2·sqrt(max(m,n));
  AdamW on biases); `muon_hybrid` (Muon ONLY on weight_hh_l0/l1); 
  `muon_ns2stage` (= `muon` but 3 aggressive + 2 stabilizing NS iterations,
  stabilizing polynomial p(s)=1.5s−0.5s³ with p(1)=1, p'(1)=0 — my design
  choice).
- Metrics: final val weighted-Pearson (clip preds ±2, weights |clip(y)|,
  per-channel correlation then mean), val weighted MSE, effective rank
  (exp(entropy of singular values)/min(m,n)) of final `weight_hh_l0` and of the
  last update direction on that parameter (orthogonalized update for Muon arms;
  m/√v direction for AdamW).

## Execution and limits (strict control after two laptop freezes)

Two freeze/reboots interrupted this session; the first grid launch died with
only `adamw_s0` completed (`adamw_s1` partial log preserved). Per the user's
instruction all processes were then run under strict gates
([strict_fit.sh](strict_fit.sh), [operations note](../../../unified/operations/SESSION_FREEZES_STRICT_PROCESS_CONTROL_20261024_R1.md)):
one synchronous fit at a time through `scripts/limited` (cgroup: CPUQuota=100%,
MemoryMax=700 MB, MemorySwapMax=0, RuntimeMaxSec=330), pre-run gates
(thermal <84 °C, MemAvailable ≥8 GB), 60 s cool-downs with logged post-run
temperatures. Observed per fit: wall 230–261 s, peak RSS 573–681 MB
(`/usr/bin/time -v`), gate temps 53–71 °C pre-run, post-run spikes to
84.05 °C twice that cool-downs cleared; no OOM, no swap, no gate aborts.
TMPDIR on physical disk (`/home/vlad/development/wunder/research_tmp`).
Grid total wall (fits only, excluding interrupted first attempt): ~50 min.
The 12 completed fits are matched (same steps/data); the interrupted
`adamw_s0` from the first attempt is byte-valid (exit 0 before the freeze) and
kept.

## Results (per-seed final val weighted-Pearson)

| Arm | seed 0 | seed 1 | seed 2 | mean ± SD | paired vs adamw (by seed) |
|---|---|---|---|---|---|
| adamw | 0.8561 | 0.8562 | 0.8547 | 0.8557 ± 0.0007 | — |
| muon | 0.8502 | 0.8496 | 0.8525 | 0.8508 ± 0.0013 | −0.0060, −0.0066, −0.0022 (mean −0.0049) |
| muon_hybrid | 0.8559 | 0.8557 | 0.8552 | 0.8556 ± 0.0003 | −0.0002, −0.0005, +0.0005 (mean −0.0001) |
| muon_ns2stage | 0.8527 | 0.8516 | 0.8532 | 0.8525 ± 0.0007 | −0.0034, −0.0046, −0.0015 (mean −0.0032) |

Val weighted MSE follows the same ordering (0.08754 / 0.09044 / 0.08767 /
0.08947). Effective rank of final `weight_hh_l0`: adamw 0.906; all Muon arms
0.946–0.953. Last-update effective rank on that parameter: adamw ≈ 0.71; Muon
arms 0.985–0.994 (near-maximal by construction). Full numbers incl. momentum
eranks and curves: [results/AGGREGATE_R1.json](results/AGGREGATE_R1.json),
per-run JSONs alongside, logs in [results/logs/](results/logs/).

## Interpretation (bounded by the caveats below)

1. **Full-muon (all matrices) loses to AdamW at this scale (−0.005 paired,
   same sign in all three seeds).** This is consistent with the repo's only
   matched competition-data screen (D02: Muon-WMSE 0.6007 vs AdamW 0.6060) and
   tempers the "Muon everywhere" story for small GRUs on this task family.
2. **The DeepSeek-style hybrid (Muon on recurrence matrices only) is at parity
   with AdamW (−0.0001, within seed noise) while still raising the recurrent
   matrix's effective rank (0.951 vs 0.906).** At this scale the geometry
   benefit is visible in the weights but does not convert into an endpoint
   win; it does not cost one either. Scope (which matrices get Muon) matters
   more than the NS schedule — the sharpest single observation of this screen.
3. **Two-stage NS coefficients recover about a third of the full-muon loss**
   (−0.0032 vs −0.0049) but do not reach parity; the stabilizing-polynomial
   choice here is my design, not DeepSeek's exact recipe.
4. **The effective-rank mechanism is real and measurable**: Muon-trained
   `weight_hh` consistently shows higher effective rank (0.947–0.953 vs
   0.906) and near-flat update spectra. If the rank story matters for our
   20k-row sequences, it must be tested where long-context stress exists —
   this synthetic task's 1500 rows barely stress recurrence.

## Threats to validity

Synthetic task ≠ competition data (simpler, no mask/propensity structure, 4
channels vs ~141 components); tiny scale (GRU-64×2, 800 steps); 3 seeds only;
one shared LR for all arms (Muon's optimal LR at this scale may differ — the
paired comparison answers "at equal LR budget", not "at each arm's best");
per-batch normalization interacts with all arms identically here but the real
trainer's 0.1–9.7× batch-mass spread is not reproduced; the two-stage
polynomial is unvalidated. None of these numbers are competition WP or a
public forecast.

## What this justifies, and falsifiers

- **C5 executed after this screen (zero-fit, real data):** the matched D02
  quartet checkpoints (verified sha256 against
  `ROOT_QUARTET_BINDINGS_R2.json`) show the Muon rank mechanism IS present on
  competition weights — `gru.weight_hh_l0` stable rank 0.0685 (AdamW) → 0.4095
  (Muon), `weight_ih_l0` 0.1417 → 0.4076, entropy-erank +0.009–0.015 — **while
  the Muon arm lost that same matched screen by −0.0054 WP.** Higher spectral
  spread of weights/updates does not convert into endpoint quality at this
  scale; the Parallax-style "optimizer unlocks architecture via rank" inference
  does not transfer to our 220-update small-screen regime. Raw:
  [results/C5_RANK_D02_R1.json](results/C5_RANK_D02_R1.json), script
  [diagnostics_c5_rank_d02.py](diagnostics_c5_rank_d02.py).
- Justified next (zero/low cost): the FP32 Newton-Schulz compliance patch
  (policy, independent of quality); C4 batch-mass screen remains the best
  local fit-budget candidate; any Muon follow-up should test the hh-only
  hybrid scope at larger scale, not full-muon.
- If a future real-data hybrid-scope arm (hh-only Muon) also shows parity with
  rank gains at larger scale, a single Kaggle confirmation cell (C2 panel
  subset) becomes worth its GPU-seconds; if it shows the full-muon loss too,
  the "own optimizer" direction closes for this competition (wholesale
  falsifier in the proposal §4).

## Artifacts

[Script](synth_optimizer_screen.py) · [strict runner](strict_fit.sh) ·
[original grid runner (interrupted)](run_grid.sh) ·
[AGGREGATE_R1.json](results/AGGREGATE_R1.json) · per-run JSONs and logs under
[results/](results/). Environment: repo `.venv`, torch 2.14.0+cu130 (CPU),
numpy 2.2.6, Fedora Linux, single core per cgroup.
