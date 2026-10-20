---
kind: review
status: complete
created_utc: 2026-10-18T22:35:38Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - ../../../specs/plans/2026-10-20-plateau-review-local-kaggle.md
---

# Imported Claude experiment plan

## Source and scope

Imported from user-provided `claude-plan.txt` on October 19, 2026 (Kyiv).
Original byte count: 7083; SHA-256: `b5f6228d56418d6cb75a099dccfd659a8db3f90a20796d9aea374e9dd955d5a0`.
The original text below is preserved verbatim. Import completion is not a claim
that its scientific assertions have been verified.

## Interpretation and next action

This supplied draft proposed a five-day loss/history/i1/sampling/transfer grid. Preserve it as a source, not the active allocation. Subsequent user steering authorizes 15 local GPU-hours plus verified available Kaggle quota and targeted full fits without a 30-run prerequisite. The accepted spec narrows the first experiment to three equally wide cold GRUs. In particular, paired_component_moments returns centered Welford moments; the imported description of six-sum reuse must not be implemented as raw addition.

The [accepted local/Kaggle plan](../../../specs/plans/2026-10-20-plateau-review-local-kaggle.md)
and [campaign frontier](../../campaign_2026-10-20/plateau_review_r1/FRONTIER.md)
control current execution. Preserve contradictions and negative evidence; no
new fit or public-score forecast follows from this imported source.

## Original user-provided text (verbatim)

# Plateau-review experiment cycle (2026-10-19 → ~2026-10-24)

## Context
- `new.txt` (untracked): external RU review of our plateau (public gap to #1 ≈0.004). Thesis: tuning/compression well explored, but representation, state protocol, loss/sample distribution, validation and student information budget stayed fixed. Proposes day1 measurement → d2–3 discriminating screens → d4 deepen → d5 confirm.
- `research.txt` (untracked): RU guideline on autoresearch plateaus (greedy keep-best, operator monotony, score-only feedback, noisy record, memory, proxy fidelity, info value). AIRA/Amazon/JANUS sources partly checked in `research/unified/methods/RESEARCH_LOOP_RESET_20261015_R1.md`.
- Admission stopped 2026-10-18 (`research/campaign_2026-10-18/online_features_r1/runtime/STOP_AUTHORITY_20261018_R1.json`).
- **User 2026-10-19:** resume 2026-10-20 on **local GPU + Kaggle** (no Vast/paid); **screens only — no 30-cell CP30, iterate fast**. Dated override of CLAUDE.md CP30 rule for this cycle; results labelled exploratory (exposed folds F0–F3, 1–2 seeds).

## Existing evidence (from exploration) — reuse, don't redo
- Paired bootstrap + six-sum moments exist: `src/wnn/gmetric.py` (`wp_sums`, `wp_from_sums`), `src/wnn/evaluation.py` (`paired_component_moments`, `paired_component_bootstrap`); position bins in `online_features_r1/kalman_recurrent_groupfast_slow_r1/grade_pair.py`; `cache/seqsums_*.npz`. **No new bootstrap tool needed.**
- Chunk length sweeps (E-083/354/356/H331) and reset sensitivity (E-417) done → drop chunk-length arm.
- Loss: options in `src/wnn/training/cli_targets.py`, dispatch `optimization.py:97-119`; **cached-distill path in `losses.py` hard-codes WMSE (`--loss` no-op)**. Huber < WMSE measured; gcorr variants ≈ noise. Untested: per-target weighting, plain MSE vs WMSE matched, corr loss on clipped preds.
- i1: columns 52–103 (`docs/data_overview.md:100-130`); **main trainer lacks group-zero/drop flag** (`cli_features.py`). No matched i0 vs i0+i1 fit.
- Untested: current-row/EMA/window history ladder; train/infer state-mismatch; loss-density / sequences-vs-repeats at fixed budget; matched cold-start direct vs distilled student; routine all-`need_prediction`-rows score.
- Blend tools: `scripts/crossfold_blend.py`, `grade_prediction_bank.py`. Predictions: `cache/pred_*.npy` + `cache/valid_targets.npz`, folds `research/campaign_2026-09-29/artifacts/splits/fold{0..3}.json`.

## Step 0 — today, no compute
1. Move (untracked → move + add):
   - `new.txt` → `research/unified/reviews/EXTERNAL_PLATEAU_REVIEW_20261019_R1.md` — NOTE_TEMPLATE frontmatter, English summary + "direction → existing evidence → gap" table (from above), RU original verbatim below.
   - `research.txt` → `research/unified/methods/AUTORESEARCH_PLATEAU_GUIDELINE_20261019_R1.md` — same pattern; mark which cited sources verified vs not.
2. README bullets in `research/unified/reviews/README.md`, `research/unified/methods/README.md`; link from `research/RESEARCH_METHODS.md`.
3. Campaign `research/campaign_2026-10-20/plateau_review_r1/`: `README.md`, `CONTRACT.json` (resume authority from user answers, supersedes 10-18 stop for this scope; local 2 CPU/5 GB task RAM/1 GPU; Kaggle; no Vast; screens-only label; scratch `/home/vlad/development/wunder/research_tmp`), `FRONTIER.md`, `runtime/`. Short ADR0042 (fast exploratory screens, CP30 waived this cycle). Update `specs/current-state.md` top; append hypotheses/experiments JSONL rows per arm; link campaign from `research/campaign_2026-10-20/README.md` and unified validation README.
4. Commit.

## Day 1 (10-20) — CPU diagnostics on saved predictions
- D1 **Fork table** with existing scorers: on F0 + F2, same mask — TL3, R3, H512, clean H384 teacher trio, T3 student, alpha0 student, served ZIP preds. Per target t0/t1, 4 position bins, `is_scored` vs all-`need_prediction`-rows score (small adapter over `evaluation.py`). Decide which review fork row applies.
- D2 **Assumption audit** `ASSUMPTION_AUDIT.md`: last ~20 comparisons, column "task assumption tested"; list constants across all.
- Meanwhile implement (python-standards, ruff, `check_training_structure.py`):
  - `--drop-feature-groups` (i0/i1/extras/trade subsets) in `cli_features.py` + feature assembly component.
  - history views: current-row (no recurrence → MLP head), causal multi-τ EMA/deviation/scale feature block, windowed-history GRU (state reset every W rows train+infer) — as focused components, not in `train_rnn.py`.
  - loss: per-target weight flag; ensure non-cached path for loss arms; gradient-path check test.
  - sampling: loss-density flag (supervise every k-th row, full state updates) if not covered by `--thin-prob`.
  - synthetic invariant tests for each.

## Days 2–3 — fast screens (local GPU + Kaggle in parallel)
Matched from-scratch H256/H512×1, short fixed budget, same seed, F0 (Kaggle cohort) + F2 (local), labels-only no-cache unless stated. Card per arm: hypothesis, outcome→next action.
- S1 loss: WMSE (ctl) vs MSE vs per-target-balanced WMSE vs WMSE+clipped corr; inference-only affine/saturation calibration (fit F0, check F2).
- S2 history ladder: current-row / +EMA block / window 32·128·512·2048 / full GRU; + full GRU with EMA block (student-capacity test).
- S3 i1: i0+extras vs i0+i1+extras (ctl) vs +i1 deltas/EMA vs split encoders + interaction block.
- S4 sampling at fixed rows processed: loss density k=1/4/16; sequences×repeats trade.
- S5 transfer: matched cold-start direct vs distilled student (same teacher), plus 2 existing teachers through identical transfer.
- Grade each: paired component bootstrap Δ, per-target/position, fixed blend vs ctl (weight F0 → F2). Independent critic per grade (≤3 active subagents).

## Day 4 — deepen 1–2 supported arms
- Aggregates help student ≥ teacher → 2×2 teacher/student × raw/aggregates + 2–3 student sizes + runtime.
- i1 interaction helps → i1 timescales/interaction form.
- Run `$research-reframe` fresh-context scout with day1–3 facts only.

## Day 5 — confirm + decide
- Best arm on extra fold + extra seed at production size; native serving runtime; ZIP only if locally supported. "Belief revised" note; park arms with scoped negatives.

## Execution rules
- Local via `scripts/limited` (2 CPU, ≤5 GB task RAM, 1 GPU, disk scratch). Kaggle via existing `kaggle_bundle/kernel/sequence` tooling, slug check, hash-bound assets, poller targets, terminal collection.
- Code in `src/wnn/training/` components; `scripts/train_rnn.py` untouched except wiring. `uv run ruff format/check`, `check_training_structure.py`, `make training-tests`.
- Commit per step/arm; README entry for every artifact; preserve negatives.

## Verification
- New flags: synthetic tests (dropped columns zero-influence; window reset matches infer; loss-arm gradient differs from WMSE).
- Each screen: source/config hashes, membership check, zero-OOM receipt, grade reproduces `gmetric` official WP, independent review.
- `scripts/research_tree.py --root . check` after ledger updates.
