# Research tree (living document; started 2026-09-25 21:40 EEST, session until 12:00)

Root: maximize global WP on hidden test (LB 0.6569 now, top 0.6651) AND build durable
understanding (EDA, explanation, pipeline). Rule: evidence and tests decide; no idea is
dropped on priors. Status: [ ] open, [~] running, [x] done (E-id), [-] refuted (E-id).

## A. Understand the data
- A1 Target law
  - [x] class 3 exact: F = round(mean mid next 4 book events) (E-079)
  - [ ] classes 0-2: exact F / loser rule for small ticks (needs tick decoding A2)
  - [ ] t0/t1 asymmetry (t1 easier globally: 0.672 vs 0.658)
- A2 Decode anonymized encodings
  - [x] class-3 code grammar -> exact ticks (E-063)
  - [-] same grammar for classes 1-2 (65-69% exact, E-085 note)
  - [ ] price codes as continuous function of (level distances in bps); learn decoder net from targets
  - [x] volume codes: full-rank mixing (8 dims for 90%/side); ICA gives bid/ask best-queue proxies, imbalance AUC 0.62 far from moves (E-090)
  - [x] a-columns semantics: a6 = real-time since last book change, a5 = since trade, a2/a3 = book intensities (E-091)
  - [ ] trade columns dp/dv semantics (side, size)
- A3 Scoring mask process
  - [x] is_scored explained by a3, i1 activity, a5 clock, book state (val corr 0.34, E-087)
  - [ ] test mask softer than valid? (critic lambda=0.02); design mask-robust training
- A4 Symbolic explanations
  - [~] SISSO engine (src/wnn/symreg.py) validated on synthetic
  - [ ] direction per class; residual of best GRU; level; per-target
- A5 Shift across train/valid/test
  - [ ] adversarial validation within class
  - [ ] recover global time order of sequences (Optiver-RV style) -> pseudo-time CV
  - [ ] instrument identity across sequences (same instrument recurring?) -> instrument embedding

## B. Information / features
- B1 [-] heuristic library: no linear lift (E-088), no gain as GRU inputs (E-089); GBM residual lift only in classes 0/1 (E-092)
- B2 [ ] class-3 Stoikov micro-price chain / next-event classifier + exact 4-step readout
- B3 [ ] i1 cross-instrument (lead-lag in event time)
- B4 [ ] stale-row regime (WP 0.36 after >170 quiet rows, 1/3 of weight)
- B5 [ ] class-3 exact tick-path features as inputs

## C. Models
- C1 GRU recipe: [x] 2 epochs +0.004 (E-083); [x] 3 epochs +0.0006 (saturated); [~] seed 4, h192 e2, lr 2e-3
- C2 diverse members: [-] GBM on heuristics (0.6481, hurts GRU blend, E-093); [ ] class-3 event model
- C3 [~] distill (GRU s3 + s4 + baseline) into one GRU (ab28); [ ] multi-head averaging
- C4 [ ] minGRU / LRU / attention over last-k events
- C5 [~] aux-next: log rows to next target change + its sign (ab25)

## D. Methodology
- D1 [x] decision rule + scripts/decide.py (seqsums cache, softened-mask lambda metric; lambda 0.02 ~ LB)
- D2 [~] mask-softening weights p^2 + eps (ab24); [ ] label uniqueness
- D3 [ ] recency weighting of valid in final fit

## E. Inference engineering
- E1 [ ] fused Rust blend (single call, shared features) to cut the 94 us/row
- E2 [ ] propensity-gated compute

## F. Crazy / long shots (tested, never dismissed)
- F1 mirror symmetry augmentation (learned code involution)
- F2 simulate LOB with a fitted queue-reactive model; pretrain on simulated data
- F3 privileged-teacher label denoising (lookahead teacher)
- F4 treat each sequence as a new instrument: in-context meta-learning on the first rows
- F5 predict the organizer's baseline error (stack on baseline residual)

## Idea inbox
research/ideas/*.md (haiku agents on random books; literature scouts in research/lit/)

## Idea triage (haiku agents, 2026-09-25 22:10) -> test type
- Lift probes (scripts/lift_probe.py, ~1 min each): H1-8 eigenvalue drift of recent deltas;
  H2-1/H3-11 Hurst / variance-ratio regime; H3-1 depth-weighted micro-price; H4-9 class-3
  code-pair association table (offline, train-fitted); H3-3/H3-7 efficiency index; H2-9 toxicity
  proxy (trade flow x spread state); H2-10 innovation factors (ICA on deltas).
- Aux heads (GRU run): H2-4 time-to-next-move; H3-9/H4-12 multi-horizon pyramid (next 1/2/4 events);
  H2-6 magnitude head.
- Loss/weights (GRU run): H3-2 residual-ACF weights; H4-6 variance-constrained; H4-8 margin weights;
  H3-12 adversarial (input noise) training; H3-8 anti-detectability.
- Structure: H4-7/H1-11 regime MoE heads (quiet/busy); H3-10 class-3-specific capacity;
  H2-7/scout reset-state memory-length probe; H2-12 per-sequence instrument embedding.
- Long shots kept in queue: H1-10 free probability i0/i1; H1-5 smile per class; H4-11 regret-bound;
  H1-3 MP-excess leak detector (also an audit tool).

## Active (2026-09-26 02:20)
- A4 residual symreg per class 0/1/2 -> cheap closed-form corrections (symreg_q)
- [-] H-038 online-norm inputs: 0.6702 vs 0.6707 (no cross-domain gain)

## V. Validation methodology (user priority, 2026-09-26 03:20)
Problem: only validation has masks (1873 seqs); paired comparisons carry sd ~0.0015; test is a later
period with a possibly softer mask; validation overstated test by ~0.022 (uniform across models).
- V1 [x] sequences chain into continuous streams (3,288 chains, E-095); decide.py uses 4 chain-grouped folds + chain bootstrap
- V2 [ ] soft-mask metric (|y| * p_hat(scored)) on OOF TRAIN sequences = 6.6x more eval data;
        validate: rank/delta agreement with the true-mask metric on validation across cached models
- V3 [~] chain-grouped K-fold over validation for the train+valid recipe (ab26): OOF predictions of the submitted recipe,
        honest value of adding validation data, unbiased stacking/blend weights
- V4 [x] adversarial AUC 0.99 within every class (E-094): train/valid = different instruments/regimes
- V5 [-] valid is not a later period: chains mix train and valid in both directions (scout); time order only within chains
        the recovered coordinate separates train from valid (valid later?) -> pseudo walk-forward folds
- V6 [x] CSCV/PBO over 20 configs = 0.000 (scout): selection bias not the issue for this config family
- V7 [ ] LB-consistency meta-check: which validation variant (plain, lambda, class-reweighted, soft-mask)
        best orders the 4 known LB points (weak evidence, 4 points)

## Night summary (2026-09-26 04:10)
- Leak re-check: GBM residual gain class 0 was chain leakage; classes 1-2 keep ~+0.002; class 3 overfits (E-096).
- No cold-start penalty: WP by step bin flat (0.66-0.71), context length is not the bottleneck.
- Levers left with evidence: ensembling diverse models (+0.002 LB baseline blend; +0.0014 seeds), distillation to
  keep ensemble gains at single-model speed, recency/extra data (train+valid K-fold), small class-1/2 GBM correction.

## Morning update (2026-09-26 07:45)
- D2 mask weighting: [x] propensity floor eps 0.03 +0.0020 (E-097); [-] eps 0.10 (E-102); [-] class-mass
  weights beyond the floor (E-104); [x] true masks on valid rows +0.0006-0.0010 (E-098); [-] label-aware
  selection model (+0.002 AUC only, E-103).
- C5 aux heads: [x] next-change head +0.0014 (E-097), stacks with the floor (+0.0030 total, E-102);
  [-] 2nd/4th-change horizons (E-107).
- C3 distillation: [x] 3-teacher student +0.0013 alone; [-] adds little on top of floor+aux (E-102).
- A4 explanations: [x] 4-term symbolic direction models per class (E-100/E-101), visual partial dependence
  (E-106): the GRU already tracks every main effect; calibration exhausted (E-109).
- Tooling: C++ symreg kernel 13x (E-105); vectorized decision sums 15x (E-108); N-member native packager.
- Next: final fits ab33 (train+valid, floor + aux + true valid masks, 2 seeds) -> candidates:
  single, 2-seed blend, GRU + baseline; Opus researchers (arch / micro / shift) pending.
