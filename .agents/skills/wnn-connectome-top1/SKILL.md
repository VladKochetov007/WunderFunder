---
name: wnn-connectome-top1
description: Continue root Wunder Connectome Challenge 2 research toward a deterministic submission with honestly expected public score above the current public best. Use when auditing, experimenting, validating, packaging, or planning this project's root Challenge 2 work. Exclude wunder/challenge-3. Do not claim a public target from proxy-only evidence.
---

# WNN Connectome Top-1 Campaign

`objective_mode`: **competition**. Optimize expected unseen public score under serving and rule constraints. Novelty is optional. A useful incremental ZIP need not prove a causal mechanism. Read `.agents/skills/conduct-creative-research/references/empirical-validation.md` before interpreting scores or defining promotion gates.

## Current contract

| Field | Value |
|---|---|
| Metric | `global_wp_clip2_v1` in `src/wnn/gmetric.py` (clip ±2, weight `|clip(y)|`, pool `need_prediction AND is_scored`, mean of t0 and t1). This is the official metric. |
| Public best | **0.6660**, seed-4 equal parent/head combo i16, portal `2e5j6bnt`, ~37m07s / 626 MB, `submissions/c20261004_combo_s4_i16/c20261004_combo_s4_i16.zip` |
| User milestone | honest expected public **0.7000** (unmet) |
| Local supplied-valid blend | 0.700771 on the same validation file. **Not** a public forecast. Do not add local deltas to 0.6660. |
| Verification tier | **C** for public transfer. Local valid, holdout, and tiny-model protocol tests are discovery. |
| Upload | human gate |
| Exclude | `wunder/challenge-3`. Nested `wunder/challenge-2` is a different format. |

Historical public 0.6622 (`final_since2_gclp`) and the “metric unresolved / report both” instruction are obsolete. Dual-reporting pooled vs per-sequence WP is optional diagnostics, not a freeze.

## Exposure (do not relabel)

Supplied validation, the 368-sequence holdout (`sha256(str(seq_ix)) % 5 == 0`), folds 2–3, and leaderboard-calibration artifacts are **exposed**. Repartitioning, a new agent, or a `wave1b` split bit does not restore an unexposed holdout. A chain bootstrap of frozen predictions conditions on the fitted models; it is not training-seed or public-transfer evidence. D256 beat seed-4 locally and lost publicly (0.6641). EMA6 public rows regressed.

If every local slice has been inspected, say so. A competition upload may still be an operational choice; it does not upgrade a scientific claim.

## Status vocabulary

Use `not_tested`, `screen_failed`, `insufficient_evidence`, `supported`, `replicated`, `contradicted` as in the creative-research artifact reference. A missed spending gate is `screen_failed` for that recipe, not a universal refutation of the family. A tiny-model protocol win is not `supported` for the champion GRU256.

## What has been screened (claim_scope = this recipe, this champion)

If `research/campaign_2026-10-06/` exists, treat `INDEX.md` as the index. Closed on the frozen seed-4 blend unless a new test uses a different comparison:

- Linear Hawkes / depletion / i1 / full volume residual: `screen_failed` (hurts or ~0).
- Lattice or class-quantum snap of this blend: `screen_failed`.
- Output-space holds and EMAs (touch, age-1, last-trade): `screen_failed` (full-file about −0.08 to −0.01).
- Linear episode collapse on far no-trade rows: `screen_failed` (−0.017).
- Tiny GRU episode **only on the hole**: protocol bar cleared (+0.011 / +0.017, two seeds) at absolute WP ~0.22 vs champion cell ~0.42. `insufficient_evidence` for a GRU256 or public ZIP.
- Tiny GRU episode on all scored rows, or mixed row+episode: `screen_failed` (−0.047 / −0.023 on the hole).
- Activation-sweep / E488 readout transfer as a path to 0.70: `screen_failed` as a public forecast. Tiny-panel gains are the wrong functional.
- Original q8 ZIPs: packaging `contradicted` (libpython3.13). Repaired q8: compatibility `supported`; public timing `not_tested`; not expected better than 0.6660.

The remaining miss is far quiet rows (age since joint i0 touch ≥ 32 and next label change > 8): ~30% of t0 weight, WP ~0.48, mostly class 3. Near rows (next change ≤ 8) at the same age are ~0.86.

## What to try next

Cheap falsifier first. Matched control. Predeclare the cell and kill bar. Do not spend a GRU256 hour on another feature stack or output hold.

A champion-scale recipe change (loss, bootstrap reanchor, since-weight) needs a **paired** tiny or one-fold control, then confirmation. Tiny hole-only episode loss did not survive as a mixed term; do not promote it from that screen.

## Resources

Use `scripts/limited`. Historical campaign ceiling: 12 CPU, ~36–40 GB RAM, one GPU worker, 5 GB observed VRAM, 80 device-GPU-hours. Obey the user's tighter cap when they set one (this session used 5 cores / 10 GB locally). Recheck `nvidia-smi` and Kaggle quota before GPU/TPU work. Long runs through the durable job path.

## Packaging (only after an expectation gate)

Fixed members and weights. No wall-clock governor. ZIP ≤20 MB, `solution.py` at root, Python 3.11 offline, one vCPU / 16 GB / 60 min. Reject `DT_NEEDED` libpython (`scripts/check_submission_abi.py`). Bind import and streaming checks to the delivered ZIP hash. See `research/campaign_2026-10-04/E462_PACKAGING_INCIDENT_20261004.md` if present.

## Session start

1. Read this skill, git status, and `research/INDEX.md` or `research/campaign_2026-10-06/INDEX.md` if present.
2. Do not discard unrelated dirty-tree changes.
3. Confirm metric `global_wp_clip2_v1` and public best 0.6660.
4. Check GPU/RAM and the user's current cap.
5. Record exposure and the comparison **before** looking at a new score.
6. Run the cheapest falsifier. Preserve negatives.

Never cite this skill as primary evidence. Open the named artifact. Do not upload unless a human asks.
