# DTW / piecewise-constant idea round — summary (2026-10-02, controller)

Four CPU scouts, each on one part of the project (brief: `BRIEF.md`). Panel: pooled 4-fold OOF of the distilled
GRU256 (alpha 0.5) family; its public-like-mix level 0.6641 equals that model's public score. Agents could not write
.md files (harness rule), so per-agent `REPORT.md` files below were saved by the controller from their final messages;
raw JSON/logs and `scripts/cr_dtw_<agent>_*.py` are theirs.

## Answer to "the target is piecewise constant, the predictions are not — what to do?"
- Segments are causally observable: every target change falls on an a6-endpoint row and every such row is a change
  (6,500,185 / 6,500,185).
- In hindsight, within-segment variation of p only adds variance (oracle: replace p by its segment mean → +0.025
  pooled, +0.078 class 3). That oracle is NOT noise removal: inside a segment the model keeps learning about the next
  quote move (WP 0.02–0.34 at ≥31 rows before the move in class 3, ~0.75 just before it; broadcasting the last
  in-segment prediction to the whole segment = 0.828).
- So every causal "make it piecewise" readout loses on all 4 folds: hold −0.146, running mean −0.056, jump-only
  −0.090, reset-EMA(0.5) −0.0018, cross-fitted stackers −0.001…−0.004. The model does not lag (perfect post-event
  response ≤ +0.0013; 1-row delay −0.010); parent and head-FT are not time-shifted (DTW offset 0.015 rows).
- Any training target averaged/aligned up to the segment end is biased (the segment end is a future event, not a
  stopping time): whole-run teacher mean −0.042, run-end TD −0.0071 (explains E-174), DTW-aligned teacher −0.0038.
  The unbiased construction is the **stopped bootstrap** (teacher at row min(t+h, segment end), else y; optional
  stopping): run-coherent noise 9x lower than alpha 0.7 at bias +0.0001 (h=8). H-059 (alpha by rows-until-change)
  is biased (+5–7% class-3 scale); its GPU screen tied alpha 0.7 (E-331).

## DTW applications tested (all cross-fitted, none survives)
| agent | idea | result |
|---|---|---|
| readout | DTW alignment prediction-path vs target-path (lag) | no lag; warps saturate at band edges, sign flips by class |
| labels | DTW-kNN over class-3 event-clock paths (L16, band 2, K64) → residual | +0.0000 (cross-fit picks weight 0) |
| labels | DBA/k-medoid prototypes (M=32) of pre-segment paths | +0.0000 |
| streams | DTW-aligned i1→i0 lead-lag (open-end DTW, pending i1 move) | ridge −0.0001 pooled (class 2 +0.0006); HGB −0.0024 |
| streams | DTW-kNN retrieval of per-sequence scale from warm-up | oracle +0.0061, retrieval +0.00001 |
| streams | class-3 quiet runs as time-warped copies | real (DTW/Euclid 0.16 vs 0.49 null) but target-neutral (corr 0.002) |
| objective | soft-DTW / DILATE-style losses | rejected analytically: boundaries exact, no lag → warp tolerance only blurs labels |

## Ranked GPU recommendations (not run tonight)
1. Stopped-bootstrap distillation target, alpha 0.7, h=8, fold 0 (objective agent; `objective/stopped_bootstrap.diff`):
   predicted +0.001 vs alpha 0.7 (+0.0005…+0.003); falsified below +0.0005.
2. Optional-stopping lambda-return target (readout agent), only if more teacher weight pays (alpha 0.85 did not, E-329).
3. Six causal i0/i1 path inputs incl. dtw_pend (streams agent), forecast ≤ +0.0002 — not recommended.
