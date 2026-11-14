# Fresh quality scout: four cards

**Target:** best causal streaming regressor under 20 MB and 60 minutes on one CPU. Anchor: GRU-256×2, Muon, 4 epochs. Label/teacher α=.5 scored .69047/.69048 across seeds; α=.9 scored .69144 once; the fixed student pair scored .69142. The fbc1k teacher scored .69142. The heterogeneous teacher blend scored .69234 train-only, .66355 public; the older all-data TL3 scored .66832 public. Six epochs, H384, born-again, no auxiliary loss, since=8, column masking, TabM-GRU, NorMuon, smoothing/calibration, and hand residual GBM failed this recipe. Full valid is exposed. These cards are proposals only; no fit or export was launched.

## 1. Time — distill adjacent-row changes

**Hypothesis:** The α=.9 teacher helps partly through its trajectory, not only its pointwise level. Add a small training loss matching `Δstudent` to `Δfbc1k`, alongside the existing label/teacher objective. This is training-time temporal transfer, distinct from since weighting or post-hoc smoothing.

- **Zero-fit gate:** On cached teacher-held-out valid predictions, test whether teacher increments contain label-aligned residual signal beyond the student, by target and source group. If not, stop.
- **Implementation/falsifier:** Add one boundary-reset adjacent-row loss term in [`training/losses.py`](../../../src/wnn/training/losses.py); freeze its coefficient, keep α=.9 and the rest of the recipe fixed. One q14-sized fit (41-minute planning basis; cap 60). Serving stays the same GRU, so no added bytes or inference work. Reject if a paired, distinct-seed comparison loses the official endpoint.
- **Dedup:** Not `since8`, smoothing, H-366's cross-target difference channel, or run-level KD. The operator here is difference across time.

## 2. Factor — align training with the scored targets

**Hypothesis:** Weighted MSE plus KD may spend gradient on scale or pooled error that the official metric does not reward. Add a small supervised weighted-Pearson term, computed separately for each target with the official clipping, mask, and label-magnitude weights; retain WMSE as the scale anchor.

- **Zero-fit gate:** Replay the frozen α=.9 checkpoint and compare, by target and source group, gradients of the official differentiable score surrogate with current WMSE gradients. Close alignment means no fit.
- **Implementation/falsifier:** One change in [`training/losses.py`](../../../src/wnn/training/losses.py), pooled within each training batch and equally averaged across targets; freeze one coefficient before scoring. One q14-sized fit, ≤60 minutes; same inference graph and size. Reject if the fixed official mean falls or a target trade-off erases the gain.
- **Dedup:** Direct supervised metric alignment, distinct from H-369's Pearson KD and the earlier per-target WMSE-mass check. Target definition: [`gmetric.py`](../../../src/wnn/gmetric.py).

## 3. Instrument — audit the blend's sequence-level support

**Hypothesis:** The heterogeneous blend's +.00092 over the student pair may be broad or driven by a few exposed groups; that determines whether its added serving cost is justified.

- **Zero-fit test:** On frozen predictions, paired-bootstrap whole sequences within source-qualified groups; recompute the complete official score for the blend, student pair, and α=.9 single. Report target/group influence and how often the blend leads. No row bootstrap or coefficient tuning. If the edge is concentrated or unstable, prefer the simplest endpoint.
- **Implementation/limit:** No fit or model change. Check exact native package size and one-CPU runtime; use the blend only if it clears 20 MB/60 minutes, else choose the strongest qualifying single. Valid is exposed, so bootstrap spread is panel sensitivity, not fresh-CV uncertainty; public .66355 is counterevidence, not a local-to-public offset. Existing [`teacher_resid_diag.py`](../local_r1/teacher_resid_diag.py) samples 400 train rows; this is full-endpoint cluster sensitivity.

## 4. Transfer — cap loss reweighting across source groups

**Hypothesis:** The local/public gap may reflect over-reliance on train-dominant regimes. If α=.9 errors vary materially across fixed source groups, capped group-DRO weighting could improve transfer without changing the model. The analogy is worst-mixture training over observed groups; it does not establish these groups as causal domains ([Sagawa et al., ICLR 2020](https://arxiv.org/abs/1911.08731)).

- **Zero-fit gate:** Compute per-group official correlation, residual contribution, size, and target coverage from the frozen α=.9 predictions. Use existing source manifests only; if there are no adequately sized, meaningfully different groups, stop.
- **Implementation/falsifier:** Apply fixed, capped exponential weights to train-only group WMSE/KD loss in [`training/losses.py`](../../../src/wnn/training/losses.py); keep the 4-epoch recipe and GRU-256×2. One q14-sized fit, ≤60 minutes; unchanged serving size/runtime. Compare against the matched ordinary-loss control and report every source fold. Reject if mean score falls without broad group improvement.
- **Dedup:** Loss reweighting, not H-225 instrument-wise pooling, input masking, or per-sequence normalization. No numerical prior.

## Priority

First do the zero-fit blend/package audit; it can change endpoint choice without training. Then inspect the source-group residual table. If heterogeneous, test capped group weighting; otherwise prioritize the increment-KD screen. Try the correlation objective only if its gradient probe shows material conflict. The 41-minute fit figure is a planning basis, not a runtime guarantee. Indexing is omitted to honor the requested one-file write scope.
