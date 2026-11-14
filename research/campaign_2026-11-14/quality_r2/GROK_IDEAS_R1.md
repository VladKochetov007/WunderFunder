The residual correlation near .995 does **not** justify dropping every teacher average. It justifies only this: an equal-weight blend of exchangeable, same-architecture fits will not move a |clip(y, 2)|-weighted metric by variance reduction. That matches the measured two-student blend tying the teacher at .69142.

The leap fails for four mechanistic reasons. Unweighted residual Pearson gives quiet rows the same vote as large moves, and the memory failures already sit on quiet rows, so .995 can be manufactured where the metric is almost indifferent. Correlation ignores scale: identical direction with different residual size still has a better weighted blend than either model. Train residual correlation between two fits on the same rows is inflated by shared idiosyncrasies. The heterogeneous ensemble’s +0.0009 over .69142 is inside the .6867/.6882 seed gap, so it neither proves diversity nor kills a one-parameter blend. KD does not change the serving choice: α=0.5 replicates below the teacher, α=0.9 ties it on one seed.

### 1. Representation — hold state on quiet rows, show gap age
GRU depth is being spent iterating the recurrent map on near-null inputs. If the map’s spectral radius is not 1, the pre-gap state is washed out exactly before the next high-|y| row, which is where |clip(y, 2)| puts the loss. A loss since-weight does not stop that iteration; since8 already lost.

**Falsifier (no refit):** replay the current GRU and copy h through rows that fail a pre-registered quiet rule (for example L1 below the train median of quiet rows). Score pooled clipped weighted correlation on the first 80% and the last 20% of train by time.

**Observable:** the skip raises the late slice and concentrates the change on the first few active rows after long gaps; dense stretches do not give the gain back. If the late slice falls, the quiet steps were doing useful work and the card stops.

**Cost:** one forward pass. Only if that passes, one 4-epoch refit with the hold plus a single causal input, log1p(steps since the last non-quiet row).

**Serving:** one causal scan, fewer GEMMs when the hold fires, state copy O(256), same 20MB class.

### 2. Objective — regress the clipped target
The weight saturates at 2, but weighted MSE to raw y still lets |y|≫2 set the gradient through a huge residual. The metric never sees that residual. Class scaling is a different object: a constant output multiplier, not a change of target. A horizon auxiliary is the wrong first test; no-aux already lost, and a second change would be unreadable.

**Falsifier:** one seed, same GRU256×2, 4 epochs, same weights, target replaced by clip(y, 2). Select only on the last 20% of train. Promote only if that lift exceeds the known seed gap, about 0.0015.

**Observable:** gradient mass shifts off the extreme tail; late-slice metric rises only if those tail residuals were pulling predictions outside the clipped geometry. A flat late slice with a better full-train MSE means in-sample level fitting, and the card stops.

**Cost:** one current training run.

**Serving:** unchanged graph. The clip is training-only.

### 3. Zero-fit diagnostic — score diversity in the metric’s inner product
Use saved teacher outputs only.

On a cut of train, report unweighted residual correlation, the same correlation with weights |clip(y, 2)|, and that weighted correlation on the top decile of |y| and on post-gap rows only. Then fit one non-negative blend weight on the early slice and score equal-weight versus that blend on the late slice. Also fit a two-parameter weighted affine map from prediction to clip(y, 2) per target on the early slice and score it on the late slice.

**Observable:** if weighted top-decile residual correlation stays above about .99 and the late-slice blend lift is inside the seed gap, duplicate GRUs are done. If the weighted or post-gap correlation drops while the unweighted figure stays ~.995, the old dismissal used the wrong product and a frozen blend remains eligible. An affine lift means part of the MSE-versus-correlation gap is calibration.

**Cost:** none. **Serving:** a frozen affine is two scales and two shifts, causal and negligible; blend weights are scalars.

### 4. Submission composition
Pre-declare the late-train 20% rule and the 0.0015 bar before any new fit. Valid stays burned.

Default artifact, if cards 1–3 fail: one GRU256×2, 4 epochs, orthogonalized recurrent momentum, AdamW on the other weights, current 211 inputs, current auxiliary, current weighted MSE. KD students are training traces, not the shipped model. Two seeds of this model do not get blended.

Replace a piece only when its own falsifier clears the bar: quiet-hold plus log1p(age) from card 1, clipped target from card 2. Add a second model only if card 3 shows a late-slice blend lift above the bar, and only a heterogeneous partner. Shared-core and same-seed blends are closed. Fit the blend weight on the early slice and freeze it.

Final step: refit the frozen recipe on train+valid once. Do not retune. Two GRUs are about 6–8MB; keep the second only after one timed pass shows two passes fit in 60 minutes on one CPU.
