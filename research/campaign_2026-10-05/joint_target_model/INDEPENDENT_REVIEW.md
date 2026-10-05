# Independent review: joint target package

**Disposition:** the fit/evaluation boundary and component provenance are sound
for the package's documented historical fold-0 falsifier. Do not launch the
frozen-state fit until the clipped-label mismatch below is corrected. The
package is not an evaluator-ready result, and it does not evaluate the separate
October 5 24-fit/16-test cohort.

This was a read-only review. No training, evaluation, or GPU job was launched.
The only data check opened the existing fit-target memmap to count rows outside
the metric clip.

## Findings

### High: frozen-state CE labels use raw targets while the atoms and metric use clipped targets

In `fit.py`, the codebook is fitted from targets clipped to `[-2, 2]` by
`fit_codebooks`, and each batch's weighted MSE target is clipped. However,
`joint_labels`, `shuffled_labels`, and both `marginal_labels` are assigned from
the raw fit targets (`fit.py:151-157`). The E2E path correctly clips its target
before assigning labels (`e2e_fit.py:171-185`), so the two trainers do not use
the same quantized-label objective.

The existing fold-0 fit export has 2,719,410 rows; **172,588 rows (6.35%) have
at least one target outside `[-2, 2]`**. Thus the mismatch affects a material
part of this exact training set. The raw value can change which boundary
centroid is nearest, even though the regression target and reported metric use
the clipped value. Assign all four label arrays from the clipped targets, as
the E2E path already does. Then keep the smoke and arm anchors unchanged.

### Medium: this package is a different cohort and parent than the October 5 benchmark

The frozen-state trainer consumes the September 29 fold-0 paired export. Its
metadata lists 1,405 fitting validation groups and 2,719,410 selected rows. It
does not load `target_mechanics/cohort.json`'s 24 fit and 16 test groups, and
its two source members are the historical fold-0 GRUs. The README accurately
limits its claims to that falsifier and says it is not the current strongest
Sol model. Keep any outcome scoped that way; it cannot serve as the October 5
cohort comparison or as evidence about the current parent.

### Medium: no evaluator or deployment replay is included

The frozen-state `fit.py` saves head state dictionaries and a fit-only receipt.
The E2E trainer saves continuation checkpoints, but there is no package scorer
that restores them, runs the official callback over evaluation groups, and
reports `global_wp_clip2_v1` with component uncertainty. The README explicitly
requires a separately registered scorer and independent review, which is the
right boundary. Treat this package as fit-only until that evaluator exists and
validates checkpoint/export identities. Before callback integration, tighten
the E2E parent's `n_out < 2` check (`e2e_fit.py:69-72`) to require exactly two
outputs; the wrapper deliberately truncates any wider head at
`e2e_model.py:27`.

### Low: frozen-state fit receipt trusts source checkpoint hashes from the export sidecar

`verify_fit_prefix` hashes every exported array and validates the frozen
manifest and fit groups (`fit.py:34-73`), then the receipt copies the source
checkpoint hashes from the JSON metadata. It does not resolve those member
names and compare the recorded hashes with current checkpoint bytes. The E2E
path does this for its selected parent (`e2e_fit.py:42-47`). The frozen-state
fit is reproducible from the exported arrays regardless, but an evaluator
should verify that its parent checkpoint is the member whose state/prediction
arrays were exported.

## Checks that passed by inspection

- The pair exporter selects `is_scored` rows after the fixed 99-row warm-up;
  the repository's validation contract defines `need_prediction` as exactly
  `step_in_seq >= 99`. The fit sidecar names that same mask. The E2E path uses
  `is_scored` and clears the first 99 rows (`e2e_fit.py:168-171`).
- The frozen-state path checks the fold manifest hash, exact fitting-group
  list, array hashes, selected-row mask, and group membership before training.
  The split loader checks fit/evaluation component separation. The E2E path
  additionally verifies the selected parent checkpoint SHA, fixed epoch, and
  no-evaluation-during-fit metadata.
- No evaluation export or evaluation group is opened by either training path.
  E2E codebook fitting and continuation use only `split.valid_groups`; hidden
  state resets at each sequence and is detached between TBPTT chunks.
- All arms instantiate the same head and the two categorical output widths are
  both 32. The candidate, ordinary-readout, shuffled-pairing, independent
  marginal, and regression-anchor arms share the same trunk, initialization,
  optimizer, and fixed pass. The zero residual and zero metric gate give every
  arm the exact clipped-parent anchor. The regression control retains the same
  quantized CE auxiliary loss, so it controls the readout mechanism while
  matching the auxiliary training signal.
- `arm_loss` adds nearest-label CE to clipped absolute-target-weighted MSE.
  The MSE is a training auxiliary objective, not `global_wp_clip2_v1` itself;
  the README says so and does not present fit loss as a score. The tilted
  decoder uses a separate `|c|` denominator per target and falls back to the
  parent if that denominator is zero.
- Nearest-label assignment is blockwise. For 2,719,410 rows, the three retained
  label arrays (joint, shuffled joint, and two marginals) use about 87 MB total;
  each joint distance block is about 4.2 MB. The float16 paired-state map is
  about 1.30 GiB and remains memory-mapped, while the default 8,192-row batch
  copies about 8.4 MB of states to float32. This is plausibly within a 2 GiB
  CPU cap, though hashing the full state file causes a complete sequential read
  and the actual cgroup peak should be recorded. E2E uses one 20k-row sequence
  and 1,024-row chunks, so its activation footprint is bounded.

## Recommendation

After clipping the frozen-state label inputs, a one-epoch fit-only falsifier is
reasonable under the declared cap. Its fit diagnostics can compare arms on
training data only. Do not promote an arm or claim transfer from those
diagnostics. The next evidence step needs an independently reviewed evaluator
on an explicitly registered component-held-out cohort and the intended parent;
the October 5 24/16 cohort and current parent require their own state export and
provenance.
