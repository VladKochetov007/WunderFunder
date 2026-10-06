# Distillation implementation critic checklist

Frozen from `CONTRACT.md` and the delegated continuation spec before opening
the trainer implementation. Treat these as admission gates, not optional
smoke tests. Use tiny deterministic arrays/models; no fit or accelerator is
needed.

## Numeric and structural cases

1. **C0 off parity.** Load identical 23-output parent weights and run the legacy
   path and the new path with distillation disabled on a fixed batch. Require
   bitwise-equal 23 outputs, loss, gradients, and one optimizer update. The
   family head and both caches must be absent/unused.
2. **Composite main target, no implicit averaging.** Give four named sources
   two-column arrays with distinct row/column values. Assert C2 materializes
   exactly one target as
   `0.30*A + 0.20*B + 0.25*C + 0.0625*D`, coefficient sum `0.8125`, with no
   normalization and no second divide by source count in the legacy loader.
   Shuffle input enumeration while retaining names; the result must not change.
3. **C1 and C4 controls.** C1 target must equal `0.8125 * strongPAIR`. C4 main
   weights must be C2 weights divided by `0.8125`, and its target must equal
   `C2_target / 0.8125`. C3/C4 retain the same `0.1` family-loss multiplier.
4. **Shapes and copied outputs.** Seed parent outputs with distinct sentinels
   across all 23 rows (`main[0:2]`, existing auxiliary `[2:23]`). After warm
   initialization, every one of those rows and its predictions must match
   exactly. The new family head is a separate 8-column output; C3/C4 reject
   seven- or nine-column family caches and never reshape/overwrite parent rows.
5. **Targets are detached.** Backpropagate a batch containing main and family
   losses. Teacher/cache targets have no `grad_fn`; teacher parameters receive
   no gradients. Student main and family-head parameters receive gradients
   only for enabled losses.
6. **Mask, order, finiteness, and weighting.** Require main/family cache shapes
   `(n_rows, 2)` and `(n_rows, 8)`; row-coded targets/groups must match the
   frozen selected-row order exactly. Reject a swapped row, wrong group, mask
   mismatch, NaN/Inf, or wrong shape. Invalid
   `need_prediction AND is_scored` rows contribute nothing. For the evaluator,
   use labels whose magnitudes change under clipping and check pooled
   `global_wp_clip2_v1` against clipped targets weighted by their original
   clipped magnitudes, averaged over t0/t1.

## Provenance and leakage cases

7. **Ancestor closure and clean fit membership.** Reconstruct the allowed
   checkpoint lineage from the full-FT fit-valid subset plus each ancestor's
   train/valid membership. Reject any unaccounted ancestor, evaluation-fit
   group, corrupt train group (all 528), or cache row outside the declared
   clean fitting groups. A matching shape or split name alone is insufficient.
8. **Producer binding.** For each teacher and cache, verify bytes against the
   recorded hash; bind fold/split hash, row order, group membership, exact
   epoch-3 checkpoint hash, epoch, and producer config. Mutating any one field,
   swapping a same-shape cache, or replacing checkpoint bytes must fail before
   training. Reused teachers must be causal and fit only on the declared
   population.

## Fit, checkpoint, and serving cases

9. **Fixed paired recipe.** For C0-C4, both seeds 3/4 use fold 0, equal
   checkpoint epoch 1 after exactly two additional epochs, `lr=0.0001`,
   `chunk=2000`, the same batch/update budget, and no evaluation during fitting.
   The served seed-3/4 pair is fixed at equal weights. Permit only the
   predeclared change between each matched pair; C3 adds the family loss, C4
   changes only main-target normalization. A fallback must become a new matched
   recipe.
10. **Round trip and serving removal.** Save/load the selected epoch and prove
    the 23 parent output rows round-trip. The serving export emits only the
    existing prediction interface and does not require, expose, or execute the
    eight-output family head. Compare served predictions before/after stripping
    that head exactly.

## Compact golden arithmetic

Use `A=[[1,10],[2,20]]`, `B=[[3,30],[4,40]]`, `C=[[5,50],[6,60]]`,
`D=[[7,70],[8,80]]`. C2 must be `[[2.5875,25.875],[3.4,34.0]]`; C4 is that
array divided by `0.8125`. With `strongPAIR=[[4,40],[8,80]]`, C1 must be
`[[3.25,32.5],[6.5,65.0]]`. These values expose row/column swaps and
renormalization.
