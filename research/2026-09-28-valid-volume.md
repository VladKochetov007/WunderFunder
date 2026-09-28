# Full-validation data dividend: preregistered comparison

The measured public champion is `final_since2_gclp.zip` at 0.6622. Its final
parent and true-mask fine-tune used the 1,505 development validation sequences;
the 368 reserved sequences were held out. This iteration tests whether the
extra labeled validation chains improve a two-GRU prediction at fixed inference
cost. The official metric is global weighted Pearson on the published exact
scoring mask, after clipping target and prediction to [-2, 2].

## Locked design

- Fold-specific parent: seed 3, four-epoch EMA, propensity power 2, floor .03,
  true-mask floor .03, since-event weight 32/floor .25, auxiliary structure .3,
  auxiliary next-change .3; two-stage gcorr fine-tune (head frozen trunk first).
- Within each held-out chain fold, compare 80% versus 100% of the other
  validation chains. Selection is seeded with 20260928 and stratified by
  market class; no chain is split. Both arms draw 12,000 sequences per parent
  epoch and 1,408 per fine-tune epoch with replacement, holding optimizer
  steps and validation-source sampling mass fixed.
- Primary comparison: each candidate replaces seed 3 in a 50/50 pair with the
  fixed fold-specific seed-4 anchor `ab62_base_s4`. Score all exact-mask rows on
  chain-pure held-out sequences. Report fold and pooled paired chain-bootstrap
  deltas; lambda .02/.05 are sensitivity checks only.
- Fold 0 must gain at least +0.0005. Proceed through fold 1 only if it does.
  Promotion requires positive gains on both folds, pooled exact-mask gain at
  least +0.0005 and larger than one paired bootstrap SD. Also require the
  100% pair to beat the existing seed-3-since plus seed-4 OOF pair; beating a
  weakened 80% control alone cannot justify an upload. If promoted, repeat
  on untouched folds 2/3 before fitting a final model. Avoid tuning on the
  reserved block, which becomes final-fit training data.
- Final candidate: fixed last-epoch seed-3 parent trained on all 1,873
  validation sequences plus train, gcorr fine-tuned on all 1,873, paired with
  the existing champion seed-4 member. Verify two members on v2 and v3,
  batch/native parity, package size and one-vCPU projected runtime <=55 min.
  A public upload is justified only after all gates pass.

## Run log

`scripts/run_valid_volume.sh` contains the matched commands (the executed
copy is `artifacts/logs/q45_valid_volume.sh`). An initial
fold-0 run was stopped before completion because unstratified chain selection
changed the class mix. The replacement sampler selects whole chains within
each class. Fold-0 full-versus-80% class shares are approximately
`[.2228,.1815,.2199,.3758]` versus `[.2230,.1805,.2212,.3752]`; fold 1 is
`[.2422,.1702,.2090,.3785]` versus `[.2395,.1707,.2125,.3772]`.
The exact-mask row rate is 0.09678 versus 0.09600 on fold 0 and 0.09591
versus 0.09591 on fold 1 (full versus subset).

After the 80% control finished but before the 100% candidate produced a score,
the champion-aware comparison was made an explicit additional gate. The 80%
control's final fine-tuned single score was lower than its parent, so the
volume contrast alone could approve a model that still loses to the incumbent.

## Fold-0 result and decision

The 80% versus 100% fine-tuned seed-3 pair with the fixed seed-4 anchor
changed exact-mask WP by **-0.000116 ± 0.000922** on 432 pure-chain
sequences. At lambda .02 the change was -0.000068 ± 0.000867. The full-data
pair also lost **-0.000972 ± 0.000938** exact-mask WP against the existing
`ab62_since` partner (-0.001165 ± 0.000868 at lambda .02). The 100% parent
finished below the 80% parent at the fixed final epoch (0.673811 versus
0.674558), although its standalone final gcorr score was higher (0.673539
versus 0.672430). This is precisely why the pair was the decision statistic.

**Reject this arm at fold 0.** No fold-1 fit, all-validation final model,
package, or public submission follows from it. The experiment tests a
replacement-sampled, matched-step recipe; it does not prove that all ways of
using the extra validation block are useless. The tested recipe cannot beat
the current pair on clean chains.

The guarded `--include-heldout-valid` final-fit loader passed a two-batch GPU
smoke check: it reported 1,873 validation training sequences and the 368-row
group reserved block as its diagnostic evaluation source. No final model was
trained or selected from that in-sample diagnostic.
