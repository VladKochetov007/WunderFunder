# Leading continuation: heterogeneous teacher compression

Inspect the independently archived [public-result source](../campaign_2026-10-06/branch_teacher_audit/post_fetch_833aaf54/iterations/2026-10-05-wave20/public-result.json)
and [branch evidence](../campaign_2026-10-06/branch_teacher_audit/RECENT_OTHER_BRANCHES.md).
The friend branch records user-reported public0.6674 for a two-member distilled
GRU256 pair, October5, submission26n9pg4y,24m57s,640MB. Our root confirmed
public remains0.6660. The portal was not independently queried in this audit.
No friend implementation is merged or reused.

## Task for the independent builder

Implement and test a protocol inspired by the successful heterogeneous compression,
using our own trainer and current checkpoint/data contracts. Improve the complete
served endpoint, then examine whether the new student and our champion blend well.
Do not equate teacher accuracy, local cohort improvement and public improvement.

The source recipe is more specific than increasing distillation alpha:

- Causal teacher families include a strong bootstrapped GRU, full-fine-tuned GRUs,
  raw/wMSE-specialized GRUs and the organizer baseline. The following child replaces
  the raw family with two-stream teachers.
- Final student:GRU256×2 initialized from its own parent at epoch3; two additional
  epochs,lr0.0001,chunk2000,batch16 in the source run; student seeds5/6, fixed
  epoch1 outputs, serving weights0.5/0.5. These source resource settings are not
  permission to exceed our caps.
- Main label mix:alpha0.7, clipping original and teacher targets at±2, original
  target-magnitude weights and true-mask alignment retained.
- A training-only8-output head predicts the two targets of each of four teacher
  families. Its extra weighted-MSE coefficient is0.1. This preserves family
  distinctions rather than training solely against their collapsed average;
  it is discarded for the served two-target endpoint.
- Full-data teacher coefficients are0.3 strong,0.1+0.1 full-fine-tuned,
  0.125+0.125 raw/wMSE and0.0625 organizer. They sum to0.8125; the source
  cache builder does not normalize this sum. Retain exact source scales in a
  recipe reproduction and isolate scaling/shrinkage in controls. Do not silently
  replace this with a normalized arithmetic mean.

Primary code was read through Git only to reconstruct the protocol. The archived
manifest binds source note/JSON bytes. Cache producer epochs, chain fit membership,
organizer pretraining uncertainty, original normalization and all teacher scales
must be audited before forming our own cache. Final all-data caches are in-sample
and cannot support holdout claims. All previously inspected folds remain exposed.

## Bounded comparisons, then deeper children

Declare paired student seeds and a frozen endpoint before fits. First compare
unchanged continuation, homogeneous teacher distillation, heterogeneous scalar
teacher distillation and heterogeneous distillation with family auxiliary heads.
Match target scale, labels, update clock, initialization and training budget;
retain both source-scale and matched-scale comparisons as distinct questions.
Do not sweep all factors at once or select only the best seed.

After a promising root comparison:

1. Replace one teacher family with causal two-stream/event-context models, keeping
   student architecture, pair construction and the mixture otherwise fixed.
2. Transfer teacher-family contrasts or a small residual basis rather than every
   raw teacher output; test whether a different representation preserves useful
   diversity and improves quiet/class3 errors. Compare against the unchanged
   family-head representation, not only a weak no-distillation baseline.
3. Widen a trained distilled student by a function-preserving construction, prove
   pre-training output parity, then compare matched narrow/wide continuation.
4. Export aligned predictions and test a frozen mixture with our champion. A gate
   or learned mixture is a separately selected arm; include selection exposure.

Give ideators different source documents and keep initial proposals blind. Assign
an independent critic to cache provenance, main/family-loss weighting, discarded
heads, function parity and the serving endpoint. Record model/tool/prompt/source
provenance. Current allowed subagent setting remains Luna/max; later model/tool
rotation follows the user's explicit selections.

Aggregate caps:5CPU,3decimalGBVRAM,40GBRAM. No new local experiments after
October6 10:00Kyiv; new experiments must use admitted Kaggle compute. Save every
attempt and EDA figure under the research lineage; append ledger and journal
records and rebuild the tree. This brief launches no experiment or submission.
