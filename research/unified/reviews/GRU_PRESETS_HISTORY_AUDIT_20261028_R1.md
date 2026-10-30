---
kind: review
status: complete_historical_audit_not_fit_acceptance
created_utc: 2026-10-28
experiment_ids: [E-GALAM-QUALITY8-20261028]
---

# GRU presets: what the existing research supports

The best-supported starting recipe is a plain H256x2 GRU with a linear/GELU
encoder, encoder skip into the head, causal event features, chunk1000, batch32,
peak LR1e-3, recurrent-gate Muon plus AdamW, and a short OneCycle schedule.
It is a historical prior, not an established optimum under our now-mandatory
official train -> complete official valid boundary. Width384 is a worthwhile
quality candidate; width512x1 is a separate, promising but mixed-evidence arm.
The new schematic GALAM-GRU72 is a different model family.

Audit source: local commit a52ccafb01a6ceb9cbef46bdde381d3fed31ae78. Three
independent source/recipe/evidence reviews are preserved in the
[audit packet](../../experiment_drafts/galam-l4-quality8-20261028/gru_history/README.md).
GitHub connector reads returned404 and commit search422, so this report does not
claim that remote main or all current PRs have been refreshed. No SSH, data
transfer, model loading, prediction replay, fit, GPU profile or test was run.

## Historical recipe cards

| Role | Architecture | Optimization and budget | Interpretation |
|---|---|---|---|
| Scratch anchor | GRU256x2, input211, no attention, dropout0 | Muon recurrent gates; AdamW elsewhere; peak LR1e-3, wd.01, B32/C1000, OneCycle, usually3-4epochs | Strong prior for a fresh baseline; old fit membership/propensity are inadmissible for new primary validation |
| Wider teacher candidate | GRU384x2, otherwise matched to anchor | Same short recipe; dropout0 first; .1-.2 and averaging are separate treatments | Wider models often helped in old screens, but no universal/current chronological win |
| Width/depth alternative | GRU512x1 | Matched training target/update budget to its own control | Positive short-F0 student result; mixed other evidence, not a substitute for the anchor by score alone |
| Historical HPO leader | H384, hidden-scope Muon, dropout about.16, WSD-linear and SWA | LR about1e-3; exact trial35 record is in the recipe review | A jointly selected package; individual effects and optimum not established |
| Warm distilled student | GRU256x2, own strong parent initialization | AdamW LR1e-4, B16, C1000-2000, alpha.7 or.9, auxiliary targets, typically2-6epochs | LR1e-4 belongs to continuation here; not evidence that a fresh raw112 GALAM should use it |

The documented default scratch graph is:

`raw112 + causal99 -> train-only z-score -> Linear211x256 -> GELU -> GRU256x2`
`-> concat(recurrent_output, encoder_output) -> Linear512x256 -> GELU -> readout`.

The exact recorded HPO trial35 settings are H384x2, dropout.1592949958,
batch32/chunk1000, four epochs, seed3; LR.001033319571, Muon hidden multiplier
.7771685584, wd.00313357239; WSD-linear with warmup.05, decay.1038307290 and
finalLRratio.0323694645; SWA last.2318759432 of the budget. Its WMSE package also
sets label-weight power.8146287591/floor.1068298558 and class3weight1.2320221433.
Event records are `hpo/studies/teacher_v2/events.jsonl:209,241,273-274`;
fold1 .66959417, fold0 .67992665, adjusted two-fold objective.66876352.
The event's restarts=2 field is **inactive for WSD**, not two executed restarts.
This is a jointly tuned historical package, not evidence for any isolated knob;
no standalone exact executable replay command was recovered.

Source: [October10 solution explanation](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/solution_report/CURRENT_BEST_SOLUTION_EXPLAINED.md),
`src/wnn/nn.py:RNNModel`, `src/wnn/features.py`, and the actual
`artifacts/runs/combo_scratch_f0ep3s3_onecycle_f0/provenance.json`.
Zero recurrent state at each sequence; carry and detach between TBPTT chunks.
The old plain model calls nn.GRU on a whole chunk. GALAM currently calls its
recurrent/attention/mixer graph once per timestamp from Python. Parameter count
does not predict their relative throughput; a new device profile remains needed.

Historical main loss is clipped-label-magnitude weighted MSE, with propensity
and true-mask weights where declared; aux_next and aux_struct defaulted to.3
each in the cited anchor. The synthesis/default scheduler specifies5% warmup
and cosine OneCycle decay. The inspected run provenance alone does not bind
warmup: its profile_warmup field is a profiling field, not scheduler evidence.
The old scratch used FP16 autocast/GradScaler; legacy Muon Newton-Schulz CUDA
work used BF16. A new FP32/BF16 recipe is an explicit adaptation, not exact replay.

## What the scores actually establish

Do not compare the following rows numerically across evaluation panels.

| Source-qualified endpoint | Recorded WP | Evidence scope |
|---|---:|---|
| Scratch256 control, old F1, seed3 | .66604948 | Exposed component-held fold; train plus other valid folds fitted |
| H384 scratch, chunk500, old F1 | .66894555 | Same report's selected teacher; different recipe, not universal improvement |
| H230 four-family student, old F1 | .66923177 | Warm-parent/distilled endpoint; conditional paired delta+.00318229 versus scratch, not gate/LR-only attribution |
| Clean-T3 H384 teachers, old F0 | .67898918/.67836691/.67969849 | Three training seeds, same repeatedly exposed F0; 'clean' means excluded-fold ancestry, not today's train-only protocol |
| Equal T3 teacher trio, old F0 | .68090199 | Actual prediction ensemble; distinct from the mean of single-model scores |
| T3 student seed3/4, old F0 | .68115536/.68125842 | Shared teacher cohort and warm matched-parent recipe; not independent teacher replication |
| Older Z4 parent -> student, old F0 | .68282334 -> .68314181 | Only+.00031847 over its own strong parent; interval crosses zero, different from comparing it to a weak scratch |
| Best preserved team public snapshot, October17 | .66832 | Production ensemble lineage, rounded tl3/tl4 records; exact portal-ID-to-ZIP binding unavailable |

Primary sources: `solution_report/fold1_ci.json`,
[model catalog](../models/CURRENT_MODEL_CATALOG.md),
[archived model synthesis](../archive/2026-10-12/pages/02-models-and-training.md),
and [validation scope table](../validation/VALIDATION_PROTOCOLS.md).
The production tl3 members were GRU256x2 descendants using211/263features,
distillation/tied-lead objectives and train+valid fits. Public success supports
that complete lineage, not a lone hidden-size or optimizer effect.

In the matched short student-frontier F0 screen H512x1 scored.67323776 versus
H256x2 .66977764 (delta+.00346012; fixed-prediction component95% interval
[+.00200307,+.00490448]); H384x2 scored.67168804. H256x1 was.6693463, while
H192x2 was.6686534. A different cold F3 width/depth comparison was essentially
tied. These are recipe-specific, one-seed development results. The larger scalar
on a different fold is not evidence of a better architecture.

## Lessons for the next allocation

1. **Distinguish scratch training and fine-tuning.** The historical strong
   scratch prior uses LR around1e-3; successful warm students use around1e-4.
   Architecture, targets, initialization and features account for much of the
   gap. Importing the student's LR into a small cold model is not a reproduction.
2. **Do not buy20epochs automatically.** An old F1 six-epoch run peaked around
   epoch3 at.6671 and ended at.6490. The47-record effects screen contains a
   six-epoch delta-.01739 and an eight-epoch/high-Muon/EMA package delta-.03623.
   These are concrete negatives for those old recipes, not proof against every
   long fit. User-accepted20epochs is a cap, not a discovered optimum. A20-epoch
   cosine schedule changes early optimization too; checkpoints do not undo it.
3. **Widening is a candidate, not a scaling law.** The47-record screen reports
   H384/dropout.2 delta+.002780, with interval[-.002524,+.008083]. H384 appeared
   favorable across several historical cells, but selection/seed/panel differences
   remain. H512 also had a poor old scratch result despite its later student win.
4. **Prefer aligned loss before exotic objectives.** In that old screen corr-only
   WP was.08369 and Huber.65710 versus baseline mean.66642; adding a correlation
   term had no resolved advantage. Those failures do not reject all formulations.
   Later fixed-ep5 F0 WMSE/control.674830, NLL.675569 and EMA+sparse+NLL.675909
   make NLL worth a matched follow-up, not a proven new default. EMA alone was
   .674779 and sparse alone.674935. One seed/four unadjusted contrasts do not
   establish stable individual or combination gains.
5. **Do not call averaging universally beneficial.** Weight EMA, input-history
   EMA, prediction averaging and teacher-target averaging are different changes.
   HPO's SWA selection is a package result; recent input EMA alone did not win.
6. **Audit the actual target arithmetic.** T3 targets can be
   `.1*clip(y)+.9*clip(.8125*mean(three teachers))`. Nominal alpha alone omits
   teacher shrinkage/clipping and true-label magnitude/auxiliary supervision.
   Bigger teacher scores or more families did not consistently improve students.
7. **Use existing Muon code carefully.** `src/wnn/optim.py:muon_param_groups`
   updates each of the three GRU/four LSTM gates separately, with AdamW on
   input/output/vector parameters; hidden scope also includes hidden readout.
   This exists in the old trainer. The new sequence/GALAM path does not yet
   support that grouping. It is distinct from the proposed attention/FFN-only
   hybrid arm. Preserve RMS scaling, group ownership and declared precision.

## Prospective presets and plan consequence

Include a **fresh plain-GRU256x2 anchor** among the eight fits, replacing one
lower-priority tuning slot such as the standalone LR arm, rather than calling
GALAM-GRU72 the historical GRU baseline. Retain GALAM-GRU72 as its own full-model
alternative. This is a declared system-level benchmark: using211causal inputs
versus GALAM112 changes representation as well as architecture. A raw112 matched
plain-GRU control answers the narrower architecture question. Do not conflate them.

For a quality-oriented plain teacher candidate use H384x2 with the matched causal
representation and optimizer recipe. H512x1, dropout/SWA and NLL remain distinct
follow-up candidates; do not silently combine every selected knob. Student/KD
recipes require newly fitted train-only teachers and count those fits separately
from the eight scientific slots. Old fitted weights/caches cannot be reused.

For the first clean adaptation freeze: officialtrain-only fitting and moments,
full officialvalid reporting, random fresh weights, past-only event features,
WMSE with declared train-label weights, no valid-derived propensity/sampling,
profiled batch starting32, chunk1000 (2000 as a named treatment), clip1 and
explicit precision. Aux heads are conditional on source/target boundary checks;
future auxiliary **labels** inside fitting sequences are not future inputs.
The usual train mask is unavailable, so do not fabricate is_scored or infer time
from seq_ix. Required changes from the old recipe must be recorded as adaptations.

Keep20as a maximum horizon with checkpoints and full-valid monitoring, but
declare scheduler, best-checkpoint and early-stop policy before admission.
Choose an update horizon consciously; history does not support stretching all
old short OneCycle settings blindly to20epochs. Report epoch5/10/20 where
completed and valid-selected best separately. A best-valid result is exposed
model selection, not independent confirmation. The optimizer/batch/epoch values
above are priors until actual source-bound chronological fits and L4 profiles.

The existing plain sequence adapter accepts raw112 only and the new live monitor
currently requires GALAM. A211feature plain-GRU live-monitored run therefore needs
explicit adapter integration; it is not ready merely because the old trainer exists.
No current config/hash was changed or scientific fit admitted by this audit.
