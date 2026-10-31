# [Research portfolio] Causal teacher–student and long-context hypotheses

1. **Status, lineage and owner.** One proposal-only draft requested for eight
   ideas. [Hypothesis cards](https://github.com/VladKochetov007/WunderFunder/blob/experiment/causal-teacher-student-hypotheses-20261101/research/experiment_drafts/causal-teacher-student-hypotheses-20261101/HYPOTHESES.md) and [recipe](https://github.com/VladKochetov007/WunderFunder/blob/experiment/causal-teacher-student-hypotheses-20261101/research/experiment_drafts/causal-teacher-student-hypotheses-20261101/recipe.json) distinguish
   teacher, student, connection, representation and objective branches. Base is
   `research-team` at `7e4074314a62d0907c9608f9220cf72958708a2d`; later source
   changes must be explicitly admitted. No new model implementation or launch.
2. **Why it should work.** Fixed teacher mixes may ignore time variation and
   complementary errors; long history and alternate state updates may improve
   teachers; shared-state students may retain diversity more cheaply. These are
   competing hypotheses, not conclusions. Each card names a matched control,
   alternative explanation and adverse observation.
3. **Compute advantage.** B300 compute can prepare diverse causal teachers,
   cross-fitted targets or longer-context models while retaining a small deployable
   student. Teacher fitting, cache export and evaluation are real costs; any
   reused checkpoints must have compatible source and fitting ancestry. No
   universal speedup, scale law or leaderboard gain is assumed.
4. **Comparison.** Each independent candidate has its own matched control,
   two paired seeds and fixed data/source, transforms, optimizer/schedule,
   updates, checkpoint rule and loss scale before admission. Candidate-specific
   parameter values and the exact common parent/config are not implemented or
   frozen here; they are mandatory in the mechanism-specific implementation PR.
   W18 and tiedlead have different epoch/EMA/initializer contracts and must not
   be treated as interchangeable by the historical fbc exporter.
5. **Fitting panel.** Seven cards propose 2 controls + 2 treatments = four new
   fits each. Disagreement weighting has three arms (constant, increased and
   decreased KD weight) × two seeds = six fits. All eight screens would cost
   **34 primary fits**, including teacher-arm fits in cards 5 and 6, plus any
   prerequisite teacher/cross-fit-parent fits and cache construction; none are
   scheduled. For card 6 choose one architecture (Gated DeltaNet or Mamba) and
   for card 8 one robustness objective before admission; testing additional
   variants requires an explicitly expanded fit budget. Reuse is conditional on exact split,
   ancestry, source and recipe identity. Two seeds on one split are not two
   independent folds. Details and potential extra teacher costs are per card.
6. **Validation and uncertainty.** Primary outcome is the official clipped,
   pooled, weighted two-target WP, on the same scored rows for every arm. A
   tentative working effect of +0.0005 WP versus the matched control motivates
   additional work, not a significance or automatic promotion claim. Record both
   seed deltas, t0/t1, component groups, total tried comparisons and uncertainty.
   Full valid and public feedback are development-exposed; conditional component
   bootstrap does not quantify fit variance or restore independent holdout.
7. **Leakage and representativeness.** Whole sequence/component boundaries;
   fit-only statistics and teacher/cache ancestry; no future rows or labels in
   teacher/student inputs, gates, tokenization or prefixes. OOF gating requires
   separately trained source-qualified teachers excluding the gate-fit units.
   Component matching is approximate; anonymous sequence IDs do not prove time.
   Pretrained models are subject to competition restrictions on external data.
8. **Decision.** A consistent practically useful gain with acceptable resource
   cost justifies a separate replication and complete served-candidate study;
   mixed/small gains remain uncertain, and negative results narrow only that
   recipe. An interval crossing zero alone does not close a direction. A stronger
   teacher without student transfer is not itself a ready submission.
9. **How to run.** Not runnable in this PR: missing exact configurations,
   adaptation, teacher/cache provenance, numerical acceptance, B300 profiles,
   receiver ownership and, for serving, Linux CPU access. No command is offered
   as an executable training recipe. Prepare/admit each card separately; do not
   mutate existing jobs or silently substitute artifacts.
10. **Evidence to return.** Code/source/config, split and all teacher/cache
    identities, source-qualified seeds and cells, losses/LR/gradient norms,
    updates, profiles, model and per-target pooled predictions, all paired
    contrasts, failures, plus separate evidence and model archive SHA256s.
    Serving candidates additionally need exact-ZIP parity, reset/causality,
    quality, size, one-vCPU runtime and memory measurements.
11. **Results and trace.** Zero new fits or profiles. Previous exposed PR52/53
    results and related existing drafts are linked in the [README](https://github.com/VladKochetov007/WunderFunder/blob/experiment/causal-teacher-student-hypotheses-20261101/research/experiment_drafts/causal-teacher-student-hypotheses-20261101/README.md).
    The eight H/E ledger entries refer to proposed work, not independent results.
    No promotion, submission or score claim follows from this draft.
