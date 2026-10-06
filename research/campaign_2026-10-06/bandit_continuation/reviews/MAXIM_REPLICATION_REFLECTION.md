# Why Maxim-inspired replication has not produced a result

Read-only audit on 2026-10-07. A successful `git fetch origin` left the teammate branch at `833aaf54`. New experiment admission remains closed. This audit did not train, profile, benchmark or launch anything.

## Finding

The heterogeneous student experiment has not run. We therefore have an incomplete replication attempt, not a negative replication of Maxim's successful recipe. The only completed student comparison was ordinary supervised continuation C0. Its equal pair changed correlation by −0.00018902 against its warm-parent pair, with component-bootstrap interval [−0.00536403,+0.00354846]. That result does not test heterogeneous targets or the family head and cannot explain failure of those mechanisms. [Actual C0 report](../local/E502_C0_EVALUATION_v2/report.md).

## What Maxim actually submitted

The branch's wave20 public-result record binds the user-reported 0.6674 to `distilled_h256_pair.zip`, submission `26n9pg4y`, SHA256 `86020dcf384430ae5fd3a848c8a8b4533cd81031caad3b0feec7647c4a5491f9`. It contains equal-weight `w18final_distilled_s5@ep1` and `w18final_distilled_s6@ep1`. These are GRU256×2 students, not an established Transformer-teacher submission.

Wave18's final command warm-starts seed-specific `fbc1k_s{seed}@ep3`, trains two additional epochs at learning rate 0.0001, batch16/chunk2000, alpha0.7, propensity weighting, structural and next-event auxiliaries at0.3 each, and four-family loss at0.1. Its final training includes validation labels; those final training metrics are explicitly in-sample. Earlier fold-specific experiments supply development evidence, not the final model's holdout score.

Main teacher target is an unnormalized weighted sum: one strong teacher0.30, two full-family teachers0.10 each, two raw-family teachers0.125 each, and organizer baseline0.0625; total0.8125. It learns the main blended target and individual family predictions. [Exact source snapshots and manifest](maxim_replication_20261007/source_manifest.json).

## Our implementation: matches and differences

- The proposed C3 matches the main alpha, total coefficient scale, two extra epochs, learning rate, batch/chunk, main-label weighting and auxiliary coefficients. `src/wnn/distillation.py` explicitly materializes the weighted sum; using one main cache avoids the old loader's teacher-count division.
- Our strong family averages two different own checkpoints at0.15 each; Maxim uses one strong checkpoint at0.30. Our warm parents and full/raw-family checkpoint histories also differ. Family names and coefficient totals do not establish equivalent teachers.
- We deliberately replace organizer baseline0.0625 with two new clean same-fold raw-only GRU128 teachers at0.03125 each. This addresses uncertain organizer fitting provenance for our local comparison, but changes the teacher target. It is an independent analogue, not exact checkpoint-level replication. Those new teachers have not been fitted.
- **Unregistered objective mismatch:** our `scripts/train_rnn.py` clamps family targets to±2; Maxim's family loss uses raw exported predictions. Both use clipped original labels for row weights. Distill builder confirms the extra family-target clipping came from carrying over a convention, not an explicitly approved replication change. Effect size is unknown until tested; it cannot explain an unrun result.
- Maxim's separate `Linear(hidden,8)` family head receives the same hidden readout features as his main final linear layer. Our appended eight linear rows receive those features too. This arrangement is structurally equivalent as a parameter partition; it is not evidence we attached the head to the wrong representation. Initialization/RNG conventions still differ and should be declared.
- Maxim's direct comparison in wave12/wave18 retained the family head while setting main alpha to0. Our C0 has no family head. Our C2-versus-C3 isolates the family-head addition, but C0 is not his exact direct comparator.
- Our exposed fold/group exclusions, seeds3/4 and evaluation units differ from final all-data seeds5/6 and from his earlier screens. Absolute local scores cannot be compared across them or translated into public deltas.

## Who owned what, and what landed

| Role | Deliverable / actual activity | Scientific result |
|---|---|---|
| Root controller | Admission, sequencing, runs, result synthesis | C0 pair fitted/scored; four existing-teacher train exports completed; no C1–C4 fit |
| `distill_builder` | Trainer/cache/family-head implementation and synthetic tests; later E510 packet and cache/profile hardening | Code only; latest isolated commit437a9af1 passes22 focused tests, but no teacher/student fit |
| Cloud actor/operator | Prepare, push and monitor E508 profile prerequisite | V1 failed manifest-key binding; correctedV2 failed train-parquet footer identity before profiling. No production teacher fit |
| Evaluator / specification critics | Evaluator, split/cache checks and independent review | Found genuine target-index, RNG and cache-closure issues; no quality candidate generated |
| Native H196 builder and critic | Relational-distillation code inspired by MiMo, independent review | Actor tests only; no training; pair-cap/exact-source-binding review issues remain |
| External ideators / Solar actor | Diverse proposals and widening draft | MiMo proposal landed; some coding routes denied. Solar draft failed a tiny API probe; GLM timed out; Grok4.7 unavailable. No novel-arm fit |

Distill builder explicitly confirms launch/training/scoring was outside its delegated scope and owned by the controller. Producing correct code within that scope is not equivalent to producing a new model. Infrastructure failure and external access failure are not scientific evidence against the hypotheses.

## Controller mistakes and bottleneck

We expanded to multiple ablations, new baseline teachers, cache integrity machinery, diverse-provider ideation and serving work before obtaining one completed heterogeneous student result. Some infrastructure was necessary for laptop safety and clean comparisons. But the critical path was not isolated tightly enough.

In particular, even C1—the homogeneous control using already available strong teachers—requires the complete four-family cache and C3 profiling in its registered dependency list. That unnecessarily couples an independently useful comparison to missing baseline fits and cloud setup. A separate minimal, reviewed C1 recipe could have avoided that dependency; it would be a new attempt, not a silent alteration of the frozen one.

The failed profile also omitted the observed footer digest, leaving diagnosis less actionable. The production packet correctly refuses to proceed without real profile evidence, but healthy local scientific arms should not inherit unrelated cloud prerequisites. The controller, rather than the builders, owns this prioritization failure.

## Steering on explicit resume

1. Give one lead actor ownership of a complete result: independently implement a declared Maxim-compatible recipe, produce the reviewed packet, fit the fixed student pair under the controller's admission, export predictions and return a comparison. A code commit alone is an intermediate milestone. State raw family targets versus clipped targets explicitly and preserve the current recipe as a separate historical arm.
2. Make a compact recipe-difference table authoritative: checkpoint ancestry, organizer substitution, teacher coefficients, clipping, auxiliary losses, warm initialization, sampler/scheduler, data exposure, endpoint and served blend. Declare remaining differences before the run; do not call an analogue an exact replication.
3. Separate dependencies by arm. Complete missing teachers only for the arms needing them. Have cloud operator resolve input identity with observed/expected hashes and mounted versions, while a separately admitted local route remains available. No weakening of row/split/resource checks and no automatic retry.
4. Run the closest heterogeneous/family-head analogue with a matched control promptly after its own checks pass. Do not require all ablations or a new creativity round before obtaining that first result. Then use C1/C2 and a raw-versus-clipped family-target child to explain the effect, as budget permits.
5. Give the critic a fixed review window and concrete falsifiers, then send unresolved implementation errors back to the actor. Avoid endless scope expansion into generic infrastructure. Grader owns actual prediction evaluation; actor cannot approve itself.
6. Update bandit beliefs only from relevant measurements: C0's result belongs to ordinary continuation; cloud errors contribute cost and reliability evidence; unrun C1–C4/H196/widening retain unknown quality. Reserve diverse agents for bounded exploration after the leading path has a result owner and runnable dependencies.

No score gain is promised. Our public best remains0.6660; Maxim's reported0.6674 is still ahead. All experiment admissions stay closed until the user resumes.

Independent read-only audit by `replication_audit_luna` (gpt-6-luna/max) agrees: family-loss denominators are equivalent, the raw-versus-clipped family targets are the concrete objective discrepancy, the linear-head partition is structurally equivalent, and completed exports do not constitute a realized eight-source cache. It independently traced the final seed5/6 submission and distinguished it from our unrun fold0 seed3/4 analogues. No new scientific work was performed for this audit.
