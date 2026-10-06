# Recent other-branch teacher/student audit

Audited `origin/research/creative-iteration-20260930@833aaf54` (the 6 Oct merge snapshot). The new merge records the local October experiment snapshot. Its teacher-strengthening screen is negative; the snapshot also carries a later public result for an earlier heterogeneous student recipe.

## Public result and provenance

`research/iterations/2026-10-05-wave20/public-result.json` records submission `26n9pg4y`, `distilled_h256_pair.zip`, score **0.6674**, 1,497 seconds, 640 MB. The source is explicitly “user-reported competition result.” Its archive hash, `86020dcf…a5491f9`, matches `wave20/candidate-manifest.json`; the uploaded body was not independently fetched. `research/CURRENT_STATE.md` reports this as +0.0008 over that branch’s prior 0.6666 record. This is the only public result found after that record: waves 24–46 contain no public-result or public-update artifact, and the state note ends with wave22 research. Reconcile the branch’s 0.6666 comparator with the root campaign’s current 0.6660 record before making a current-best comparison.

The submitted pair is `w18final_distilled_s5@ep1` and `w18final_distilled_s6@ep1`, each hidden 256, equal weight. `wave18/final_student_lane.sh` records a two-layer GRU256 student, seed-specific `fbc1k_s{seed}@ep3` initialization, two training epochs, learning rate 1e-4, and `--distill-alpha .7`. Its full-data teacher target is the direct weighted sum

`0.30*w18final_fbc1k_s5 + 0.10*w18final_ffull_s3 + 0.10*w18final_ffull_s4 + 0.125*w18final_frawW_s3ft + 0.125*w18final_frawW_s4ft + 0.0625*orgbase`.

The coefficients sum to 0.8125; `prepare_full_teachers.py` sums weighted outputs without renormalizing. The public pair was final-data trained with held-out validation included, so its local training result is not a holdout score. The checked manifests name recurrent/family sources; they do not establish a Transformer teacher in this public recipe.

## What replicated locally

`wave12/wave-report256.json` reports the fixed distilled pair at **0.7181503** versus its teacher at **0.7153250** (+0.0028253; conditional chain-bootstrap 95% interval [+0.0019575,+0.0038363]) on 425 true-mask sequences / 123 chains from the reused fold-0 validation. All train chains were excluded because the teacher used all training data; this is not a train holdout. The direct-supervised student scored 0.7165485, but distilled-minus-direct uncertainty crossed zero. Organizer pretraining provenance is uncertified.

`wave18/composite-report.json` is a distinct train-half-0 replication: known learned teachers and students were restricted to train half 0 and validation folds other than 0. On 436 sequences / 129 chains, the distilled pair scored 0.7126204 versus its teacher’s 0.7108775 (+0.0017429; interval [+0.0005600,+0.0029585]). The source calls the train mask estimated and diagnostic only. In wave19, adding a two-stream family changed the student pair by +0.0002137 on the same reused fold-0 validation; `wave20`’s public pair still uses the earlier wave18 students, so that later teacher change has no public score.

The new `wave20` result should be kept separate from the new `research/iterations/2026-10-02-distill07` screen: alpha .7 versus the incumbent alpha .5 scored 0.7022527 vs 0.7023797 (delta −0.0001269; conditional interval [−0.0003517,+0.0000974]) on exposed folds 0/1. It screens out that fixed recipe, not heterogeneous distillation as a family.

## Practical next tests

- For a new Transformer teacher, fix the student, initialization, optimizer, sampler and target blend; compare Transformer-only distillation with the current teacher blend and a supervised student. Generate teacher predictions without the evaluated fold’s labels, then score every arm on the same untouched fold and report seed variation separately.
- Test teacher complementarity before combining teachers: one predeclared teacher-family swap at a time, with the student recipe held fixed. Wave19’s small positive shift is a useful hypothesis; it does not yet transfer to the public package.
- Treat the public result as transfer evidence for its exact ZIP only. Preserve the package hash and source of each leaderboard number, and keep local validation gains and serving-time changes as separate claims.

Controller clarification: no untouched local fold is established in this campaign. The proposed comparisons must use the same declared split and preserve its exposed status; fresh confirmation is a future option only if genuinely uninspected evidence becomes available.
