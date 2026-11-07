# Independent review of portfolio README

**Date:** 2026-11-07  
**Scope:** README cards A-D, validation, hardware, and claims only. No experiments, profiles, model imports, or other file reads.

## Verdict

The README is appropriately explicit that the architecture cards are proposals and that scores, throughput, CPU serving, and public/private transfer are not forecast. It is not a runnable admission packet yet. Resolve the following five material items before using these cards to allocate fits or treating any result as deployable.

1. **Card A is not a CPU-serving candidate yet.** Root's fresh adapter inspection found the old Mamba-2 adapter at bd78 pinned to CUDA-only mamba-ssm 2.2.4, with no CPU serving path; no actual profile exists. Card A can remain a teacher/research hypothesis, but its "highest new representation priority" must not be read as a qualified final model. Before admission, name the compact student architecture and distillation ancestry, or explicitly scope A to a teacher-only learning-curve experiment; then profile the actual implementation and separately qualify the CPU-served artifact.

2. **PR67 status is prepared, not executed.** Fresh root status is a ready handoff with zero fits and a chronological manifest. Keep "after receiving acceptance" and make zero completed fits explicit wherever PR67 is called runnable or strongest handoff. PR61's terminal comment reports 10/12 fits; the two Transformer scores, 0.6717207 and 0.6729532, are teacher results. The student fits never started because the profile-identity gate failed before target construction. No teacher-to-student quality conclusion exists. See the [PR61 terminal comment](https://github.com/VladKochetov007/WunderFunder/pull/61#issuecomment-6639856413).

3. **Pin the exact mask contract next to the validation rule.** The plan should say explicitly: fit on all fitting-train rows after row 99; process warm-up and unscored rows causally; evaluate only with the official valid need_prediction AND is_scored mask; never use is_scored as a training feature or label-selection signal. The optional 211-channel input is 112 raw plus 99 derived event inputs, not 211 additional event fields. The global TARGET_QUANTA values discovered on validation cannot initialize a fit or act as a gate oracle. Keep the README's warning that PR65/66 F0/F2 recipes need official-policy adaptation and refreezing before any primary use. The current prose is directionally correct, but these details are admission-critical.

4. **Add metric and endpoint provenance to the PR61 score row.** The table labels TL3 validation, public TL3, and final all-label W18 as different evidence, which is good. The two PR61 Transformer numbers still lack an explicit metric/mask/checkpoint endpoint label in the table. State whether each is full official-valid clipped weighted Pearson and identify its checkpoint rule. Do not compare those values directly to clean TL3's 0.6923774 unless the exact validation membership, scorer, and checkpoint basis match. Preserve the dated public audit values as dated portal observations, not current live checks.

5. **Do not count overlapping hybrids/residuals as independent new families.** A, B, and C all combine recurrent or state memory with local attention; B and C are especially close unless the screen isolates completed-patch memory from immediate-row attention under matched controls. Also, the README's E card already contains causal EMA/difference summaries, and dense four-expert residual heads are already listed as reserves; the conditional-residual lens repeats these mechanisms. Keep them as one bounded residual family, not extra independent allocation arms. The card C phrase "proven recurrent trunk family" should refer only to the tested recurrent baseline lineage: PR61's GRU256x2 teacher mean was 0.6851345 versus the stronger TL3 reference, and the fixed 90/10 LSTM endpoint is a separate small blend result.

## Hardware and claim check

The README correctly marks the 0.72M count and other dimensions as estimates, all CPU/GPU throughput as unmeasured, and the eight-device packing as a candidate rather than a selected profile. Do not infer a B300/L4 fit budget or completion time from those arithmetic estimates. Keep CPU-only submission constraints as a final artifact gate even when a B300 teacher is useful.




## Review disposition after README revision

Root incorporated the five findings in the portfolio README. This closes planning/documentation blockers only; it is not empirical approval.

1. **Addressed in scope:** Card A is now explicitly teacher-only, describes the CUDA-only mamba-ssm adapter, and names GRU256x2 and LSTM192x2 as separate student options with additional cost. The adapter, profiles, and CPU package remain unqualified. I corrected the package name above: version 2.2.4 is mamba-ssm, not PyTorch.
2. **Addressed:** PR67 is described as a handoff with zero new fits or profiles, contingent on acceptance and its chronological manifest.
3. **Addressed:** The README now specifies fitting rows at indices >=99, causal advancement over every row, evaluation scoring by need_prediction AND is_scored, no train is_scored signal, event211 as 112 raw plus 99 derived inputs, and the exclusion of validation-derived TARGET_QUANTA from fit initialization/gating.
4. **Addressed as source-reported provenance:** PR61 is labeled full-valid WP at the configured four-epoch endpoint and explicitly not independently regraded. The README says not to treat these values as a matched architectural-effect estimate.
5. **Addressed as portfolio bookkeeping:** A/B/C are identified as sibling representation arms; E groups cheap residual readouts; conditional EMA/expert heads are not extra independent families; the patch arm needs its own local-memory control. Keep card C's "proven recurrent trunk family" limited to historical lineage evidence: its reported GRU256x2 teacher score does not establish parity with TL3.

No new sources, model imports, tests, profiles, or empirical claims were added.

