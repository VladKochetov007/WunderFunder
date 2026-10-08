# E538 independent code and experiment bug review

H232 V4 has two confirmed admission defects. Its valid real-device profile was rejected before fitting: the gate expects width 160 instead of the packaged width 211, and it rejects legitimate full-hash verification of auxiliary files. The identified V5 successor fixes both without changing the scientific recipe. The independent V5 replay and transport inspection pass.

A broader bounded audit reproduced a latent preprocessing-state defect when different recipes call the same imported trainer's `main()` in one process. No completed affected fit was established. It also independently reproduced the already corrected H230 incumbent comparison's fitting overlap. AMP skips advance the scheduler clock in actual delivered runs; this is a declared optimization choice, not a demonstrated score defect.

## Contract and evidence scope

Owner: `/root/sol_code_experiment_bug_review`, independently assigned GPT-6.1 Sol reviewer. Root Challenge 2 only. Worktree: `/home/vlad/development/wunder/worktrees/wnn-sol-code-bugreview-E538`, branch `review/sol-code-bugreview-E538`, base commit `9a6f1bd20f0e69e61791325b98fed3376ebcfd3c`. Newer raw receipts and pinned sources were read from the shared main workspace. No production edits, fits, GPU requests, provider operations, credential reads, grader changes or budget-authority changes were made. Support executions used `scripts/limited`, 1 CPU, 1.5 GB RAM, one thread and a 120-second bound.

Applied instructions: AGENTS.md; specs README/objective/current state and active allocation; ADR0012/0016; conduct-creative-research including empirical-validation; research-recipe-audit; research-evidence-review; bug-hunters; wnn-connectome-top1; Python standards. Existing E535/E536 reports were leads, not substitute evidence. Claims below use inspected source and newly executed small reproductions. Neither metric scores from large prediction arrays nor teacher ancestry were newly regraded in this role.

Source commits are stated where available. New untracked execution archives have no source commit to cite; exact archive/member SHA256 binds those claims. [TRAINING_INVARIANTS_REPRODUCTION.json](TRAINING_INVARIANTS_REPRODUCTION.json) records the reviewed source and provenance hashes, direct membership intersections, counters and numerical observations. [H232_V4_REPRODUCTION.json](H232_V4_REPRODUCTION.json) records the V4 source/receipt hashes and exact gate replay. Frozen originals remain unchanged.

## Findings and action map

Severity reflects the consequence in its identified execution path, rather than predicted performance gain.

| Finding | Severity / confidence | Demonstrated affected experiment or decision | Required action |
|---|---|---|---|
| F1: gate requires nonexistent full-feature width 160 | P0 admission blocker; proved | H232 conditional V4, terminal `reject_no_fit` | Preserve V4 as infrastructure-censored. Run identified corrected V5 profile → conditional fit; no V4 model to rescore. |
| F2: gate forbids expected auxiliary full-hash fallbacks | P0 admission blocker; proved | Same H232 V4 terminal rejection | Use scoped profile-window raw-path evidence and original-verifier fallbacks. Corrected in reviewed V5. |
| F3: trainer preprocessing flags survive a later invocation | P2 latent defect; raw/online triggers reproduced | Future mixed-recipe reuse of one trainer module; no demonstrated completed affected fit | Reset all preprocessing globals or launch a fresh trainer process before mixed-recipe reuse. Current H232 and H217 paths are unaffected by this trigger. |
| F4: incumbent comparison checks bytes/order without equivalent fitting-membership eligibility | P1 interpretation defect; direct overlap reproduced; previously corrected | H230 seed-3 grade's comparison with the two-component incumbent | Retain descriptive metric values, withdraw held-out inferiority interpretation and any rejection relying on that contrast. Matched H230 control and public scores remain valid observations. |

No blanket rerun of H230, C1, E521 or all earlier negative arms follows from these findings. No gain is predicted. Earlier receipts outside the exact paths above remain unaudited by this bounded review unless explicitly covered below.

### F1 — H232 V4 rejects the actual packaged feature width

Exact source: [V4 profile_worker.py](../continuous_execution_E529/h232_conditional_v4/packet/profile_worker.py), lines 928–929, `if n_in != 160`. SHA256 `0bbebb928a2a628cf4e953b8c088223f01bc57af206a7251eec5f5f2062cf20d`. Source archive SHA256 `afde296e1c5e2f0c47a6fb352c2dea8e0a3df75900f53c26f103b3921df1f565`; downloaded trainer member SHA256 `cc0821a144fbea572dfaa03832fbbf31cf1ea605a917b0943c3db5da6b763e1a`; downloaded `features.py` SHA256 `284c119d150b74bff0ccc2628bc2a195ca61c351a628b085b7d9dd85512bfeee`. These are the executed V4 archive, not the current main trainer.

Trigger and executed reproduction: `reproduce_h232_v4.py` imports the downloaded packaged `event_features`, runs it on a tiny `[1,3,112]` array and obtains 99 event channels, hence 211 model inputs. It then executes the exact V4 gate against the terminal trace, measured shapes, timing, identity and passing resource evidence. Trainer arguments are reconstructed from the gate's exact required-argument table; the terminal receipt independently reports zero argument mismatches. The replay returns exactly the two recorded rejection reasons, including `profile_input_width_is_not_full_feature_160`.

The actual train and validation export-shape probes each record `[16,4000,211]` model input and `[16,4000,112]` raw input. All 230 warmup/timed optimizer attempts completed with finite loss/gradient evidence and zero AMP skips. Four-epoch fit projections were 9,893 seconds p50 and 11,041 seconds p90; resource and lifetime checks passed. The terminal `profile_gate_result.json` records `fit_started=false` / `reject_no_fit`. This attempt never produced a negative scientific fit result.

Smallest fix and meaningful regression: derive the expected width from the pinned raw schema and actual pinned feature function, compare both declared count and observed output, then require the real profile/export shape to match it. A test that invents a width-160 fixture repeats the defect. V5 independently probes `[2,3,112] → [2,3,99]`, accepts the real width-211 shape and rejects width 160.

Affected scope: V4's exact gate/recipe archive only. No assertion is made that V1–V3 had this same gate. V5 keeps the same trainer/archive and scientific fit settings; a new actual-device profile is still required by the worker before fitting. Preserve V4 raw receipts and identify the retry; there is no V4 checkpoint to rerun evaluation on.

### F2 — digest admission confuses raw reuse with auxiliary verification

Exact source: V4 `profile_worker.py`, lines 937–942, requires aggregate misses and fallbacks to be zero. Its handoff is deliberately limited to raw train/valid files; authorization/identity binding is in the same worker around lines 2240–2290 and in [verified_digest_handoff.py](../continuous_execution_E529/h232_conditional_v4/packet/verified_digest_handoff.py), lines 96–103, 131, 212 and 250. Handoff SHA256 `c351f88ab9f040867da8e30f767133b8017cc8d15aefdf4c8af65040fb51688f`. The pinned portable verifier remains unchanged.

Trigger: the trainer verifies two registered raw files plus auxiliary inputs outside that registry. Those auxiliary lookups correctly use the original full-hash verifier. The actual profile-window delta is six lookups, two reuse hits, four misses and four original-verifier fallbacks. Pre-profile counters were 6/2/4/4; post-profile counters 12/4/8/8. Requiring 6/2/0/0 contradicts the registry's scope.

Executed reproduction: `reproduce_h232_v4.py` uses the exact handoff class with two tiny independently hashed raw files and four tiny auxiliary files. It verifies/authorizes only the raw paths, then invokes the patched portable verifier on all six. Every digest equals its independent expected digest; raw path counts are each 1/1/0/0, the four auxiliary files each call the original full verifier, and aggregate counts reproduce 6/2/4/4. Together with the exact V4 gate replay, this proves the rejection criterion is defective without bypassing file verification.

Evidence limit: V4's primary `input_identity.json` raw per-path counters are from initial loading. V4 did not persist post-profile per-path counters. Their initial raw reuse, unchanged exact stat bindings, aggregate window counts and tiny scope reproduction support the explanation; they are not a direct recorded post-profile path trace. V5 now persists genuine before/after path deltas around `module.main()`.

Smallest fix and meaningful regression: require exactly one verified reuse hit or one original-verifier fallback for each raw train/valid path during profiling; require misses equal full-verifier fallback calls and aggregate lookups equal hits plus misses. Retain digest-registry comparison, pinned verifier and unchanged stat-identity checks. Tests must accept 6/2/4/4, reject a miss with no fallback, reject missing/zero profile path evidence, and reject changed raw identity. An unpushed V5 draft initially reused preflight path counts; this reviewer identified that phase mismatch, the actor corrected it before freezing, and the zero-profile-delta test now rejects. No executed experiment used that draft.

Affected scope and remediation are the same V4 attempt as F1; this is not evidence of corrupt raw data or a teacher-cache verification bypass.

### F3 — preprocessing options are one-way global assignments

Exact main source: `scripts/train_rnn.py`, commit `f40bff87304db1fa3bd727bfd096035c958bba4b`, SHA256 `b49233a7ed7e7e44d1a67cd765879a96e9294a5934071840b77da0d37d7424c7`. Globals are defined at lines 516–523; `build_inputs` consumes them at 527–550. `main()` sets `RAW_ONLY=True` at 1810–1811, `HEURISTICS=True` at 1813–1816, `ONLINE_NORM=True` at 1819–1820, and `INPUT_PROPENSITY` at 1823–1824 only when each option is present. There is no false/None assignment for a later invocation. In contrast, research features, privileged mode and sentinel state are assigned each time.

The same one-way assignments exist in the actual pinned H230 source `continuous_execution_E529/h230_source_v2/scripts/train_rnn.py`, SHA256 `fc4f714396d6756237a6de4d237a06ab86bed17690fd6243b2f297e6b2195055`, at 1783–1797. That pinned source is explicitly the H230 execution path; current main lacking some archived H230 flags does not mean those flags were absent from its fit.

Executed reproduction: `reproduce_training_invariants.py` executes the original AST option branches and original `build_inputs`, with tiny data and no call to the fitting/data-loading path. Raw-only then full-feature options leave `RAW_ONLY=True`; all 99 event channels remain zero, while a fresh full-feature view has 384 nonzero entries in this toy. Online normalization then plain options recompute declared `n_in=211`, while the retained online flag makes actual input width 323. The heuristic and propensity cases are analogous source-traced risks, not separately executed reproductions here.

Scientific consequence: mixed recipes in one module can silently inherit the previous raw view, or fail dimension checks after an earlier feature augmentation. A run name and new CLI namespace do not reset Python module globals.

Affected scope: no completed wrong fit was established. H232 uses the same false preprocessing flags in its profile and conditional fit, so this trigger does not invalidate V4 evidence or V5. The H217 V2 worker (`worker.py:55–71`, SHA256 `11e3b98cc8aaeeff9db4ee152eaab0ebfbf1f17878d17845474cb3489dd94309`) launches a new subprocess for every phase; its internal `runpy` wrapper runs inside that child, so separate phases do not share these globals. A future warmed multi-recipe session, including a planned H248-style worker, is affected only if it reuses the same trainer module across differing preprocessing flags. Keeping data staged does not require doing that.

Smallest fix: assign every preprocessing global from the current arguments at entry, including false/None cases, or use a fresh trainer process/module per recipe. Regression: in one process, raw→full, online→plain, heuristics→plain and propensity-input→none must each match a fresh module's declared width, loader shape and built input values. No historical fit needs rerunning without actual mixed-process provenance demonstrating the trigger.

### F4 — the H230 incumbent contrast has unequal fitting exposure

Exact grader source: [H230_SEED3_INDEPENDENT_GRADE_V1.py](../wide_baseline_heterogeneous_H230/H230_SEED3_INDEPENDENT_GRADE_V1.py), SHA256 `45c42ea92a45d1e476e2b4aff5b66efe06c5f44a5a8d1db05143681e5d08dd47`. Candidate membership is checked at lines 304–307. Incumbent components receive byte/order checks at 384–399 and package identity checks thereafter; lines 467–475 compute the same-evaluation-group 0.5 parent + 0.5 head contrast without an equivalent incumbent fitting-membership eligibility gate. This archived file has no tracked commit in the inspected workspace; its exact digest binds the finding.

Executed reproduction: directly intersect original `provenance.json` fitting-validation group lists with the original fold-0 evaluation membership. Both `c20261004_combo_s4_final` and `c20261004_combo_head_s4_final` fitted 374 of 468 evaluated groups. `bd26_h230_hetero_s3_f0` and `bd26_h230_familyoff_s3_retry1_f0` each fitted zero directly overlapping validation groups. Exact provenance and split hashes are in the reproduction receipt. Recursive teacher/propensity/init ancestry was not newly recomputed in this check.

Scientific consequence: the H230-vs-incumbent −0.008115 contrast is descriptive, but cannot establish held-out inferiority or justify rejecting H230 on that basis. This reproduces the prior membership correction; it is not a newly discovered additional leak. Preserve the original grade and [membership correction](../wide_baseline_heterogeneous_H230/H230_INCUMBENT_MEMBERSHIP_CORRECTION_V2_20261008.json).

Smallest fix and meaningful regression: before a held-out contrast, validate fitting and relevant teacher/init/propensity ancestry for every endpoint, then reject or label the contrast when eligibility differs. Use a fixture with a clean candidate and one contaminated blend member; byte equality/order must not make it eligible. Exposed-but-unfitted selection groups still need their exposure label.

Affected decisions: the specified H230 seed-3 incumbent comparison and any downstream rejection explicitly based on it. The matched family-on/off observation, 0.681719502 versus 0.681658487 (+0.000061015), remains an exposed-fold, one-paired-seed observation; this role did not rerun that grade. Public H230/C1/incumbent scores are unaffected. Do not relabel all E513/E514/E518 decisions together: their exact ancestry and actual contrasts differ; E535's separately recorded forensics and original receipts govern those cases.

## Confirmed choices and confounds that are not score bugs

**AMP scheduler/update clock.** In the actual H230 pinned trainer, lines 2310–2318 call `scaler.step(opt)`, `scaler.update()`, `sched.step()` and increment `step_count` unconditionally. EMA/SWA updates below use that attempted-step counter; `training_clocks` explicitly calls the EMA clock “attempted updates including AMP skips.” Directly extracted delivered provenance reports:

| Actual run | Attempts | AMP skips | Completed optimizer updates |
|---|---:|---:|---:|
| H230 family-on S3 | 14,920 | 6 | 14,914 |
| H230 matched family-off S3 | 14,920 | 6 | 14,914 |
| E521 raw GRU384 S3 | 29,840 | 15 | 29,825 |
| E521 raw GRU384 S4 | 29,840 | 15 | 29,825 |
| C1v2 S3 | 14,920 | 6 | 14,914 |
| C1v2 S4 | 14,920 | 7 | 14,913 |

The executed H230 source proves that its skipped attempt advances scheduler time without a parameter update. The other table rows establish delivered skip/counter totals and declare the attempted-update EMA clock in provenance; their older scheduler executables were not separately traced in this bounded check. The H230 schedule difference is not proof of invalid fitting, nor evidence of a gain from changing it. Its paired arms have equal aggregate skips. Keep faithful reference diagnostics and current frozen recipes unchanged. A future completed-update schedule is a separately declared treatment requiring an appropriate matched comparison. Current V5's zero-skip profile admission policy was preserved and adversarially tested; a universal claim that any isolated scaler backoff invalidates every scientific fit is unsupported.

**H232 versus E521 is not an isolated feature ablation.** H232 B8 and E521 B16 give 1,492 versus 746 outer batches per epoch for 11,940 fit sequences. At four epochs this changes optimizer attempts and scheduler trajectory alongside the input view and numeric policy. It remains useful teacher production. An attribution to full features requires a matched batch/schedule/numeric control; changing V5's scientific recipe to remove the confound would exceed the minimal admission repair.

**Targets, masks and denominators.** Twenty independently generated heavy-tailed label/prediction cases, with NaNs confined to excluded rows, matched a separately centered float64 weighted-correlation implementation to maximum error `2.22e-16`. The grouped-moment candidate point score matched the same oracle to `2.22e-16`; this is not a test of bootstrap coverage. A tiny loss case with selected labels beyond ±2, fractional valid weights and an excluded row matched clipped-label wMSE (`3.337777853`) and raw family-target loss (`36.226661682`) with original clipped-label magnitude weights and the explicit four-family denominator. No denominator or official clipping defect was found in these tested functions. Their exact current module hashes are preserved in the receipt; this limited negative result does not certify every loss branch or every historical trainer.

Reference source coefficients whose sum is below one, raw family targets, chunk-wise loss normalization and attempted-step scheduling are distinguishable recipe choices. They should not be silently “fixed” as admission bugs. Causal serving/native equivalence, all teacher-cache group/epoch mappings and broader historical rejection decisions did not receive exhaustive numerical validation here; no global correctness claim is made.

## V5 frozen review and transport handoff

PASS receipt: [H232_V5_INDEPENDENT_REPLAY.json](H232_V5_INDEPENDENT_REPLAY.json), SHA256 `52e3ab25c1fcaddd3d980fa21463ee597fa2656d8989391584de577d5c67512f`. Independent reviewer script: [review_h232_v5.py](review_h232_v5.py).

| Independently inspected artifact | SHA256 |
|---|---|
| V5 worker | `557bad7673af3261c792a27d5b7897ff9af0baae9df5e170c5b1b18f9987cb83` |
| V5 plan | `3b25c00d5e59f0c0686213a8f7f03cbd17e54efbac7c9448d2c261ce10703542` |
| Unchanged trainer archive | `afde296e1c5e2f0c47a6fb352c2dea8e0a3df75900f53c26f103b3921df1f565` |
| Original pending-review wrapper | `8f936379e7680c518a35c2c97d08e515c4f2fa23fb1db31f23d3b11b77b8468d` |
| Original pending-review embedded ZIP | `63a1aeeaf0ffb8decc4ba46bc9fb560a447dc9952251781d850a7a645b3098c8` |

Nine executed cases: real V4 shape/trace/resource evidence with separately reproduced scoped path evidence passes; width 160, missing raw path evidence, zero profile path lookups, auxiliary miss without fallback, changed raw stat identity, one AMP skip under the frozen policy, and changed loader byte budget reject; one raw original-verifier fallback with correct accounting passes. The source feature probe and 73,120,000-byte loader budget were checked. Profile-window raw deltas are captured immediately before `module.main()` at V5 lines 2873–2877 and after it in `finally` at 2986–2994; the gate caller uses that delta at 3141. Initial identity metrics remain separately recorded.

Transport inspection decoded the exact built wrapper payload, verified all eleven embedded members against their packet files, verified all 54 trainer archive members against the source manifest, checked private/offline/GPU metadata and the two permitted private dataset sources, and compared scientific profile argv with V4 after normalizing only operational run-name/output changes. Scientific archive, targets, batch/chunk, epochs, numeric policy and profile/fit settings were unchanged. No actual-device profile or fit was executed by this reviewer.

Root subsequently issued a separate final approval and rebuilt the transport with changes limited to `external_review.json` and `packet_meta.json`; its [root approval receipt](../continuous_execution_E529/h232_conditional_v5/H232_V5_ROOT_FINAL_APPROVAL_20261008_R1.json) records unchanged scientific member hashes and new wrapper `29fdab7e26f64e49e2ccff2677d27e56ff24b36b047e9d9a9e31bf7a69f75f7f` / bundle `c8d5ef9b0fa9b21378303a1a8dcd82c96e4c4ef504ec1222c735f1dd8ed46fc4`. Those reseal and provider decisions are root-owned. This review's original pre-approval receipt is preserved, not overwritten to imply that the reviewer approved provider spending or a successful fit.

## Executed reproduction commands

From this isolated worktree, each script below was executed with `--main-root /home/vlad/development/wunder/wnn_connectome_starterpack` and the corresponding local JSON output. Use the resource wrapper around each invocation:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
research/campaign_2026-10-06/bandit_continuation/code_experiment_bug_review_E538/reproduce_h232_v4.py \
--main-root /home/vlad/development/wunder/wnn_connectome_starterpack \
--output research/campaign_2026-10-06/bandit_continuation/code_experiment_bug_review_E538/H232_V4_REPRODUCTION.json
```

Substitute `reproduce_training_invariants.py` / `TRAINING_INVARIANTS_REPRODUCTION.json` for the global-state, numerical, membership and delivered-update-counter checks. Substitute `review_h232_v5.py` / a **new** output path for a new V5 packet inspection; do not overwrite the frozen PASS receipt after root's metadata reseal. The original review intentionally bound the pending-review transport; a rerun against a resealed wrapper must identify the new transport rather than silently reuse its old hash assertions.

All three scripts completed their substantive reproductions successfully. Ruff formatting/checks are scoped to these reviewer-owned scripts. No production regression tests were installed by this role; their owners can reuse the explicit invariants above when implementing fixes. Root owns repaired execution, terminal checkpoint collection, GPU export and independent grading. Healthy H217 work should continue; this review establishes no reason to terminate it.
