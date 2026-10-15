# Imported sources

This domain holds external writeups and source material that informed research. These files report methods from other competitions or conversations; they are not results from our own fits and must not be presented as evidence that a method transfers to this task.

## Literature searches and source provenance

- [Research-cycle source verification and local application](../methods/RESEARCH_LOOP_RESET_20261015_R1.md#verified-external-basis-and-limits) checks primary AIRA-dojo2507.02554, Amazon2609.30541 and JANUS2610.13194 onOctober15. Operator quality/proxy selection, production delivery failures and information-aware allocation motivate concrete local interventions. The note does not adopt reported benchmark gains as competition evidence or claim every numerical statement in the user's supplied survey was verified.

- [Net2Net transfer source and proposed GRU growth test](../ideas/QUALITY_BOTTLENECK_AND_STUDENT_TRADEOFFS_20261015_R1.md#theory-and-useful-falsifiers) links [Chen, Goodfellow and Shlens](https://arxiv.org/abs/1511.05641), verified October15. Function-preserving expansion motivates transferring a strong parent into wider students. The gate-aware GRU embedding described in the note is our unimplemented proposal, with sequence/reset/gradient falsifiers; the paper does not establish its local quality or native numerical parity. Related H-STUDENT-PARENT-GROWTH-20261015.

- [Distillation quality precedents and their local tests](../ideas/QUALITY_BOTTLENECK_AND_STUDENT_TRADEOFFS_20261015_R1.md#theory-and-useful-falsifiers), checked October15, links primary [Born Again Networks](https://proceedings.mlr.press/v80/furlanello18a.html), [Cho–Hariharan](https://arxiv.org/abs/1910.01348) and [Beyer et al.](https://arxiv.org/abs/2106.05237). They motivate testing student benefit, capacity mismatch and training duration rather than assuming a teacher-score ceiling. Published vision/language findings are hypotheses for this recurrent regression task, not measured local gains; related H-KD-CONTINUATION-VALUE-20261015. CVF HTML returned403 on recheck; primary arXiv/PMLR records and CVF PDF search records were available.

- [Audited teammate methods and evidence](../../campaign_2026-10-13/team_models_r1/TEAM_METHODS_AND_EVIDENCE_R1.md) pins October6 branch833aaf54 and separates family distillation, unnormalized0.8125 teacher scale, head-FT and causal regime probes. Historical reports support bounded recipe follow-ups, but current final-model CV arrays/FTT ancestry are missing; supplied probe weights make inference replayable despite source-branch payload gaps.

- [AMD EPYC9224 architecture sources and runtime application](../../campaign_2026-10-13/team_models_r1/EPYC_RUNTIME_REFLECTION_R1.md#source-records) records primary AMD product and Zen4 white-paper references checked October13,2026. They establish VNNI support, 256-bit internal paths and cache structure; they do not establish our hosted runtime or a kernel winner. Related independent ELF audit and H-TEAM-NATIVE-20261013 are linked there.

- [Official team ZIP/source comparison](../../campaign_2026-10-13/team_models_r1/README.md) — October13 user-authorized reuse of teammate code and weights under ADR0035. Exact inventories cover pair_ftt2, xteam_h4, e6a and class probes, with source/weight deduplication and independent ancestry review. The later FTT source/CV remains unlocated; archive labels alone do not establish held-out evidence. Original weight-bearing files stay in ignored physical storage.

- [Hugging Face archive source/API references](../models/huggingface/README.md) — links the official HF upload and CLI documentation used to implement authenticated private-model transfers and checkpoint lookup; operational reference rather than model-quality evidence.
- [October 12 arXiv search receipt](../../campaign_2026-10-12/strategy_review_r1/LITERATURE_SEARCH.json) — records queries, primary paper links and the seeded passage selection behind the [public and future private strategy](../reviews/CHALLENGE2_PUBLIC_PRIVATE_STRATEGY_20261012.md). Sources motivate causal representations, memory, distillation and robustness experiments; none establishes a measured improvement in this competition.

## Competition writeups and transcripts

- [Rogii first-place writeup](competition-writeups/rogii_top1.txt) — imported competition evidence whose model table shows a particle-filter feature was not universally useful and ensemble weights varied with context; used only as a hypothesis about conditional component value, not as a transferable score or method result.
- [Santander competition transcript](competition-writeups/santander_transcript_english_translation.md)
- [BNP competition transcript](competition-writeups/semenov.txt)
- [Discord feature and ensemble notes](competition-writeups/discord_tricks.txt)

Keep provenance and source links with each imported item. Generated idea cards derived from these sources belong in [ideas](../ideas/README.md); critical reviews belong in [reviews](../reviews/README.md). Imported writeups live in this canonical source tree.
[QAT primary-source components](../../campaign_2026-10-13/qat_renewable_r1/SOURCES.md) records He et al. recurrent gate/interlink quantization, Fasoli et al. end-to-end four-bit RNN-T QAT and Esser et al. learned step sizes, verified against primary arXiv records on2026-10-13. These motivate H-QAT-W4/A8/LSQ-20261013; published task/hardware results are not local CPU or model-quality evidence.

- [CPU student primary sources and theory](../../campaign_2026-10-14/cpu_student_frontier_r1/THEORY_AND_DISCRIMINATING_TESTS_R1.md) links session-verified AMD Zen4, depth/width, T-MAC, contextual RNN, minGRU and low-rank recurrence papers. It distinguishes their claims from our derived matrix/cache/initializer arithmetic and proposes discriminating tests. H-CPU-POLICY-20261014/H-STUDENT-LADDER-20261014/H-STATE-INIT-20261014; hypotheses, no new measured performance.

- [Distillation and minGRU source assessment](../../campaign_2026-10-14/cpu_student_frontier_r1/DISTILLATION_AND_RECURRENCE_THEORY_R1.md) records Stanton et al.2021 and Mirzadeh et al.2020 abstract claims about imitation/optimization/capacity, plus direct inspection of Feng et al.2024 minGRU equations and implementation appendix. It separates those external results from our own algebra and competition hypotheses; none establishes a local score gain.

- [minGRU Appendix B implementation use](../../campaign_2026-10-14/cpu_student_frontier_r1/MINGRU_REFERENCE_R1.md): primary paper v3 was re-opened for candidate/gate/state semantics. The local implementation uses direct affine composition and exact initial state, with separate gradient/causality tests. Paper benchmark claims do not establish competition performance or a CPU speedup.

- [Recurrent timescales and initializer gradients](../../campaign_2026-10-14/cpu_student_frontier_r1/RECURRENT_TIMESCALES_AND_INITIALIZER_GRADIENTS_R1.md): chrono initialization (Tallec/Ollivier2018), minLSTM initialization analysis (Feng et al.2024), truncation-bias literature and official checkpointing API inform a distinct minGRU bias arm and complete-sequence initializer training. Six scalar checks and five synthetic gradient tests pass; no fitted-model benefit. H-MINGRU-TIMESCALES-20261014 / H-STATE-INIT-20261014.

- [Directed event-time literature and seeded Rogii passage](../ideas/EVENT_STATE_AND_METRIC_OBJECTIVE_REFRAME_20261015_R1.md#source-records) links primary Engle–Russell ACD and Eisler et al. order-event-impact sources, plus the seeded random writeup passage. These support testing event duration and context-conditional value; none establishes gains here. The random Rogii table reports conditional particle-feature weights and a strong weighted ensemble on its own CV, not transferable Challenge2 scores.

- [Initializer-preservation sources](INITIALIZER_PRESERVATION_SOURCES_20261015_R1.md) verifies primary LwF2016/2017 and L2-SP2018 sources. Their fixed-output and parameter-reference mechanisms motivate an untrained regression control; vision/task-transfer results are not claimed for this competition. H-FROZEN-PARENT-ANCHOR-20261015.
