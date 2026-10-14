# Imported sources

This domain holds external writeups and source material that informed research. These files report methods from other competitions or conversations; they are not results from our own fits and must not be presented as evidence that a method transfers to this task.

## Literature searches and source provenance

- [Audited teammate methods and evidence](../../campaign_2026-10-13/team_models_r1/TEAM_METHODS_AND_EVIDENCE_R1.md) pins October6 branch833aaf54 and separates family distillation, unnormalized0.8125 teacher scale, head-FT and causal regime probes. Historical reports support bounded recipe follow-ups, but current final-model CV arrays/FTT ancestry are missing; supplied probe weights make inference replayable despite source-branch payload gaps.

- [AMD EPYC9224 architecture sources and runtime application](../../campaign_2026-10-13/team_models_r1/EPYC_RUNTIME_REFLECTION_R1.md#source-records) records primary AMD product and Zen4 white-paper references checked October13,2026. They establish VNNI support, 256-bit internal paths and cache structure; they do not establish our hosted runtime or a kernel winner. Related independent ELF audit and H-TEAM-NATIVE-20261013 are linked there.

- [Official team ZIP/source comparison](../../campaign_2026-10-13/team_models_r1/README.md) — October13 user-authorized reuse of teammate code and weights under ADR0035. Exact inventories cover pair_ftt2, xteam_h4, e6a and class probes, with source/weight deduplication and independent ancestry review. The later FTT source/CV remains unlocated; archive labels alone do not establish held-out evidence. Original weight-bearing files stay in ignored physical storage.

- [Hugging Face archive source/API references](../models/huggingface/README.md) — links the official HF upload and CLI documentation used to implement authenticated private-model transfers and checkpoint lookup; operational reference rather than model-quality evidence.
- [October 12 arXiv search receipt](../../campaign_2026-10-12/strategy_review_r1/LITERATURE_SEARCH.json) — records queries, primary paper links and the seeded passage selection behind the [public and future private strategy](../reviews/CHALLENGE2_PUBLIC_PRIVATE_STRATEGY_20261012.md). Sources motivate causal representations, memory, distillation and robustness experiments; none establishes a measured improvement in this competition.

## Competition writeups and transcripts

- [Rogii first-place writeup](competition-writeups/rogii_top1.txt)
- [Santander competition transcript](competition-writeups/santander_transcript_english_translation.md)
- [BNP competition transcript](competition-writeups/semenov.txt)
- [Discord feature and ensemble notes](competition-writeups/discord_tricks.txt)

Keep provenance and source links with each imported item. Generated idea cards derived from these sources belong in [ideas](../ideas/README.md); critical reviews belong in [reviews](../reviews/README.md). Imported writeups live in this canonical source tree.
[QAT primary-source components](../../campaign_2026-10-13/qat_renewable_r1/SOURCES.md) records He et al. recurrent gate/interlink quantization, Fasoli et al. end-to-end four-bit RNN-T QAT and Esser et al. learned step sizes, verified against primary arXiv records on2026-10-13. These motivate H-QAT-W4/A8/LSQ-20261013; published task/hardware results are not local CPU or model-quality evidence.

- [CPU student primary sources and theory](../../campaign_2026-10-14/cpu_student_frontier_r1/THEORY_AND_DISCRIMINATING_TESTS_R1.md) links session-verified AMD Zen4, depth/width, T-MAC, contextual RNN, minGRU and low-rank recurrence papers. It distinguishes their claims from our derived matrix/cache/initializer arithmetic and proposes discriminating tests. H-CPU-POLICY-20261014/H-STUDENT-LADDER-20261014/H-STATE-INIT-20261014; hypotheses, no new measured performance.
