# Imported sources

- [Verified primary-source scope for the October21 cycle](../../../specs/plans/2026-10-21-next-local-kaggle-cycle.md#primary-source-scope) — SWAv3, CPCv2 and test-time-trainingv3 abstracts checked through arXiv; separates AdamW averaging and52-summary Huber adaptations from original methods. TTT remains deferred; no dataset-specific evidence or imported assets.

- [October21 user-reported public receipt](../../campaign_2026-10-21/next_cycle_r1/PUBLIC_RECEIPT_R1.json) — latest package 0.66667/53m47s versus TL3 0.66832/39m12s. Public transfer counterevidence supplied in accepted plan; exact ZIP hash unresolved, not independently queried or clean CV.


External writeups and literature provenance; external results do not establish local model gains.

## Current navigation

- [SWA primary paper](https://arxiv.org/abs/1803.05407) and [model soups primary paper](https://arxiv.org/abs/2203.05482) — abstracts inspected for the [zero-fit WMSE weight-mean proposal](../../campaign_2026-10-20/submission_delivery_r1/wmse_weight_mean_r1/README.md). They motivate averaging compatible checkpoints without adding inference models. Original cyclic/constant SGD and large-model fine-tuning settings differ from our two-epoch AdamW continuation; no local gain follows from these papers.

- [NLL variance-gradient weighting](NLL_GRADIENT_WEIGHTING_20261021_R1.md) — primary ICLR2022 beta-NLL paper and pinned author implementation, connected to measured strong clipping in the current GRU fit. Derives how a detached variance factor changes our bounded mean gradients; conditional diagnostic only, with no implementation or admitted fit.

- [Teacher-scaling source assessment](../ideas/TEACHER_QUALITY_SCALING_20261019_R1.md#sources-and-what-transfers) — primary abstracts for Chinchilla, repeated-data scaling, two time-series scaling studies and muP; separates useful study designs from inapplicable token ratios and unverified Muon transfer. No imported weights/data or local gain.

- [Large B300 teacher primary-source register](TRANSFORMER_B300_SCALE_SOURCES_20261019_R2.md) — actual October19 inspection of the user's2606.30634 gradient-delay paper and later2602.23784 TradeFM event-modeling paper, NVIDIA B300 hardware, Muon source/API, FP32 attention/DDP, numeric encoders, WSD, dynamics pretraining and teacher/student limits. Records the upstream Muon BF16 cast and the gap between external LLM results and this unrun time-series proposal; no imported data or weights.

- [October17 no-fit source register](../../campaign_2026-10-17/research_reset_r1/sources/README.md) — 32 primary works, independent R1 searches and live official-rule checks; inspection depth and transfer limits are explicit, no imported external market data or measured model gain.
- [KalmanNet-inspired innovation-gain GRU source assessment](KALMANNET_INNOVATION_GAIN_GRU_20261018_R1.md) — summarizes three primary learned-filtering papers and limits their transfer to this anonymous time-series task. They motivate a causal latent gain operator; they provide no local score, calibrated covariance claim, or online-weight-learning evidence.

- [Repository audit](../reviews/REPOSITORY_AUDIT_20261017.md) — existing project evidence, separate from outside proposals.
- [Storage and source map](../operations/STORAGE_MAP.md) — source histories and physical artifact locations.

For active status, start with [current state](../../../specs/current-state.md).
The dated entries below retain their original descriptions and links. Their
“pending”, “active” and authorization statements reflect their recording time;
use a later result receipt or the current audit to establish final status.

<details>
<summary>Dated index preserved before the October 17 maintenance</summary>

This domain holds external writeups and source material that informed research. These files report methods from other competitions or conversations; they are not results from our own fits and must not be presented as evidence that a method transfers to this task.

## Literature searches and source provenance

- [Autoregressive Transformer / GRU distillation source assessment and B300 recipe](../ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md#source-records), checked October17, links primary DeepAR, scheduled sampling, Transformer building blocks, AdamW, Muon, WSD, PyTorch Distributed, distillation/teacher-assistant and data-constrained scaling papers, plus official DDP and FlashAttention references. Actual arxiv-search CLI queries and their limited relevance are recorded. Lagged hidden future-price labels are excluded as inputs. AR scan and weighted distributed-loss equations are local derivations; the quality-first ~207M/~682M/~1.65B scale portfolio, long training, context and student-transfer routes are proposals without execution evidence. Related four H-TRANSFORMER-* hypotheses are linked in the note.

- Transformer-teacher architecture references checked October17,2026: [RoFormer / RoPE](https://arxiv.org/abs/2104.09864) supplies a relative-position mechanism; [Transformer-XL](https://arxiv.org/abs/1901.02860) motivates an optional segment-memory arm; [PatchTST](https://arxiv.org/abs/2211.14730) and [iTransformer](https://arxiv.org/abs/2310.06625) establish their patch/channel and variate-token designs, respectively; [FlashAttention](https://arxiv.org/abs/2205.14135) describes IO-aware exact attention, without certifying a particular B300 software/precision path. These primary papers informed the user's architecture-selection discussion for eight B300 GPUs and eventual GRU distillation. The proposed baseline is a causal temporal decoder with numeric row embeddings and per-row regression; grouped input embeddings, longer context and segment memory are distinct comparisons. Published forecasting/language results do not establish local gains or successful Transformer-to-GRU transfer. No fit, hardware profile or numerical promotion claim accompanies this source entry.

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

- [B300 skill search and project adaptation](B300_AGENT_SKILLS_REVIEW_20261017_R1.md) records the inspected NVIDIA/Han Lab profiling skills, official training catalogue and external B300 field report. Motivates a project-authored skill with source links; no bundle installation, hardware profile or throughput claim.

</details>

## Current sequence-moment investigation

- [RevIN official implementation and its causal-use limitation](../validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md) records the ICLR2022authors' [repository](https://github.com/ts-kim/RevIN) and inspected RevIN.py reduction over the supplied temporal axis. Relevant to mean/variance normalization: a whole-current-sequence call would leak future observations in this competition; the local expanding-prefix treatment is an adaptation, with no imported performance claim.

- [User-supplied previous competition discussion and reflection](../../campaign_2026-10-18/online_features_r1/PREVIOUS_COMPETITION_REFLECTION.md): December2025–March2026 participant reports of SWA, featureextractors, recurrent/attention hybrids, boosting and diversity, contrasted with current callback/metric/data differences and measuredgeometry results. No source URL/figures provided; anecdotes and causal explanations are unverified, numerical gains not transferable.

- [Seed-5 production payload manifest](../../campaign_2026-10-06/bandit_continuation/reviews/maxim_replication_20261007/candidate-manifest.json) and [preserved final-student recipe audit](../../campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_MAXIM_RECIPE_AUDIT_V1_20261008.md): the payload hash matches TL3's replaced student, and the source command includes supplied validation rows with 70% teacher supervision and family targets. Reused in the [October21 exposure receipt](../../campaign_2026-10-21/next_cycle_r1/TL3_STUDENT_EXPOSURE_RECOVERY_R1.json); exact recursive teacher/initializer hashes and execution group receipts remain incomplete.
