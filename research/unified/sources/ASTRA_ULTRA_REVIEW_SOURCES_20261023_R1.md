---
kind: review
status: complete
created_utc: 2026-10-23T16:42:37Z
updated_utc: 2026-10-23T17:28:29Z
hypothesis_ids: []
experiment_ids: []
evidence_paths: []
---

# Primary literature for the ten-reviewer research audit

## Scope

This is the source register for the user-requested [independent Astra
review](../reviews/astra_ultra_audit_20261023_r1/README.md). It records external
claims and transferable components, not reproduced Challenge 2 results.
Each reviewer selects sources independently and cites them in its own report.
This register is not supplied as a shared conclusion.

The arxiv-search skill is applied through web search and primary arXiv pages.
No local helper, package installation, Python import or downloaded code execution
is used during the experiment stop. Abstract and full-text checks are distinguished.

## Controller checks

| Source | Inspected material | Useful claim or component | Transfer limit |
|---|---|---|---|
| Cawley and Talbot, [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html), JMLR 2010 | Primary journal abstract, retrieved October 23 | Finite-sample selection criteria can themselves be overfit; selection variability can be comparable with algorithm differences. | Does not quantify this project's bias or invalidate all exposed-fold comparisons. Motivates recording selection and separating conditional uncertainty from transfer. |
| Muennighoff et al., [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264v5), 2023/2025 | Primary arXiv v5 abstract | Repeated-data compute has diminishing returns; unique data, repetition and parameter count are distinct scaling axes. | Language-model findings are not a numeric epoch, token-to-row or model-size rule for correlated market sequences. Already cited by the October 19 proposal; not a new project idea. |
| Zeng et al., [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504v3), 2022 | Primary arXiv v3 abstract | A simple linear comparator can expose weaknesses in complex forecasting benchmarks. | The datasets, task and models differ. This does not establish that attention or large teachers cannot work here. The broad architectural explanation is the authors' interpretation, not a theorem accepted here. |

Searches included the titles above, model-selection bias, data-constrained
scaling and Transformer/linear time-series comparisons. Secondary search hits
were not used as evidence. No quality gain, causal explanation, scaling exponent
or public forecast follows from these citations.

## Reviewer-selected sources

The individual reports are the canonical claim–source records for their searches.
The final synthesis distinguishes useful components from weak analogies;
a cited mechanism is not a validated solution to this competition.

| Primary source | Reviewer inspection and usable component | Boundary of the claim |
|---|---|---|
| [GRU-D](https://arxiv.org/abs/1606.01865); [published methods](https://www.nature.com/articles/s41598-018-24271-9) | [02](../reviews/astra_ultra_audit_20261023_r1/02_representation_and_hidden_structure.md) inspected the abstract; [03](../reviews/astra_ultra_audit_20261023_r1/03_memory_and_time.md) inspected methods/limitations. Separate value from observation timing. | An unchanged observed field is not a missing measurement. No imputation or decay gain is established here. |
| [Can recurrent neural networks warp time?](https://arxiv.org/abs/1804.11188) | 02 inspected the abstract; 03 inspected the introduction and equations 3–7. Gates motivate considering time scales explicitly. | Does not imply a fitted row-time GRU is invariant to skipped updates, or that clock invariance benefits the target. |
| [On Embeddings for Numerical Features in Tabular Deep Learning](https://arxiv.org/html/2203.05556v4) | 02 inspected Sections 3.1–3.2 and the abstract. Piecewise-linear encodings supply a nonlinear numeric comparator. | Tabular gains do not validate atom identity, anonymous feature semantics or sequential transfer. |
| [Deep Sets](https://papers.nips.cc/paper/6931-deep-sets.pdf) | 02 inspected Section 2 and its theorem qualifications. Restricted shared aggregation offers a testable symmetry prior. | The representation theorem does not establish that these fixed feature identities are exchangeable. |
| [Unbiasing Truncated Backpropagation Through Time](https://arxiv.org/pdf/1705.08209) | 03 inspected Sections 1–3 and equations 8–9. Distinguish carried state from truncated credit. | Does not establish that ARTBP is useful, stable or affordable here; it is not an admitted implementation. |
| [Skip RNN](https://arxiv.org/pdf/1708.06834) | 03 inspected model equations and the update-budget objective. Learned update/copy decisions are a distinct mechanism. | A novelty threshold retrofitted onto a row-trained checkpoint is not a reproduction of learned skipping; gate overhead and downstream effects remain unmeasured. |
| [On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks](https://arxiv.org/html/2203.09168v2), with [pinned author implementation](https://github.com/martius-lab/beta-nll/blob/669c9f251eb41464c1ec8b43751c7c742dd75cf9/README.md#quick-implementation-of-beta-nll) | [04](../reviews/astra_ultra_audit_20261023_r1/04_objectives_and_metric.md) inspected Sections 2–4 and the detached variance-factor implementation. Distinguishes noise weighting from neglect of hard regions. | This project's weighted, clipped, teacher-heavy targets and auxiliaries differ. β=1 matches the direct mean-output derivative to WMSE but retains shared variance-gradient/clipping interactions; no local improvement follows. |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | [05](../reviews/astra_ultra_audit_20261023_r1/05_teacher_scaling_and_capacity.md) inspected the abstract. Vary model scale and training exposure together. | No language-token ratio or exponent is imported into this task. |
| [Adaptively Truncating Backpropagation Through Time to Control Gradient Bias](https://proceedings.mlr.press/v115/aicher20a.html) | 05 inspected the primary proceedings abstract. Truncation bias is distinct from forward memory. | Its decay assumptions and adaptive procedure are not verified here; supports a bounded fixed-credit contrast. |
| [Transformer-XL](https://arxiv.org/html/1901.02860v3) | 05 inspected Section 3.2 and the stopped-gradient memory equation. Forward access and historical gradient access are distinct. | Compressed or detached memory does not automatically solve a credit bottleneck or improve these targets. |
| [Scaling Law for Time Series Forecasting](https://arxiv.org/html/2405.15124v4) | 05 inspected Sections 4.2–5.1 and limitations. Context and data granularity are separate axes. | Different forecasting settings and assumptions prevent inferring this project's optimal context or size. |
| [Deep Double Descent](https://arxiv.org/html/1912.02292v1) | 05 inspected the introduction/discussion. Size and training-duration rankings can be nonmonotonic. | Counters categorical parameter-count exclusions; provides no evidence that this dataset occupies that regime. |
| [Managing Diversity in Regression Ensembles](https://jmlr.csail.mit.edu/papers/v6/brown05a.html) | [06](../reviews/astra_ultra_audit_20261023_r1/06_ensembles_and_distillation.md) inspected the journal abstract. Accuracy and covariance jointly determine mixture value. | Squared-error results do not directly optimize clipped weighted Pearson. |
| [On the Efficacy of Knowledge Distillation](https://arxiv.org/abs/1910.01348) | 06 inspected the abstract. More accurate or larger teachers need not teach a fixed student better. | Does not establish a capacity mismatch here or justify stopping a large teacher. |
| [Knowledge Distillation as Semiparametric Inference](https://arxiv.org/html/2104.09732v1) | 06 inspected the problem definition, overfitting discussion and Section 4.2. Teacher data reuse motivates explicit exclusion/cross-fitting. | Borrowed as a residual-learning design principle; classification/i.i.d. assumptions do not prove improvement in dependent weighted regression. |
| [Net2Net](https://arxiv.org/abs/1511.05641) | [07](../reviews/astra_ultra_audit_20261023_r1/07_causal_identification_and_controls.md) inspected the abstract. Preserve the initial function when extending a model. | Equal initial predictions do not imply equal optimization trajectories. |
| [Neural Tangent Kernel](https://arxiv.org/abs/1806.07572) | 07 inspected the primary abstract/search record. Parameter derivatives distinguish function equality from learning geometry. | The report's finite-head algebra is direct; no infinite-width prediction is claimed for GRUs, AdamW or these heads. |
| [Test-Time Training](https://arxiv.org/abs/1909.13231) | 07 inspected the abstract. Observable inputs can support an auxiliary adaptation signal. | Image results do not establish target alignment, nonstationarity, or gains here; future labels remain unavailable. |
| [Covariate Shift Adaptation by Importance Weighted Cross Validation](https://jmlr.org/papers/v8/sugiyama07a.html) | [08](../reviews/astra_ultra_audit_20261023_r1/08_transfer_and_information_limits.md) and root inspected the primary abstract. Input-distribution correction has explicit conditional-law assumptions. | A hidden scoring mask may depend on outcomes unavailable in causal history; estimating selection propensity cannot by itself identify the desired conditional target law. |
| [Underspecification Presents Challenges for Credibility in Modern Machine Learning](https://jmlr.org/papers/v23/20-1335.html) | 08 and root inspected the primary abstract. Similar validation performance can coexist with different deployment behavior. | Does not establish a specific shortcut or future shift in this project; motivates fixed stress tests, not an automatic worst-domain veto. |
| [MIT Feedback Systems recitation 17](https://ocw.mit.edu/courses/6-302-feedback-systems-spring-2007/fb1e56673aeec2190270f6a45c7b6765_6_302rec17.pdf) and [Caltech lead-compensator explanation](https://murray.cds.caltech.edu/Why_do_we_need_pole_in_the_lead_compensator%3F) | [10](../reviews/astra_ultra_audit_20261023_r1/10_unrestricted_red_team.md) inspected MIT pages 3–4 and the full Caltech explanation. Bounded lead/lag operators suggest a causal response-speed discriminator. | Prediction is not closed-loop control. A failed smoothing model does not prove beneficial sharpening; the proposed finite-difference test is not a reproduction of a continuous-time controller. |
| [Imani and White, Improving Regression Performance with Distributional Losses](https://proceedings.mlr.press/v80/imani18a/imani18a.pdf) | 10 inspected the abstract and Section 3, including the local gradient bound. Soft histograms offer an optimization hypothesis. | They do not improve the Bayes information set or establish local gain; prior E421 evidence argues against an unmotivated repeat. |
| [Bellemare et al., A Distributional Perspective on Reinforcement Learning](https://proceedings.mlr.press/v70/bellemare17a.html) | 10 inspected the abstract. Distributional value learning supplies a distant analogy. | Bellman dynamics are absent here; this is context for a deferred idea, not direct evidence. |

## Seeded local-library draw

Reviewer 10 used the four sorted authorized text names `discord_tricks.txt`,
`rogii_top1.txt`, `santander_transcript_english_translation.md`, `semenov.txt`,
under `research/unified/sources/competition-writeups/`. Seed **2026102310** and
LCG32 `s=(1664525*s+1013904223) mod 2^32` produce 446324557 and 1724086344.
Modulo indexing selects [Rogii](competition-writeups/rogii_top1.txt), then lines
49–64 in its 528-line text. Root independently checked the arithmetic and read
that short passage. Context lines 40–84 were inspected by the reviewer afterward.

The selected passage lists quadratic fitting coefficients and residual RMSE
among geological features. The transfer is a hypothesis about smooth versus
irregular prediction corrections, not geosteering coordinates, future-aware
smoothing, task labels or reported scores. The exact small quote and complete
sampling record remain in report 10. No local extraction or code import ran.

The reviewer sought Friedman's original boosting paper but encountered access
failures; no detailed claim relies on it. Failed older Caltech links were replaced
by the accessible primary teaching materials above. Reviewer 09 required no new
external literature. This selective search is not a systematic literature review.
