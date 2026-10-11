# Distillation literature survey for the WNN connectome student (2026-10-11)

Scope: read-only survey for our setting (causal per-row regression, h256 GRU student, h384/ensemble teachers, student gain +0.0008 from teacher signal vs +0.003 from inference-time blending). See `ideation/workspace/DISTILL_FACTS.md`.

Provenance rule: every entry below was opened (arXiv abstract page, arXiv HTML, PMC full text, or search-result snippet where marked). Statements marked **[abstract only]** come from the abstract page; **[snippet]** from a search-result summary that I did not open in full; **[unverified]** are my own inferences or claims I could not confirm. No claim here is a measured result on our data.

## Diagnosis first: why the teacher signal is small for us

Three literature-backed reasons, in order of how well they fit our numbers:

1. Our targets are static and the student barely moves. We use cached teacher outputs (a fixed, single view), 3 epochs, lr 1e-4, warm start from a 0.666-class parent. Beyer et al. report that fixed precomputed teacher targets were much worse than teacher and student seeing the same augmented view, and that fixed targets overfit after ~600 epochs while function matching kept improving to 9600 epochs (see 4a). Our 3 epochs is the opposite end of that axis.
2. The student starts at teacher quality, so the gradient of `MSE(student, teacher_mean)` is mostly teacher-vs-student disagreement noise. Stanton et al. find students often fail to match even when capable, primarily due to optimization, and that fidelity does not equal generalization (1b).
3. Ensemble gain lives in the *disagreement* between members (members share 96-99.9% residual in our EDA, so the usable signal is the 0.1-4% diverse part). A mean target at 0.7 weight dilutes it. Matching individual teachers (multi-head) or the teacher spread may preserve it better (2, 3b).

## 1. Classic KD, capacity gap, teacher assistants

**1a. Hinton, Vinyals, Dean 2015, Distilling the Knowledge in a Neural Network.** https://arxiv.org/abs/1503.02531 (opened, abstract). Compress an ensemble into a single model by training on the ensemble's soft outputs; also proposes specialist models.
- Apply to us: this is our current recipe in principle; the abstract gives no schedule detail so it cannot answer the epoch/LR question [abstract only].
- Cheapest test: none beyond the baseline; use as the control.

**1b. Stanton, Izmailov, Kirichenko, Alemi, Wilson 2021, Does Knowledge Distillation Really Work?** https://arxiv.org/abs/2106.05945 (opened, abstract; PDF text could not be extracted). Students often have a surprisingly large predictive discrepancy to the teacher even with enough capacity; they attribute it mainly to optimization, not capacity; distillation dataset matters; more fidelity does not always mean better generalization.
- Apply: our student plateaus at teacher-gap not because the h256 GRU cannot represent the mean, but because 3 epochs at 1e-4 do not optimize the matching loss. Measure fidelity directly (corr of student vs teacher-mean on valid) alongside the metric.
- Cheapest test: train the student for N epochs on teacher loss only (alpha=1.0) and plot valid-fidelity and valid-metric vs epoch. If fidelity climbs while metric stalls, optimization is not the bottleneck; if both climb, go long.

**1c. Cho and Hariharan 2019, On the Efficacy of Knowledge Distillation.** https://arxiv.org/abs/1910.01348 (opened, abstract). More accurate teachers are often not better teachers (capacity mismatch); sequential steps did not help; early-stopping the teacher reduces the mismatch.
- Apply: our h384 muon-all teachers (0.670) gave a *worse* student (0.66836) than h256 teachers (0.66938). This matches the finding. Try earlier-epoch or less-converged h384 checkpoints as teachers, or a mid-size (h320) teacher assistant.
- Cheapest test: reuse existing intermediate checkpoints of the h384 teacher, if saved, as teacher caches; same student recipe.

**1d. Mirzadeh et al. 2019, Teacher Assistant KD.** https://arxiv.org/abs/1902.03393 (opened, abstract). Student quality degrades when the teacher is much larger; insert an intermediate-size assistant.
- Apply: chain h384 -> h320 (or h256 ensemble-average) -> h256. The gap here is small (2x width), so expected effect is small; the more useful analog is a "TA" that is an ensemble-averaged h256 student.
- Cheapest test: use the existing 0.66938 student as a teacher for a second distillation round (it is already a TA-like object) and add the h384 cache as a second teacher.

## 2. Ensemble / multi-teacher distillation, and why students underperform ensembles

**2a. Allen-Zhu and Li 2020, Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning.** https://arxiv.org/abs/2012.09816 (opened, abstract). Under a multi-view data model the ensemble's gain is transferable to a single model trained on the ensemble outputs; self-distillation implicitly combines ensembling and distillation. The abstract does not state sample-size or width conditions [abstract only].
- Apply: theory predicts transfer works when the student sees enough *diverse inputs* where members disagree. Our cached targets are only on the train split rows, which the members were fit on, so member disagreement on train is small (they have memorized it). The gain therefore likely needs *held-out-style* disagreement: OOD/augmented paths (see 4) or teacher outputs from checkpoints that did not train on those rows.
- Cheapest test: measure teacher-member disagreement variance on train rows vs valid rows (read-only EDA on caches). If train disagreement is far below valid disagreement, targets are the issue.

**2b. Furlanello et al. 2018, Born-Again Networks.** https://arxiv.org/abs/1805.04770 (opened, abstract). Same-architecture students beat their teachers on vision and language modeling; CWTM and DKPP objectives.
- Apply: sequential self-distillation chain h256 -> h256 -> h256 with the previous generation(s) averaged. We already have the ingredients: a mean over generations is a free ensemble.
- Cheapest test: generation 2 student from the 0.66938 student plus the original parent as two teachers, from scratch rather than warm-start (born-again re-initializes) vs warm start.

**2c. Lin, Kong, Stich, Jaggi 2020, Ensemble Distillation for Robust Model Fusion in Federated Learning.** https://arxiv.org/abs/2006.07242 (opened, abstract). Server model is trained on *unlabeled data* using the outputs of the client models; works for heterogeneous architectures. (My first guessed ID, 2006.07253, was a different paper and was discarded.)
- Apply: supports using extra unlabeled inputs and heterogeneous teachers; matches the "out-of-sample path distillation" idea in the facts file.

**2d. Zuchniak 2023, Multi-teacher KD as an effective method for compressing ensembles.** https://arxiv.org/abs/2302.07215 (opened, abstract). Single student mimics all teachers at once; reported better than standard training at equal complexity; no numbers vs the full ensemble in the abstract [abstract only].
- Apply: multi-head variant, one output head pair per teacher plus the mean head, shared trunk. Cheap form of keeping disagreement information.

**2e. Nature/PMC, DEGU (Distilling Ensembles for Genomic Uncertainty-aware models).** https://pmc.ncbi.nlm.nih.gov/articles/PMC11601481/ (opened, partial: first 100k of 121k chars). A 10-model same-architecture regression ensemble is distilled into one network that predicts ensemble mean (MSE) plus ensemble SD (extra head, MSE, weight 1). Students "generally approached but did not fully match" the ensemble; distilled performance plateaued near 5 teachers while the ensemble kept improving to ~10. Dynamic distillation (each minibatch augmented and relabeled by the teacher ensemble) gave modest in-distribution gains but better teacher approximation under shift.
- This is the closest regression precedent to our setting. It also confirms the honest expectation: the student recovers only part of the ensemble gain.
- Cheapest test: add an auxiliary head predicting teacher SD (or per-teacher outputs); loss weight 0.1-1.0; read metric of the mean head.

## 3. KD for regression and sequence models; feature / hidden-state matching

**3a. FitNets (Romero et al. 2014).** https://arxiv.org/abs/1412.6550 (opened, abstract). Intermediate-layer hints with a learned regressor from student to teacher feature space.
**3b. Attention transfer (Zagoruyko and Komodakis 2016).** https://arxiv.org/abs/1612.03928 (opened, abstract; CNN-specific, low direct relevance).
**3c. PKD for BERT (Sun et al. 2019).** https://arxiv.org/abs/1908.09355 (opened, abstract). Student learns from several intermediate teacher layers (PKD-Last / PKD-Skip), which helped in NLP.
- Apply to us: teacher GRU hidden state (384) vs student (256) needs a learned linear projection (FitNets-style, discarded at serving, so no CPU cost). Matching the final-layer pre-head vector or the layer-1 GRU output at weight ~0.1 gives the student more gradient information per row than 2 scalar outputs. This addresses "targets carry only 2 numbers per row".
- Cheapest test: cache teacher final-layer hidden states (384-d) for the train rows for one teacher, add a Linear(256 -> 384) probe with MSE at weight 0.1-0.3. Storage: 384 floats per row may be large; use fp16 or a PCA-64 projection.

**3d. Time-series specific: Sequence-level KD (Kim and Rush 2016).** https://arxiv.org/abs/1606.07947 (opened via search snippet only). Train on teacher-generated outputs; mainly about decoding in NMT, limited direct relevance to our non-autoregressive regression [snippet].
**3e. MemKD, memory-discrepancy KD for time series classification.** https://arxiv.org/html/2601.04264 (search snippet only). Says prior forecasting distillation often aligns the last hidden states directly and requires same architectures; matches memory-retention across subsequences [snippet, [unverified] for details].
**3f. Distilling time-series foundation models.** https://arxiv.org/html/2601.12785 (snippet only). Adds a projection module mapping student embeddings into teacher space [snippet].
- Apply: justifies the learned projection in 3c. No source found with a clean recurrent-regression recipe; evidence base here is thin.

**3g. Regression: uncertainty/spread targets.** DEGU (2e) is the verified example. A general "uncertainty propagation in KD" review and an aggregator page (Emergent Mind) were seen only as snippets and some attributions were flagged unverifiable by the search tool: https://www.themoonlight.io/en/review/how-is-uncertainty-propagated-in-knowledge-distillation [snippet, [unverified]].
- Apply: weight each row's teacher loss by inverse teacher-spread (precision weighting) or only distill rows where the ensemble spread is above a quantile. Cheapest test below (E7).

## 4. Data augmentation, unlabeled/OOD data, function matching

**4a. Beyer et al. 2022, A good teacher is patient and consistent.** https://arxiv.org/abs/2106.05237 (opened; details from arXiv HTML v2). Recipe: pure KL to teacher with no hard-label loss; teacher and student receive the identical augmented view (precomputing fixed teacher targets performed much worse); function matching with mixup on both; very long schedules (ImageNet student 81.54% at 1200 epochs, 82.18% at 4800, 82.31% at 9600; fixed-teacher overfit after ~600 epochs); label-only baseline clearly worse and overfit.
- Apply to us: this is the best-grounded argument against our current recipe: (a) cached fixed targets, (b) 3 epochs, (c) 0.3 label loss. For us "same view" means the teacher must be run on-the-fly on the same augmented sequence the student sees. Teacher cost is training-time only and relaxed. Mixup across *rows* is not causal; use sequence-level perturbations that preserve causal structure: random sub-sequence start (state reset), feature noise, feature dropout, time-warp/subsample not valid because labels step at book events [unverified, check against label semantics], cross-stream mixing of whole inputs with convex combos is not obviously valid for GRU state.
- Cheapest test: online teacher forward (h256 teacher ensemble of 2 + h384) on each batch with input Gaussian noise / feature dropout applied identically, alpha=1.0 (no labels) for 10-20 epochs at lr 3e-4 cosine; compare vs 3-epoch cached baseline. Report fidelity.

**4b. MixKD (Liang et al. 2020).** https://arxiv.org/abs/2011.00593 (opened, abstract). Student learns from teacher outputs on interpolated pairs; claims improved generalization especially with limited labeled data.
**4c. Choi et al. 2022, Role of Mixup in KD.** https://arxiv.org/abs/2211.03946 (opened, abstract). Smoothness is the connection between mixup and KD [abstract only].
- Apply: mixing at the level of GRU inputs is questionable (non-linear recurrent state; linear blend of two market streams is not a market stream). A safer analog is to mix at hidden-state/embedding level after the encoder or mix feature channels between two streams with a small lambda (< 0.2). Low confidence.
- Cheapest test: lambda ~ Beta(0.4,0.4) capped to [0.8,1] mix of *encoder outputs* from two aligned sequences; run teacher on the same mixed input. [unverified feasibility]

**4d. Out-of-sample paths (from DEGU, 2e; Lin et al., 2c).** Federated ensemble distillation uses unlabeled data; DEGU's dynamic augmented relabeling improved teacher approximation under shift. No external data are allowed for us, but valid.parquet inputs without labels would be unlabeled data if the fitting-membership rules in CLAUDE.md allow it (they do not for held-out claims; use only fold-appropriate train rows). Augmented/permuted train sequences are permitted.

**4e. Data-free KD.** Not opened; not found with a verified source in this session. Skip.

## 5. Self-distillation, online/mutual learning, co-distillation, weight averaging

**5a. Deep Mutual Learning (Zhang et al. 2017).** https://arxiv.org/abs/1706.00384 (opened, abstract). Student cohort teaches each other; no powerful teacher needed; claimed to beat distillation from a static stronger teacher on CIFAR-100 and Market-1501 [abstract only].
**5b. Online distillation / codistillation (Anil et al. 2018).** https://arxiv.org/abs/1804.03235 (opened, abstract). Two networks on disjoint data subsets agree with each other's (possibly stale) predictions; improves reproducibility and speed.
- Apply: train two h256 students from different seeds/parents simultaneously, each regularized to the other's prediction (and to the cached h384 teachers). This keeps disagreement alive during training instead of freezing it in a cache. Needs two models in memory (h256 is cheap).
- Cheapest test: co-train 2 h256 warm-started from different parents; loss = label + 0.5 * MSE(student_i, detach(student_j)); evaluate each and their average.

**5c. Mean Teacher (Tarvainen and Valpola 2017).** https://arxiv.org/abs/1703.01780 (opened, abstract). EMA-of-weights teacher provides consistency targets continuously.
**5d. SWA (Izmailov et al. 2018).** https://arxiv.org/abs/1803.05407 (opened, abstract). Average SGD iterates under a constant/cyclic LR; flatter optima; approximates FGE with one model; almost no overhead.
**5e. Be Your Own Teacher (Zhang et al. 2019).** https://arxiv.org/abs/1905.08094 (opened, abstract). Deep sections teach shallow sections within one network; CNN specific.
- Apply: SWA/EMA over the student's last epochs is a near-free ensembling-in-weight-space; since same-parent warm-started students are linearly connected, averaging checkpoints from several distillation runs (model soup) may recover part of the blend gain without any teacher. [unverified for GRUs]. Also use the EMA of the student as the teacher in the second half of training (Mean Teacher).
- Cheapest test: keep an EMA (decay 0.999) of the student during the existing 3-epoch run and evaluate the EMA; zero extra data.

## 6. Decoupled / relational / correlation-based KD

**6a. DKD (Zhao et al. 2022).** https://arxiv.org/abs/2203.08679 (opened, abstract). Splits KD loss into target-class and non-target-class terms (classification-specific). No analogue for 2 continuous outputs; skip.
**6b. RKD (Park et al. 2019).** https://arxiv.org/abs/1904.05068 (opened, abstract). Transfer relations among examples (distance and angle losses).
**6c. DIST (Huang et al. 2022), KD from a stronger teacher.** https://arxiv.org/abs/2205.10536 (opened, abstract). A strong teacher's predictions deviate more from the student, so exact KL matching hurts; they use a correlation-based (Pearson-style) relation loss at inter-class and intra-class levels.
- Apply: our metric is weighted pooled Pearson. A Pearson-matching loss between student and teacher over a batch/sequence window ignores per-run level offsets (94% of our residual is per-run offset) and so lets the student ignore the offset noise while matching the rank/shape. This is the best structural fit from this family. Use batch-level (weighted) correlation loss on student vs teacher-mean plus student vs label.
- Cheapest test: replace the 0.7*MSE term by `1 - weighted_pearson(student, teacher_mean)` computed over the (batch x time) tensor with the metric's weights; or add it as an extra term. Also try a per-sequence detrended version.
- Also a listwise/ranking distillation angle (Tang and Wang, arXiv 1809.07428) was not opened; mark [unverified].

## 7. Noisy-label, label correction, privileged information

**7a. Lopez-Paz et al. 2016, Unifying distillation and privileged information.** https://arxiv.org/abs/1511.03643 (opened, abstract). Generalized distillation: a teacher trained with privileged information (data available only at training time) passes knowledge to a student that lacks it.
- Apply: this exactly fits "teacher sees future rows, student causal". Our labels lead features by ~1 row; a bidirectional (non-causal) teacher with access to t+1..t+k features would be a privileged-information teacher whose soft output at t is a denoised target. Training-time only, serving stays causal. Caution: leakage rules concern evaluation fitting groups, not teacher access to future rows within train sequences, but the teacher's outputs must still not be derived from the evaluated fold's labels. (Validity note: use train-split sequences only for local comparisons.)
- Cheapest test: train a small bidirectional GRU (or causal GRU with +1 lookahead window) on the train split; cache outputs; distill to the causal student. Larger scientific bet than the others.
**7b. Zhang et al. 2020, Distilling Effective Supervision from Severe Label Noise.** https://arxiv.org/abs/1910.00701 (opened, abstract). Uses a small clean set to estimate per-example weights and pseudo-labels; classification focus.
- Apply: our labels have large irreducible level-offset noise. The analog is a trusted subset (e.g. market class 1/2 rows where local corr is .75-.81) for weighting the label term, and using teacher outputs as pseudo-labels where the label disagrees heavily. Weak evidence base for regression [unverified transfer].

## 8. Kaggle / competition evidence (thin)

I looked for Jane Street / Optiver / G-Research / Ubiquant / JPX solutions that distilled an ensemble into one network and found **no verified instance**.
- Ubiquant 1st place: average of 5-fold LGBM and 5-fold TabNet; a custom MLP was dropped as unstable in CV (search snippet of https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/k-i-y-1st-place-solution-our-betting-strategy; page not opened) [snippet].
- Ubiquant 8th place: weighted ensemble of 10 LGBM + 30 NN + hand-built (snippet, not opened) [snippet].
- Optiver 9th place: GBDT, no NN (snippet) [snippet].
- BirdCLEF+ 2026 1st place used distillation per backbone inside a diverse ensemble, not as an ensemble replacement (snippet) [snippet]: https://www.kaggle.com/competitions/birdclef-2026/writeups/1st-place-solution-noisy-student-meets-distillati
- Playground S6E5 8th place used self-distilled students on OOF predictions of stacked levels (title opened, body not retrievable) [snippet].
- NVIDIA playbook (snippet): multi-round pseudo-labeling with ensemble teachers beats single-pass; fine-tune on original labels after to reduce noise. https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/ [snippet].
- Takeaway: the competition literature supports "distill/pseudo-label then fine-tune on true labels" and "keep ensembles diverse in the final blend", consistent with our observation that inference-time blending beats the distilled student. Medium article https://wbrucek.medium.com/knowledge-distillation-ac148f66faff returned HTTP 403 (not read).

## Learning-rate / epoch scaling evidence
- Beyer (4a): long schedules; ImageNet gains continued from 1200 to 9600 epochs with Adam lr 0.03 batch 4096 cosine, warmup 5000 steps, grad clip 1.0. Absolute LRs do not transfer to a GRU; the transferable parts are "much longer", "cosine to zero", "same view", "no hard-label term".
- Stanton (1b): optimization-limited matching suggests longer/better-conditioned schedules before changing the loss.
- Cho (1c): stronger teachers are not automatically better; try intermediate-quality teachers.
- We have no verified source on the "right" lower LR for warm-started students; lr 1e-4 over 3 epochs is a very short, low-energy schedule, so the direction supported is *more steps and a warm cosine restart*, not necessarily lower LR [unverified].

## Ranked experiments for our pipeline

Baseline for all: warm start from 0.666-class parent, cached 2-seed teacher mean, alpha 0.7, 3 epochs, lr 1e-4 (0.66938 / 0.66861 control). Fold 1 only first; one-run noise sd 0.002, seed sd 0.001, so differences < 0.002 need seeds.

| # | Experiment (flags) | Mechanism / source | Expected gain over 0.66938 | Cost |
|---|---|---|---|---|
| 1 | Long schedule on cached targets: epochs 3 -> 12/24, cosine from lr 3e-4 to 0, alpha 0.7 -> 1.0 in last half, record fidelity | Beyer 4a, Stanton 1b | +0.000 to +0.002 (up to ceiling set by cache overfit) | Low: GPU hours scale with epochs, no code change |
| 2 | Online teacher with identical noisy view: teachers run on-the-fly on feature-noise/dropout-perturbed batch, student sees same view, alpha 1.0, 10+ epochs | Beyer 4a, DEGU 2e | +0.001 to +0.003 (largest plausible; unproven for rows that are market streams) | Medium: teacher forward each step (h256x2 + h384), memory for 3 models, needs augmentation validity check |
| 3 | Pearson-matching distill loss: `1 - wPearson(student, teacher_mean)` replacing/added to MSE term; optional per-sequence detrended | DIST 6c | 0 to +0.002 | Low: loss swap, cache reuse |
| 4 | Multi-teacher heads: separate heads per teacher (h256 seeds, h384 seeds) plus mean head, MSE each, mean head used for serving | Zuchniak 2d, DEGU 2e, Hinton 1a | 0 to +0.002 | Low-medium: cache per-teacher outputs (already have per-seed caches), small head change |
| 5 | Teacher spread head / precision weighting: aux head predicts teacher SD (weight 0.3), weight MSE rows by 1/(sd^2+eps) clipped | DEGU 2e, uncertainty-weighted 3g | 0 to +0.001 | Low |
| 6 | Hidden-state hint: cache teacher penultimate state (fp16 or PCA-64); student projector Linear->teacher dim, MSE at weight 0.1-0.3 | FitNets 3a, PKD 3c | 0 to +0.002 | Medium: cache storage (384 floats/row), one projector layer |
| 7 | Weight averaging: EMA 0.999 and SWA over last epoch(s) of the student; optionally soup across distillation runs from the same parent | Mean Teacher 5c, SWA 5d | +0.000 to +0.001 each; may stack with 1 | Very low: eval-only plus checkpoints |
| 8 | Co-distillation of two students: loss = label + 0.5*MSE to the other student (detached) + cached teachers | Anil 5b, DML 5a | 0 to +0.002 | Medium: two models in memory, GPU 2x |
| 9 | Born-again / teacher-assistant round: second-generation student (from scratch or warm) with teachers {parent, 0.66938 student, h384 caches}; also try earlier-epoch h384 teachers | Furlanello 2b, Mirzadeh 1d, Cho 1c | 0 to +0.001 | Low-medium: one more distill run |
| 10 | Privileged-information teacher: bidirectional (or +1..k lookahead) teacher on train sequences; distill outputs into the causal student; keep serving causal | Lopez-Paz 7a | Unknown, -0.001 to +0.003; highest variance, highest novelty | High: new teacher training, careful leakage review |

Suggested order of work: 1 and 7 are nearly free and may reveal if the bottleneck is optimization (Stanton) or target quality; 3 and 4 are one-line changes on existing caches; 2 is the largest-upside item and should follow once 1 shows whether more steps help at all; 10 is the strategic bet.

Expected-gain brackets are my judgment from the literature's qualitative effects and the +0.0008/+0.003 gap in DISTILL_FACTS; they are not measured and not confidence intervals. Combined, the realistic ceiling is near the blend gain (~+0.003); DEGU suggests students recover only a part of ensemble gains [partially verified, see 2e].

## Diagnostics worth running before any fit (read-only, cache-level)
- Teacher-member disagreement on train rows vs valid rows (2a).
- Student-teacher fidelity curve vs epoch (1b).
- Decompose teacher-mean advantage: how much of the +0.003 blend gain survives if teachers' per-run level offsets are removed (6c, 94% of residual is per-run offset).
