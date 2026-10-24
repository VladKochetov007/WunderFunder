---
kind: review
status: complete
created_utc: 2026-10-24T18:43:30Z
lens: questions-and-external-directions
author: independent subagent (fresh context)
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/unified/sources/README.md
  - research/unified/sources/ASTRA_ULTRA_REVIEW_SOURCES_20261023_R1.md
  - research/unified/reviews/astra_ultra_audit_20261023_r1/SYNTHESIS.md
  - research/campaign_2026-10-24/resume4h_r1/CAMPAIGN_SYNTHESIS_R1.md
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/RESULTS_R1.md
  - src/wnn/gmetric.py
  - src/wnn/training/losses.py
  - docs/data_overview.md
  - docs/rules.md
---

# Questions and directions — 2026-10-24

One-shot, read-only review through a questions-and-external-directions lens. No fits, no repo scripts, no mutations, no writes other than this file. A complete file from an earlier attempt at this path was read first; every repo claim kept from it was re-verified (one correction found: per-target weight mass is already measured as even — see Card 5), and its unique verified content is incorporated. External evidence is abstract/search-result level unless stated; no external result is a Challenge2 gain. Additive to the 2026-10-23 Astra synthesis and to siblings [01_value_allocation_R1.md](01_value_allocation_R1.md) (endgame/value lens) and [02_elephants_red_team_R1.md](02_elephants_red_team_R1.md) (noise-floor/TL3-anatomy lens), skimmed to avoid duplication.

## Main judgment (bullets)

- **The final-event rule makes upload discipline itself a research question.** Verified in `docs/rules.md`: final ranking is private-only, computed by re-scoring *previously submitted and scored* solutions ("you will be asked to choose some of your _previously submitted and scored solutions_"), 5 submissions/day, tie-break to the earlier upload. A model never uploaded has zero placement value regardless of local evidence. The question set starts from the portfolio of scored ZIPs, not the next +0.0003 screen. Kaggle's own selection guidance (web-verified) adds: avoid two near-identical selections; hedge with one public-best plus one local-CV/robust pick.
- **The official metric has structural features we interrogate less than our models.** From `src/wnn/gmetric.py` directly: rows with `clip(y)=0` carry zero weight and drop out of every pooled moment; the score is the mean of two *separate* per-target correlations (`wp = 0.5*(t0+t1)`), while training WMSE pools both targets under one weight denominator (`src/wnn/training/losses.py`: `(w*(out[...,:2]-tgt)**2).sum()/w.sum()`); all moments are pooled across sequences, not per-sequence. Each yields a zero-fit, decision-relevant diagnostic on frozen prediction arrays (Q3–Q6, Q9, Q14–Q15).
- **The most useful external transfer for the online-adaptation family is a control, not a method.** A 2025-style delay-aware audit of test-time adaptation for forecasting (abstract-level, re-verified this session) reports: a hyperparameter-free bank of RLS closed-form correctors beat three of four audited learned adapters; an ELF-style linear corrector beat all four; leaky next-step updates inflated simple adapters' apparent gains up to 110%; the backbone training recipe moved frozen online error by up to 25x, more than any adaptation effect. Our callback receives **no labels at all**, so the algorithms do not transfer — the comparator discipline and the backbone-dominance finding do.
- **Weighted-metric competition precedents converge on boring-member ensembles plus selection hedging.** M5 accuracy winners (web-verified): ensembles of many direct/recursive LightGBM variants; Kaggle guidance: diversify final selections. Our own history rhymes: blends/ensembles are the largest exposed gains (+0.0037), while every single-model mechanism screen of the last three weeks landed at +0.0001–0.0004. New inference: our member pool is 100% GRU-family — a non-recurrent diversity member is untested (Card 8).
- **State-dependent excitation is the sharpest representation lead.** The state-dependent Hawkes LOB paper (abstract re-verified) estimates spread and queue-imbalance as state variables on real LOB data and finds excitation/endogeneity strongly state-dependent, most pronounced in *disequilibrium* states. We feed 99 event features additively; the literature says their mechanism is the *interaction* with book state — testable zero-fit on frozen residuals before any fit (Q10–Q11, Card 2, W3).
- **Teammate PR32 consumption remains the highest-leverage open allocation question** (630M/1.54B at ~50% updates, 0/4 evaluated, no artifact channel). External foundation-model evidence (abstract-level) argues for domain-native teachers over imported public TSFMs, which also matches the rules. The decision-changing deliverable is a frozen evaluation+distillation consumer packet prepared *before* artifacts land (Q8, Card 7).
- **The LOB reading is documented, not speculative** (`docs/data_overview.md` names L11 books, trades, two instruments, anonymized indices). The productive semantics question is which *anonymization-invariant* quantities (group sums/differences/ratios: spread proxy, imbalance, volume shares) are already inside the 211 inputs and which named state variables condition feature value (Q10–Q11, Card 6).

## Questions we should be asking (ranked)

Each entry: question / decision it changes / cheapest answer path. Zero-fit means analysis of existing arrays/receipts only; all such analyses on exposed folds are diagnostics, not clean validation.

1. **What exact portfolio of previously scored ZIPs will we offer the private re-score, and are the two quota ZIPs in it before anything else happens?** Changes: the upload schedule to 2026-11-15, the definition of "done" for every research arm, and the selectable-slot count (portal mechanics — "some" solutions — must be verified by the user; sibling 01 flags this). Cheapest: written selection plan enumerating scored ZIPs by composition/runtime diversity from `submissions/` receipts and `research/campaign_2026-10-22/quota_retained_pair_r1/PUBLIC_RECEIPT_R1.json`; zero-fit.
2. **Can we pre-declare the decision rule a public receipt for the two quota ZIPs will trigger, before uploading?** Changes: whether/when to spend quota on TL3+pair vs TL3+S3, and how later local→public reads are calibrated. Cheapest: write the rule (e.g., "pair public ≥ TL3−0.0005 keeps pair as finalist; otherwise revert to S3"), then upload; zero-fit, two submissions.
3. **What fraction of scored rows per target/fold has `clip(y)=0` exactly (zero official weight), and where does official weight mass sit by |y| decile?** gmetric.py excludes zero-weight rows from every moment; WMSE training mirrors this via `w=|z|`, but tail-vs-mass effort allocation does not know the split. Changes: tail-row priority, zero-row shortcut analyses, training-weight sanity. Cheapest: zero-fit scan of `valid.parquet` targets under the given mask (or reuse existing frozen caches).
4. **Where in the |y| weight distribution does the pair-vs-TL3 gain (+0.0037 F0 / +0.0002 F2) live — small-|y| mass rows or large-|y| tail rows — and is it the same targets/components on both strata?** Sibling 01 asks the per-target/breadth decomposition; this adds the weight-decile axis. Changes: which members a final assembly needs for tail coverage. Cheapest: zero-fit decomposition of frozen arrays.
5. **Is the training-time t0/t1 balance right, given the official score averages two separate correlations while `main_loss` pools both targets in one denominator?** Mass is already measured even (target-0 shares 49.74%/49.11%, batch-mass audit), so the open part is per-target *loss/signal* balance, not mass. Changes: whether a future admitted fit uses per-target denominators or per-target heads. Cheapest: zero-fit per-target score/loss decomposition of frozen members (gmetric already returns `t0`, `t1`); a 2-fit A/B only if one target is systematically under-served.
6. **How much score is at stake in the clipping band: what fraction of frozen predictions sits outside ±2 per target, and would any per-target affine recalibration (a·p+b) of frozen predictions change pooled clipped WP?** Correlation is affine-invariant except through clipping and zero-weight-row structure, so this bounds the entire output-calibration family in one measurement. Changes: kill or justify all recalibration work. Cheapest: zero-fit sweep with a handful of pre-declared transforms; exposure-labeled.
7. **What is the marginal local WP per minute of callback for candidate members (TL3 ~39 min reported; pair +24.1% callback; public replays observed up to 3227 s of the 3600 s cap per sibling 01), and what member count fits a runtime gate under unknown final volume?** Changes: final ZIP composition and any new member worth training. Cheapest: zero-fit — combine measured runtimes with fixed-combination grades on the quota sample.
8. **When PR32's four teacher endpoints finish, what runs first — official-protocol evaluation or a distillation packet — and is that packet frozen now?** Changes: sequencing of ~108k Kaggle GPU-seconds and the B300 follow-up. Cheapest: write and freeze the evaluation+distillation consumer packet (code-only) and its cost model today; zero-fit.
9. **Are t0 and t1 one signal at two horizons or two different signals?** Changes: whether target-specific members/mixtures (older H-393 proposal) or per-target heads deserve a slot. Cheapest: zero-fit cross-target analysis — per-target scores of frozen members plus residual correlation of t0 predictions given t1 predictions; near-collinear targets kill H-393 cheaply.
10. **Which anonymization-invariant LOB quantities are already inside the 211 inputs / 99 event features, and which computable ones are missing?** Group sums/differences (mid proxy, spread proxy, bid-minus-ask aggregates, volume shares, queue-imbalance variants) need no depth ordering, so they survive anonymization (`docs/data_overview.md`). Changes: the next cheap feature arm. Cheapest: zero-fit inventory audit mapping inputs to named quantities; queue-imbalance ancestry already exists in-repo (`research/lit/opus_micro.md`, H-010) and must be listed, not re-proposed.
11. **Does the measured value of innovation/event features concentrate in book-disequilibrium states (queue-imbalance or spread buckets)?** State-dependent Hawkes work on real LOBs reports excitation effects strongest in disequilibrium states (abstract-level). Changes: prioritize event×state interaction features over flat event features. Cheapest: zero-fit bucket analysis of parent residuals (or the innovation-feature analogue) on frozen arrays.
12. **Can a closed-form comparator (RLS-style bank on causal input moments or proxy residuals) match the learned online heads, and is the F0/F1 head discrepancy estimator maturity rather than adaptation?** Changes: close or keep the online family; the audit-grounded version of the already proposed Z-versus-A control. Cheapest: zero-fit closed-form variant on existing caches; ≤4 small-head fits only after the comparator is written.
13. **Is the hidden `is_scored` mask predictable from causal row features, and does its local structure differ between strata?** Changes: whether local panels can be reweighted to mirror the official tilted measure (propensity-weighted *diagnostics* only — the Astra synthesis already bounds what propensity correction can identify). Cheapest: zero-fit classifier on `valid.parquet` mask features; exposure-labeled.
14. **How much member disagreement is between-component vs within-component, and could per-component mixture weights help under pooled scoring?** The metric pools all rows; component-conditional assembly can only act through pooled covariance. Changes: whether component-conditional blend work (beyond H-393) gets budget. Cheapest: zero-fit variance decomposition of frozen member disagreement by component.
15. **Would per-sequence output standardization (z-scoring predictions within each 20k-row sequence, then clipping) change pooled WP?** Pooled moments mean between-sequence prediction variance orthogonal to target covariance dilutes correlation; clipping and the mask make the sign unpredictable. Changes: a zero-training serving-side transform applicable to any final ZIP. Cheapest: zero-fit on frozen predictions of all members (W2).
16. **What does the ~100-upload public history teach about public-score noise by composition class?** (102 recorded as of 2026-10-22 per sibling 01's receipt; 8 public points, TL3-family 0.66771–0.66832 vs non-TL3 0.65839–0.66667.) The additive question is the empirical spread within composition classes. Changes: how much one new public point should move member choice. Cheapest: zero-fit tabulation of receipts grouped by composition class.
17. **Do any EDA observations contradict the documented LOB reading (65 event pattern classes, 0.074% boundary censoring, group/family vs per-field age divergence on 11–14% of scored pairs)?** Changes: how far LOB-specific priors may be trusted for feature arms. Cheapest: zero-fit contradiction scan of existing EDA records against documented LOB invariants.
18. **What is the highest-information first spend of ~108k unreserved Kaggle GPU-seconds: teacher-cache builds + student distillation fits (PR32 consumption), the i1 ordinary-diversity replication, or frozen F0/F1 online recovery?** Changes: the next cycle's shape. Cheapest: zero-fit arithmetic from existing cost receipts into a ranked menu with decision gates; preparation, not admission.
19. **Given measured full-endpoint seed noise (sibling 02's receipts), which current small positives would a pre-declared two-seed third-cell replication actually move, and is it worth its fits before the deadline?** Changes: replication vs new arms. Cheapest: zero-fit power reasoning; the replication itself is ~4 fits and needs separate admission.
20. **Should the F1 mean-shrink non-reproduction demote the online family's priority at all, or is F1 evidence too weak (interval [−0.00011,+0.00011]) to demote anything?** Changes: ordering of the proposed Z/A control versus other heads. Cheapest: zero-fit re-read of the two shrink receipts; an explicit decision memo.

## Transferable components

Format per research-reframe contract. NEW = not located in the repo by my searches; ALREADY-IN-REPO entries carry a path and are cited only as ancestry a card must beat or extend materially. Banned/duplicative transfers (GRU-D, TBPTT/Transformer-XL credit separation, numerical embeddings, Deep Sets, Beta-NLL, semiparametric distillation, data-constrained scaling, lead/lag compensation, time-warping, KalmanNet gains, TensorSketch, minGRU/minLSTM, chrono init, basic soups/SWA) are not re-proposed; any contact is explicit.

### Card 1 — Closed-form corrector bank as the online-family control
- **Source component:** "Training on the Future: A Delay-Aware Audit of Test-Time Adaptation for Time-Series Forecasting" (arXiv abstract, verified this session: RLS-bank correctors with no tunable hyperparameters beat 3 of 4 learned adapters; ELF-style linear corrector beats all four; leaky next-step updates inflate simple adapters' gains up to 110%; backbone recipe shifts frozen online error up to 25x, larger than any adaptation effect).
- **Structural analogy:** our online heads are learned adapters around a frozen backbone; the audit attacks exactly this family's evidential basis in dependent-stream settings, and its backbone-dominance finding mirrors our own "parent quality dwarfs head deltas" grades.
- **Concrete intervention:** implement a closed-form recursive corrector (RLS or per-coordinate median-of-RLS) on causal input moments / proxy residuals around the frozen parent as the comparator learned A/B/C heads must beat on parent-excluded groups; pair with the already-proposed equal-width zero-state Z control.
- **Assumptions:** label-free adaptation leaves input-moment correction as the only admissible signal; the audit's datasets had delayed labels, ours none — the transfer is comparator discipline, not the algorithm's gains.
- **Predicted observable & direction:** if learned heads carry relationship-specific information they beat the comparator out-of-component; otherwise comparator ≈ heads at a fraction of cost.
- **Cheapest informative test:** zero-fit closed-form variant on existing F0/F1 caches with the official grader.
- **Falsifier:** comparator dominates or matches all learned heads → move the learned-online budget to assembly.
- **Cost:** zero-fit analysis; ≤4 small-head fits only if heads survive.
- **Reusable artifacts:** saved F0/F1 head caches, parent predictions, official grader, direction-probe moments.
- **Family / status:** online-adaptation. NEW (the Z control alone is an in-repo proposal; the closed-form comparator is the added arm). Complements Astra card 07-B (maturity control) without repeating it.

### Card 2 — State-dependent event excitation (event features × book-state interaction)
- **Source component:** state-dependent Hawkes processes applied to LOBs (arXiv abstract, verified: spread and queue imbalance as state variables; excitation strongly state-dependent; endogeneity most pronounced in disequilibrium).
- **Structural analogy:** we have an event/innovation family (99 event features; innovation arms with small inconclusive positives at one fidelity) and documented book-state quantities (queue imbalance, spread proxies). We feed them additively; the source says their product carries the mechanism.
- **Concrete intervention:** append K fixed interaction inputs — innovation/event summaries × normalized queue-imbalance bucket (one-hot or tanh), per instrument — as a zero-initialized suffix on a frozen parent, warm continuation.
- **Assumptions:** group sums define imbalance without depth ordering (true by `docs/data_overview.md`); component heterogeneity does not wash out the interaction under pooled scoring.
- **Predicted observable & direction:** residual reduction concentrated in high-|imbalance| buckets; larger endpoint delta than the flat innovation arm at matched input count.
- **Cheapest informative test:** zero-fit residual bucket analysis (Q11) on frozen predictions before any fit.
- **Falsifier:** innovation-feature value flat across imbalance buckets → no headroom; drop before fitting.
- **Cost:** zero-fit diagnostic; ~2 fits for the warm suffix if positive.
- **Reusable artifacts:** parent caches, imbalance computation from raw 112 inputs, grader.
- **Family / status:** input representation / event-clock. NEW as an *interaction*; flat queue-imbalance features and the innovation clock are ALREADY-IN-REPO (`research/lit/opus_micro.md` H-010; `research/campaign_2026-10-18/online_features_r1/innovation_clock_r1`).

### Card 3 — Normalization component decomposition (RevIN parts, not RevIN whole)
- **Source component:** "On the Role of Reversible Instance Normalization" (arXiv abstract, verified: decomposes normalization into temporal input shift, spatial input shift and conditional output shift; finds several RevIN components redundant or detrimental).
- **Structural analogy:** training normalizes per-batch with equal-batch weighting; the batch-mass audit measured each batch's relative multiplier shifting 0.10–9.74x (F0) / 0.10–8.29x (F1) versus pooled rows, and component concentration dropping (inverse-HHI 118.94→175.69, 125.41→177.11). The source says the *parts* of normalization have separable effects — exactly the axes our audit did not separate.
- **Concrete intervention:** decompose the next normalization comparison into fixed cells — (stats source: per-batch vs expanding-prefix causal vs sequence-prefix) × (shift only vs shift+scale) — measured on the existing fit-suffix arrays first; then at most one matched pair on the most-changed axis, holding membership/order/update count/LR progress fixed (per the audit's own conditions).
- **Assumptions:** causal prefix statistics only (whole-sequence RevIN leaks; in-repo RevIN causal-limit note).
- **Predicted observable & direction:** multiplier spread narrows or flips by component class under prefix statistics; quality direction unknown — that is the point of separating axes.
- **Cheapest informative test:** zero-fit extension of the batch-mass audit with per-stat-source weighting measures.
- **Falsifier:** all decomposition variants produce near-identical batch measures → the normalization axis is dead for the remaining budget.
- **Cost:** zero-fit; one matched fit pair only with separate admission.
- **Reusable artifacts:** batch-mass audit scripts/receipts and fit suffixes.
- **Family / status:** objectives/normalization. NEW as a decomposition; basic RevIN source is ALREADY-IN-REPO (`research/unified/validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md`).

### Card 4 — Diversity-aware fixed blend-weight family, preregistered (Gibbs/NCL-style)
- **Source component:** "Generalized Gibbs Ensemble Weighting for Forecast Combination" (arXiv abstract, verified: Gibbs-style exponential-in-loss weights with diversity-aware score corrections; no universal dominance claimed; simple rules like mean/median/inverse-loss are strong baselines) and "Regularized Ensemble Forecasting" (abstract: variance-minimizing weights combining current forecasts and historical performance). Ancestry: negative-correlation learning already registered (Brown 2005 in the Astra source register).
- **Structural analogy:** blends are our largest exposed gains (+0.0037), but current weights are hand-fixed; a closed-form weight family with an explicit diversity term gives a small preregistered grid computed only from local multi-fold predictions.
- **Concrete intervention:** on frozen member predictions across F0/F2 (and F1 if arrays exist), evaluate a pre-declared set: equal, inverse-loss Gibbs, one diversity-corrected Gibbs variant, one variance-minimizing regularized variant, and the historical fixed weights; select nothing else; carry the winner as fixed coefficients.
- **Assumptions:** all folds exposed ⇒ selection is exposure-labeled; the grid stays tiny to limit adaptive selection; no leaderboard feedback enters the weights (standing policy; J1's fitted weights already lost to equal weights in-repo).
- **Predicted observable & direction:** diversity-corrected weights ≥ equal on both strata, or the family is dropped; intervals conditional on fixed predictions.
- **Cheapest informative test:** zero-fit arithmetic on existing arrays with the official grader.
- **Falsifier:** equal weights beat all variants on both strata → keep equal weights, close the family.
- **Cost:** zero-fit.
- **Reusable artifacts:** quota-sample grades, member prediction arrays, greedy-saturation receipt E-ENSEMBLE-GREEDY-FOLD1-20261010 (ALREADY-IN-REPO evidence that few members saturate, selection-biased).
- **Family / status:** ensembles/blends. NEW as a bounded preregistered weight family; greedy selection itself is ALREADY-IN-REPO.

### Card 5 — Per-target loss balancing, reframed after the mass measurement
- **Source component:** multi-task balancing literature (generic, abstract level) — support only; the load-bearing fact is local and verified: the official score averages two separate per-target correlations (`gmetric.py` `wp=0.5*(t0+t1)`), while `main_loss` pools both targets in one weight denominator (`src/wnn/training/losses.py`).
- **Structural analogy:** even at even weight mass (measured 49.74%/49.11% target-0 shares, batch-mass audit), a pooled denominator lets the target with more *learnable signal* or larger residual scale absorb capacity; the official score grades each target independently. **Correction to the earlier attempt:** the mass-ratio measurement is already done and even — the open question is per-target loss/signal balance, not mass.
- **Concrete intervention:** zero-fit per-target decomposition of frozen members (scores via `gmetric` outputs; per-target WMSE from caches); if one target is systematically under-served relative to its official 50% role, one matched pair with per-target denominators or per-target heads.
- **Assumptions:** chunked batching preserves per-target ratios (measurable); per-target headroom exists at all (falsifiable cheaply).
- **Predicted observable & direction:** the weaker per-target correlation improves without degrading the other; mean official score rises.
- **Cheapest informative test:** the zero-fit per-target decomposition itself (extends sibling 01's priority 4 and the audit's own "report both per-target official scores" condition).
- **Falsifier:** per-target scores already balanced and per-target losses proportional → drop without fitting.
- **Cost:** zero-fit; 2 fits if admitted later.
- **Reusable artifacts:** batch-mass audit target masses; frozen member per-target arrays; grader per-target outputs.
- **Family / status:** objectives. NEW as the *training-side* per-target question; H-393 (ALREADY-IN-REPO, `research/unified/reviews/CHALLENGE2_PUBLIC_PRIVATE_STRATEGY_20261012.md`) covers blend-level target-specific mixtures only.

### Card 6 — Anonymization-invariant LOB level normalization (group-relative encoding)
- **Source component:** DeepLOB-family preprocessing and modern LOB microstructure guides (web abstracts: normalize prices as differences from a reference level, volumes as shares) — abstract level.
- **Structural analogy:** p/v groups are documented but index-shuffled (no depth ordering). Depth-free analogues exist: `p_k − group-median(p)`, `v_k / Σgroup(v)`, bid-group minus ask-group aggregates. The failed dictionary/geometry arms tested *discrete identity*; this tests *continuous ratio/difference structure* with named LOB semantics.
- **Concrete intervention:** append a fixed small block of group-relative continuous inputs (per instrument: median-relative prices, volume shares, group sum differences) as a zero-initialized warm suffix on a frozen parent.
- **Assumptions:** anonymization preserved relative structure within groups (supported by documented group semantics; spot-check zero-fit first).
- **Predicted observable & direction:** exact standalone parity at init, then residual reduction vs a matched-width control suffix of random continuous transforms.
- **Cheapest informative test:** zero-fit variance/explainability screen of the new inputs against parent residuals on existing caches.
- **Falsifier:** the random-transform control suffix matches the LOB-normalized suffix → semantics add nothing beyond generic nonlinearity.
- **Cost:** zero-fit screen; ~2 fits if admitted.
- **Reusable artifacts:** raw 112-input readers; warm-suffix machinery from the i1 work.
- **Family / status:** input representation. NEW as continuous group-relative encoding; discrete atoms/dictionaries are ALREADY-IN-REPO-negative (`research/unified/ideas/DISCRETE_OCCUPANCY_HEURISTICS_20261017_R1.md`).

### Card 7 — Domain-native teachers over external pretrained TSFMs (guidance card)
- **Source component:** "How Foundational are Foundation Models for Time Series Forecasting?" (arXiv abstract, verified: zero-shot skill tied to pretraining domains; fine-tuned TSFMs do not consistently beat smaller dedicated models relative to their size).
- **Structural analogy:** rules allow public pretrained models without external market data; the open question is whether importing a public TSFM (Chronos/TimesFM/Lag-Llama style) as teacher or initializer beats consuming the teammate's domain-native 630M/1.54B teachers.
- **Concrete intervention:** none now — a decision rule: spend zero effort on external TSFM importation; prioritize PR32 consumption (Q8).
- **Assumptions:** abstract-level external evidence; anonymity and licensing make TSFM import awkward anyway.
- **Predicted observable:** none; this card reallocates attention.
- **Cheapest informative test:** already taken (the scan).
- **Falsifier:** a teammate demonstrates a public-TSFM-derived endpoint beating domain-native teachers locally.
- **Cost:** zero.
- **Family / status:** teacher scaling / portfolio. NEW as guidance; the teacher-first policy is ALREADY-IN-REPO (`specs/adr/0043`, teacher-scaling amendment).

### Card 8 — A non-recurrent diversity member (GBM/window-regression family), M5 precedent
- **Source component:** M5 accuracy competition results (web-verified search level: winners used ensembles of many direct and recursive LightGBM variants; weighted RMSSE metric; most top entries gradient-boosting based). No M5 method detail beyond this is imported.
- **Structural analogy:** our member pool is entirely GRU-family plus GRU-distilled students. Blend gains (+0.0037) came from *within-family* diversity; the forecast-competition record says cross-family diversity is where combination usually earns more. A small gradient-boosted or ridge regression on causal window summaries (rolling moments, event counts, imbalance aggregates — all computable causally) has a maximally different error structure from recurrent state.
- **Concrete intervention:** stage 1 zero-fit: score a cheap ridge/GBM proxy on cached causal window features against frozen members' residuals to estimate complementarity; stage 2 (if complementary): train one small GBM member on existing caches and test fixed blends; stage 3 only if blends win: distill the blend into a compliant student (serving a GBM per-row inside the 60-min CPU replay is a real risk — the student path avoids it).
- **Assumptions:** causal window features are already materialized or cheaply materializable from caches; local blend value transfers to a served endpoint only via distillation or a native inference path.
- **Predicted observable & direction:** proxy residual correlation with members' residuals low, own WP non-trivial → complementarity headroom; direction of served gain unknown until the blend is graded.
- **Cheapest informative test:** the zero-fit proxy screen on existing caches.
- **Falsifier:** proxy residuals highly correlated with existing member residuals, or own signal near zero → drop before any fit.
- **Cost:** zero-fit screen; ~1–2 fits for a member; distillation extra.
- **Reusable artifacts:** existing window/event feature caches, grader, blend tooling.
- **Family / status:** ensembles / member diversity. NEW (no non-recurrent member family located in the repo; boosting appears only as imported competition-transcript context, not a tested member).

## Wildcards (higher-risk)

### W1 — Submission quota as an information-design problem
- **Source component:** Kaggle's official selection guidance (web-verified: avoid two very similar final selections; public LB is a small split, so hedge — one public-best plus one local-CV/robust pick) applied to the verified rule that only *previously submitted and scored* solutions are re-scored privately.
- **Structural analogy:** 5 submissions/day until the deadline is a sensing budget over public-sample noise; near-identical ZIPs waste it, deliberately diverse compositions measure public→private transfer spread.
- **Concrete intervention:** design the remaining uploads as a small fixed portfolio: the two quota ZIPs, one deliberately diverse composition (different member family), and one repeat of a prior ZIP to measure public-score determinism; all compositions fixed in advance, no coefficient tuning from feedback (repo policy).
- **Assumptions:** public sample fixed across days; public/private split independent of submission order; portal selection slot count TBD (Q1).
- **Predicted observable & direction:** repeat-ZIP public scores identical (determinism check); diverse-composition gap vs TL3 estimates transfer spread rather than a single point.
- **Cheapest informative test:** one quota day with the repeat-ZIP determinism check.
- **Falsifier:** public scores vary run-to-run for identical ZIPs → all public-feedback reasoning needs a noise model first.
- **Cost:** zero compute; quota days only. **Risk:** over-reading public feedback; bounded by pre-declared rules.
- **Family:** submission strategy. NEW.

### W2 — Self-calibrating per-sequence output standardization
- **Source component:** conformal/online-recalibration ideas (Conformal PID and EnbPI abstracts, verified this session) re-targeted to a label-free setting: use the model's *own prediction stream* within a sequence as the calibration population.
- **Structural analogy:** the metric pools moments across sequences; between-sequence prediction variance orthogonal to target covariance is pure dilution. Per-sequence standardization of predictions (then clip) is a serving-side transform requiring no labels and no retraining, applicable to any final ZIP.
- **Concrete intervention:** at serving time, maintain running per-sequence mean/std of the model's own past predictions and emit standardized (then clipped) values; evaluate the identical transform zero-fit on frozen predictions first.
- **Assumptions:** prediction streams non-degenerate per sequence; standardization does not interact badly with clipping mass.
- **Predicted observable & direction:** pooled WP moves either way — the zero-fit measurement is the entire first experiment; direction unknown.
- **Cheapest informative test:** zero-fit on all frozen member arrays (Q15).
- **Falsifier:** transform changes WP by <0.0001 on every member → drop.
- **Cost:** zero-fit; serving cost O(1) state if adopted. **Risk:** interacts with deterministic-replay requirements (must stay exact and causal; running statistics use only past predictions, so causality holds, but float-order determinism must be verified against the native path).
- **Family:** serving/metric structure. NEW.

### W3 — Cross-instrument state-dependent delay estimation (i1 leads i0)
- **Source component:** state-dependent Hawkes cross-excitation (same verified source as Card 2, cross-instrument kernels) plus the in-repo i1/history signal (+0.0003 mixture, mechanism unresolved).
- **Structural analogy:** t0/t1 describe i0's future; i1 has no targets but its event stream may lead i0's. If cross-excitation delays i1→i0 are estimable per component from fitting groups, lagged i1 event summaries aligned by *estimated delay* (not fixed row lag) differ from every fixed-lag history arm we ran.
- **Concrete intervention:** zero-fit: estimate per-component i1→i0 event-time cross-correlation on fitting sequences; report the delay distribution's concentration (pre-declared statistic). Only if concentrated, propose delay-aligned i1 event inputs.
- **Assumptions:** component identity stable within a fold; delays stationary enough to estimate causally.
- **Predicted observable & direction:** concentrated nonzero delay distribution → delay-aligned features have headroom; diffuse delays kill it.
- **Cheapest informative test:** the zero-fit delay estimation itself.
- **Falsifier:** delay distribution ≈ flat or zero → the i1 gain is not a lead-lag effect; family reverts to the error-diversity explanation.
- **Cost:** zero-fit; ~2 fits if positive. **Risk:** multiple-testing on component heterogeneity; pre-declare the concentration statistic.
- **Family:** representation / cross-instrument. NEW; distinct from the ALREADY-IN-REPO cross-instrument correspondence fundamental card (H-ASTRA24-FUND), which asks whether pairing exists, not whether *timed* excitation exists.

## Literature log

All external claims above are abstract-level or web-search-result level (what the tools returned); no full text was read, no numbers beyond abstracts/summaries quoted, no paper claim treated as a Challenge2 result.

arxiv-search queries run (skill at `.agents/skills/arxiv-search/arxiv_search`), spaced to respect the API:

1. "test-time adaptation time series forecasting distribution shift" — used: delay-aware TTA audit (Card 1, top find, claims re-verified from the returned abstract); noted, not carded: PETSA (parameter-efficient adapters), TTA survey, Tweedie/GP intermittent forecasting (zero-inflation/tail framing).
2. "Hawkes process limit order book self exciting" — used: state-dependent Hawkes LOB paper (Card 2 / W3, claims re-verified); noted, not carded: functional limit theorems, fractional Hawkes, ephemeral self-excitation.
3. "forecast combination ensemble weights accuracy" — used: GGEW (Card 4) and Regularized Ensemble Forecasting (Card 4 secondary); noted: hit/win-rate weighting (metric-tailored weights exist as a field; not carded).
4. "reversible instance normalization time series" — used: "On the Role of Reversible Instance Normalization" (Card 3); other hits (energy-distance clustering, causality normalization, TSAT) not used.
5. "foundation models time series forecasting evaluation" — used: "How Foundational are Foundation Models..." (Card 7); noted: Lag-Llama (existence proof of public TSFMs), pre-trained forecasting models as feature extractors (classification; not used).
6. "M5 competition forecasting accuracy weighted" — used: M5 Uncertainty/GAMLSS paper (competition-metric context); GEFCom2014/2017 papers noted as competition precedents (quantile/lasso), not carded.
7. "correlation coefficient loss function deep learning optimization" — no usable hits (honest failure; consistent with the in-repo derivation that tilted-measure WMSE direction already aligns with population WP, so nothing was lost).
8. "mixture of experts time series forecasting" — WaveMoE, SCINet, Deep Poisson Mixture noted, not carded; the in-repo warmup-router negative result is closer evidence than any abstract.
9. "conformal prediction time series online" — used: Conformal PID and EnbPI abstracts for W2 framing only (both require label feedback, which our callback never receives).
10. "state space models long range time series forecasting" — no directly usable hits for our causal per-row setting (AeroChem is domain-specific physics-informed; LLM-Mixer recurring). Honest no-result.

WebSearch queries run and used:
- Kaggle final-submission selection strategy — used: official docs and discussion guidance (avoid near-duplicate selections; hedge public-best with CV-best; public LB is a small split) → W1, Q1–Q2. Sources: [Kaggle competitions docs](https://www.kaggle.com/docs/competitions), [Kaggle discussion 414544](https://www.kaggle.com/discussions/general/414544).
- M5 accuracy winners (first call network-failed; retry succeeded) — used: winners In & Jung used multiple direct/recursive LightGBM models; weighted RMSSE; LightGBM dominant among top entries → Card 8 and judgment bullet. Sources: [M5 accuracy paper](https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf), [IIF special issue intro](https://pmc.ncbi.nlm.nih.gov/articles/PMC9232271), [M5-methods repo](https://github.com/Mcompetitions/M5-methods).

Access failures recorded honestly: one WebSearch network failure (M5 query, retried successfully); the arXiv API returned results for all ten queries this session after spacing calls; no full-text claims are fabricated.

## Sources inspected

Repo (read in full or skimmed): `research/unified/sources/README.md`, `research/unified/sources/ASTRA_ULTRA_REVIEW_SOURCES_20261023_R1.md`, `specs/research-objective.md`, `research/unified/reviews/astra_ultra_audit_20261023_r1/SYNTHESIS.md`, `research/campaign_2026-10-24/resume4h_r1/CAMPAIGN_SYNTHESIS_R1.md`, `research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/RESULTS_R1.md` (per-target mass correction), `research/unified/ideas/README.md`, `docs/data_overview.md`, `docs/ARCHITECTURE.md` (skim), `docs/rules.md` (final-event and quota mechanics verified), `src/wnn/gmetric.py` (zero-weight rows, per-target averaging, pooled moments), `src/wnn/training/losses.py` (pooled two-target WMSE denominator verified), `research/unified/reviews/CHALLENGE2_PUBLIC_PRIVATE_STRATEGY_20261012.md` (H-393 ancestry), sibling reports `01_value_allocation_R1.md` and `02_elephants_red_team_R1.md` (structure/judgment skim for additivity), and the earlier attempt's file at this path (verified then superseded by this rewrite). Targeted checks: `research/lit/opus_micro.md` queue-imbalance ancestry, `E-ENSEMBLE-GREEDY-FOLD1-20261010` in the experiment ledger, H-393 in `research/hypotheses.jsonl`, `research/campaign_2026-10-22/quota_retained_pair_r1/PUBLIC_RECEIPT_R1.json` public roster (via sibling 01), `submissions/` inventory listing.
