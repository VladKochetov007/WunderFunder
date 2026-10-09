# Freebuff GLM | learned update-rule and timescale lens

# E530: independent online-learning research proposals

The user explicitly requested a new online-learning family for long-term competition winning and diverse proposals from Grok, Freebuff/GLM, OpenCode/Muse and Kimi. Use `conduct-creative-research` and `research-reframe`. This is independent ideation: do not read another provider's proposal before committing your own. No fit, paid API fallback, rental or submission is requested by this packet. Native fallback scouts are Luna/max and must be labeled as such.

## Task and information available

Wunder Connectome Challenge2 is a deterministic causal time-series regression task. Each independent sequence has 20,000 rows of 112 anonymous order-book/trade/additional features. The callback receives current features, sequence identity, step and whether output is required. The first 99 rows are warm-up. Every later row needs two predictions for hidden future-movement indicators `t0` and `t1`. The hidden scoring mask and targets are never passed to the callback. Sequence IDs do not encode time and cannot be a feature. All model/adaptation state must reset on a sequence change; only current and earlier rows within that sequence may affect a prediction.

The official score pools clipped predictions/targets (±2) with clipped target-magnitude weights; its mask and definition are fixed. The supplied train/valid labels are available for offline learning, including meta-training an online update. Previously inspected evaluation groups remain exposed; fitting-source ancestry must exclude evaluated groups. All learned weights and implementation must be our own. Reading literature and teammate methods is allowed; copying teammate code or using teammate weights is forbidden.

Local reference models are independently trained recurrent predictors; there is no requirement to keep that family. Final submission is offline Python 3.11, one CPU, 16 GB RAM, ≤20 MB ZIP and ≤60 minutes total. Research teachers may exceed the serving envelope on authorized local/Kaggle compute and later be compressed. No serving GPU. Local runs collectively must stay within 6 CPU cores / 20 decimal GB RAM / 5 decimal GB observed VRAM, one GPU worker; use only one CPU and ≤1 GB observed RSS for this ideation task.

Primary competition sources: https://wundernn.io/connectome/docs/data_overview, https://wundernn.io/connectome/docs/submission_guide, https://wundernn.io/connectome/docs/rules; local reviewed copies live in `docs/`. They explicitly allow causal sequence state and require reset. They do not explicitly discuss gradient-based parameter adaptation; distinguish direct text from interpretation. Causal fast weights can be formulated as resettable state with an offline-trained update function. Do not presume delayed target feedback or exact target reconstruction from anonymized features. Do not reject all online learning because that particular feedback is absent.

## What to deliver

Generate at least four mechanistically distinct ideas, including a serious attempt at one unconventional transferable component. Extract useful parts of sources even when their headline task does not match. Cite primary papers or author code. Clearly label your proposed transfer as an inference. No invented applicability or numerical expected-gain estimates.

For each idea give: mechanism; the precise available information used at prediction and update time; adaptation state and reset; offline objective; online objective or update; smallest useful matched controls; measurable success/failure; likely CPU cost and memory; a bounded first experiment; possible deeper descendants and teacher/distillation/blending value even if standalone score is weak. Separate scientific uncertainty from execution/resource eligibility. Keep exploration open; a confidence interval crossing zero is not an automatic rejection.

Favor learned or analytically cheap within-sequence updates, but also consider training-time online optimization and larger teachers if they create a viable compression path. Explain what would distinguish actual useful adaptation from extra state, additional parameters, or merely an easier auxiliary objective. Avoid a huge framework prerequisite.

Return public final proposals only. Do not reveal or archive private reasoning. Do not read credentials, `.env`, private authentication stores or unrelated account resources. Do not share/publish source or data. Record actual CLI/provider/model and whether inference really completed; access failure is not a negative scientific result.

## Independent lens for this provider

Explore online optimization and meta-learning as a family: offline-trained update rules, fast parameters versus latent optimizer state, timescale separation, and within-sequence adaptation using only permitted current/past observable features. Consider training episodes that simulate sequence boundaries without leakage. Identify a transferable small algorithmic component even if the source's headline task differs. Give primary papers or author-maintained code references; label the proposed transfer as inference. Do not assume ground-truth feedback is revealed during validation/inference. Return only public final proposals, with no private reasoning or work trace.
