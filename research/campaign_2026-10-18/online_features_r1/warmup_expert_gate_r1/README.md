---
kind: result
status: complete
created_utc: 2026-10-18T04:13:18Z
updated_utc: 2026-10-18T04:16:53Z
hypothesis_ids: [H-WARMUP-EXPERT-GATE-20261018-R1]
experiment_ids: [E-WARMUP-EXPERT-GATE-20261018-R1]
evidence_paths:
  - recipe.json
  - preflight_r1.json
  - result_r1.json
  - execution_receipt_r1.json
  - independent_review_r1.md
---

# Warmup-only expert routing screen

## Question and frozen method

Can input-only summaries from the first 99 observations of a sequence predict which of four existing F0 model endpoints will score best later in that same sequence? The gate receives 39 features: mean, log standard deviation, and normalized first-20-to-last-20 drift for 13 declared input groups. Its labels are each expert's per-sequence official score on gate-training components only. A standardized multi-output ridge model, temperature 0.01 and 50% shrinkage toward equal weights produce the router.

The four pinned experts are the warm parent, the full-geometry adapter, moments-only adapter, and dictionary-only adapter. Evaluation uses the fixed F0 split's 468 validation groups, 141 source-qualified components and four deterministic component-grouped cross-fit folds. The comparison metric is `global_wp_clip2_v1`; scoring excludes the first 99 rows. The frozen recipe is [recipe.json](recipe.json), and the exact input, prediction, fit-membership and ancestry preflight is [preflight_r1.json](preflight_r1.json).

## Results

| Cross-fit endpoint | Pooled WP | Difference from training-selected fixed expert |
|---|---:|---:|
| Gate | 0.68083333 | −0.00039318 |
| Training-selected fixed expert | 0.68122651 | — |
| Equal four-expert mixture | 0.68082591 | −0.00040060 |

The training-selected endpoint was the warm parent in all four folds. The gate was below that fixed parent in every held-out fold:

| Fold | Held-out groups / components | Gate WP | Fixed parent WP | Difference |
|---:|---:|---:|---:|---:|
| 0 | 136 / 36 | 0.68166308 | 0.68223648 | −0.00057340 |
| 1 | 108 / 35 | 0.62093787 | 0.62159850 | −0.00066064 |
| 2 | 115 / 35 | 0.71532200 | 0.71535970 | −0.00003771 |
| 3 | 109 / 35 | 0.70133905 | 0.70162575 | −0.00028669 |

The paired component bootstrap for gate minus the fixed expert is −0.00039318, with conditional 95% interval [−0.00074834, −0.00008756] over 141 components. The gate's difference from the equal mixture is only +0.00000742, with interval [−0.00000032, +0.00001518]. These intervals condition on the four fitted gates and fixed endpoints; they do not estimate seed variation or transfer.

The gate did not route meaningfully by sequence. Mean gate weights were [0.2542, 0.2463, 0.2504, 0.2491]. Its held-out mean per-sequence L1 distance from fold-average weights was 0.0121–0.0173 in each fold; none reached the preregistered 0.10 threshold. The preregistered B300-admission criteria were +0.001 pooled WP over the training-selected expert, a positive effect in at least three folds, and material input-dependent routing in at least three folds. This treatment met none of them.

## Interpretation and limits

On this repeatedly inspected F0 panel, first-99 input summaries did not identify a better one of these four endpoints. The fitted gate stayed close to equal weights, and the parent was the strongest fixed choice in every cross-fit fold. The screen cannot distinguish whether this reflects uninformative warmup summaries, weak diversity among these closely related endpoints, or both. It does not reject other causal adaptation, online learning with verified feedback, Kalman-derived features, learned state features, teacher improvements, or student-side transfer.

All candidate predictions and F0 labels were previously exposed. Cross-fitting prevents the router from using the held-out component labels to fit its weights, but does not make these validation groups fresh. No public score or clean-CV gain is inferred. No new B300 draft PR is warranted for this particular warmup-summary router. The existing retained-raw Kalman teacher/student handoff remains [draft PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17); this screen neither confirms nor replaces it.

## Execution and review record

Feature extraction was label-blind and used only the first 99 input rows from each frozen F0 group. The consumed input-slice digest is `d4339b9dba20dd5a2595dc855537d17b7ea37561bf9caa64e41905bc05b0a40a`; the extracted feature digest is `849e0a93b897468aa2a82346d29e8463984b493570b4589ba60dc8b57ffffc25`. The feature array and extraction receipt are retained under `/home/vlad/development/wunder/research_tmp/online_features_r1/warmup_gate_r1/extract_r2/`; their hashes and source identity are recorded in the grader receipts.

Two grader attempts stopped before target or mask numeric values were decoded or used. The first found a 63-character duplicated parent-sidecar hash constant in the grader; the recipe and artifact SHA-256 were correct. The second found an overly strict guard that required every prediction artifact to contain all 1,873 validation groups, although the three geometry artifacts correctly contain exactly the 468 F0 groups. The grader now uses hashes from the recipe and accepts only the exact frozen evaluation order or exact full-validation order. The corrected [independent static review](independent_review_r1.md) confirms the alignment and preflight ordering.

The final CPU-only grade exited successfully under `scripts/limited` with a 1.5 GB memory cap, 50% CPU quota, one thread and a 900-second runtime cap. Parent cgroup `oom` and `oom_kill` counters were both 4 before and after, with no delta; no GPU training or public submission occurred. A peak for the completed task scope was not retained, so this note reports the configured cap, not a measured peak. See [execution_receipt_r1.json](execution_receipt_r1.json) for the attempts and resource evidence.

## Source records

- [Frozen recipe](recipe.json)
- [Feature extractor](extract_warmup_features.py)
- [Gate grader](grade_gate.py)
- [Preflight](preflight_r1.json)
- [Result](result_r1.json)
- [Execution receipt](execution_receipt_r1.json)
- [Independent review](independent_review_r1.md)
- [Hypothesis ledger](../../../hypotheses.jsonl)
- [Experiment ledger](../../../experiments.jsonl)
