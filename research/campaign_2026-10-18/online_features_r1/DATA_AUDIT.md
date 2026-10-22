# Data-sentinel audit: observable data and online adaptation
Created: 2026-10-17 23:37 UTC
Scope: bounded, read-only inspection of competition docs, callback code, source-qualified split/provenance metadata and existing receipts. No parquet/array payload scan, model or native execution, fitting, or shared-repository edit. Scratch filesystem was reported as btrfs; outputs are confined to this directory.

## Audit result

The test callback is feature-only and causal. It receives the current row's seq_ix, step_in_seq, need_prediction, and 112-float state; it never receives t0, t1, or is_scored. It processes each 20,000-row sequence in order, including steps 0–98 warmup, must update state on every row, must return predictions on every required row 99–19,999, and must reset state when the sequence changes. seq_ix is shuffled and cannot encode sequence order or time. A prediction may use only current and earlier rows of the same sequence.

The copied rules prohibit external data and permit public pretrained models only if they contain no external market data. Current official data, FAQ, and submission docs were confirmed by root at 2026-10-17 23:37 UTC (URLs are in the receipt). The rules do not explicitly resolve online learning. Deterministic per-sequence RLS/fast-weight state updates from current/past observable features (and causally generated model state) fit the callback's input contract when they use no true labels and reset on every new sequence. Treat that as an input-contract-compatible research interpretation, not a blanket organizer ruling for global neural fine-tuning. Do not carry adaptive state across sequences.

## Safe first experiments

1. Keep the fitted F0 endpoint frozen and add a small deterministic per-sequence RLS/fast-weight or running-statistics state updated from the feature stream only. Reset it at seq_ix changes; process warmup and every post-warmup row. Any update rule/objective must use only inputs available by that step and fixed choices declared before scoring. This can test whether later-sequence behavior improves without test labels.
2. Independently diagnose sequence-position behavior using cached predictions and evaluator-only validation labels/masks. Predeclare position bins; report per-bin scores as diagnostics while retaining official pooled global WP as the primary metric. Never pass targets or masks into the predictor.

Use source-qualified fold manifests and record fit, teacher, initializer and evaluation membership. F0 has a usable component-excluded reference: warm_mechanism_f0_s3_c_r3@ep1, checkpoint SHA 78357eca…, bound to fold0 manifest SHA 09263ad4…; its three T3 teachers (seeds 3/4/5, epoch 2) have metadata showing membership matching the same fold0 fit groups. Its 468-sequence/141-component evaluation panel is repeatedly exposed and was selected on, so it supports a matched exposed-F0 screen, not a fresh-holdout claim. This known F0 lineage remains usable for its declared comparisons.

Canonical source-qualified manifests for folds 0–3 exist at research/campaign_2026-09-29/artifacts/splits/fold{0,1,2,3}.json. F2/F3 input manifests and portability receipts also exist in research/campaign_2026-10-06/bandit_continuation/ops_continuation_20261011/h382_local_clean_crossfold_f2_f3_r1/; that packet is prepared, not admitted or fitted. Its README says F2/F3 evaluation components had been inspected earlier. These can support cross-fold robustness comparisons with clean ancestry, not a pristine holdout. Fold manifests and existing recipes identify fit membership by source and row group; src/wnn/splits.py additionally checks component separation.

## Ancestry and semantic unknowns

submissions/team_tl3_original_20261018.zip is bound by TL3_IMPORT.json to SHA d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569. It has fixed members distilled_s5@ep1 (0.34), fttall_tlead_s5@ep3 (0.33), and fttall_tlead_s6@ep3 (0.33). Its embedded description calls the parent teachers all-data, but actual training, teacher, initializer and sequence ancestry is unknown until the source artifacts are inspected. It may be reused as an official team asset under standing authority, but cannot serve as a clean-fold teacher/reference or support held-out claims until that ancestry is verified. This uncertainty does not block the known component-clean F0 recipes.

The target definitions are anonymous indicators of future movement for i0; their exact meaning is undisclosed. Docs describe p/v/dp/dv feature families as price-, volume- and trade-related, but provide no venue, calendar/exchange timestamps, or corporate-action/adjustment method. step_in_seq is only an ordinal within a sequence. Do not infer exchange, timestamp, event, instrument, or corporate-action facts from names or public score.

## Invalid claims

- Test-time supervised adaptation on t0/t1, adaptation keyed to is_scored, or any claim that the callback observes either is invalid.
- Looking ahead to later rows, carrying state across sequences, ordering by seq_ix, or updating only rows believed to be scored violates the documented contract.
- Calling TL3 or any train+valid-fitted endpoint a clean holdout without full source/teacher/initializer ancestry is invalid.
- Calling exposed fold comparisons pristine holdout evidence, turning a public score into local expected WP, or inferring timestamps/venue/corporate-action treatment from anonymous feature names is unsupported.
- A quick single-fold/single-seed result may be an engineering or feasibility screen; it is not the project's prescribed scientific comparison. The project comparison protocol requires 30 paired fold/seed cells per arm, with fold and ancestry exposure explicit.

## Evidence limits

No complete dataset scan or output prediction replay was performed. Validation labels and masks are available to the offline grader, but were not read here as payloads and are never callback inputs. The audit establishes the documented interface and reviewed provenance metadata; it does not determine undocumented organizer intent about online learning or reconstruct TL3 ancestry.

