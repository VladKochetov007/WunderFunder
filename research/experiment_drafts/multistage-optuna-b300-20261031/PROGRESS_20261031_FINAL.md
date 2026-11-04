# Completed B300 wave, 31 October 2026

All scoped training and GPU scoring have finished before the hard deadline of
05:41:09 UTC. The wave used all eight B300 devices for the auxiliary study;
four fresh full-primary refits later used GPU0-3. No additional fits are queued.
Foreign NIER and MPS processes remain untouched.

Forty scientific attempts produced **39 configured-final model artifacts and one
partial GALAM**: 15 base models, eight fine-tunes including two earlier pilots,
12 paired distillation/control models, and four full-primary fresh refits.
All 56 bounded engineering profiles were used. The fit cap of48 was a ceiling,
not a promise of48 completed jobs. The original ledger retains nine
`terminal_unverified` exits and one stale GALAM `running` state; model-artifact
completion and present process absence are separately verified. Unknown outer
exit codes remain unknown; no old fit was resumed.

## Full-primary results

| Candidate | Official valid WP |
|---|---:|
| Fixed fresh TL3 reference |0.692377425|
| Equal four-member ensemble |0.684699354|
| Frozen Stage4 direct-four weights |0.684304429|
| Fresh LSTM192x1,8 epochs |0.682520594|
| Fresh GRU384x2,4 epochs |0.682425941|
| Fresh regularized GRU384x2,4 epochs |0.681210508|
| Fresh Transformer384x4,4 epochs |0.676479699|
| Earlier frozen eight-member auxiliary-fit ensemble |0.683227279|

The best new mixture is **0.007678071 below TL3**. The baseline was not beaten.
The four fresh refits and TL3 use the same10,325 eligible fitting train members,
zero fitting valid members, and all1,873 official-valid sequences /3,528,273
scored rows. Training recipes and budgets still differ. TL3 is a previously
evaluated fixed reference, not rescored during this collection. The earlier
eight-member ensemble fits only8,260 members and has an unmatched data budget.
No new public score or submission is claimed.

The two four-member vectors and the four configurations/seeds were frozen before
fresh fitting. No primary-valid labels optimized weights or selected epochs.
Primary valid is development-exposed reporting, not independent confirmation.
The extra8,260/2,065 train-only holdout remains a nonchronological proxy; its
identifiers do not establish time order. Official primary membership and all
normalizer/teacher/initializer ancestry remain clean under ADR0044.

Stage2 improved four of six main continuations versus their parents on the
exposed inner proxy; extra training epochs were not controlled. Stage3's six
matched pairs had positive inner WP deltas from0.000774 to0.004964, across
different seeds and five distillation coefficients. These are not six
replications of one recipe. One inner-selected pair improved primary WP from
0.6758899455 to0.6782913914. CPU Optuna completed24 ensemble trials using only
train-derived control predictions; the original23 records were preserved.

## Artifacts and checks

[Final results, CSV, PNG/SVG, frozen weights and raw process receipts](https://github.com/VladKochetov007/WunderFunder/tree/results/tabdata-multistage-b300-20261031/research/experiment_drafts/multistage-optuna-b300-20261031/results/full-primary-final-r1)
include native train/valid trajectories for all four full-primary refits.
Train diagnostic curves use16 fitting sequences/all rows after warmup; valid
uses all1,873 official sequences/is_scored. Their absolute gaps use different
sampling and masks. All earlier stage graphs remain in the endpoint reports
and dashboard at http://127.0.0.1:18777/.

An independent NumPy/PyArrow verifier read raw validation labels and masks,
checked all prediction hashes/order, and reproduced all six new outputs with
maximum absolute metric error7.993605777301127e-15. Both mixtures reconstructed
bit-exactly in FP32. This verifies fixed-prediction arithmetic, not independent
training replication or public transfer. Full feature-Parquet hashing belongs
to the original admission, not this label-only check.

[Four fresh full-primary configured-final models](https://github.com/VladKochetov007/WunderFunder/releases/tag/tabdata-pr53-full-primary-checkpoints-20261031-r1)
and [33 auxiliary configured-final models with replay sources](https://github.com/VladKochetov007/WunderFunder/releases/tag/tabdata-pr53-multistage-checkpoints-20261031-r1)
are research prereleases. Every uploaded asset passed an actual GitHub download
SHA256 check. Two historical pilots and the partial GALAM are excluded from
those37 published models. Native per-epoch checkpoints remain on the node;
configured-final endpoints are not relabelled as validation-selected best.

The final collector R2 preserves the complete R1 source archive and adds an
independent GNU timeout watchdog for every scorer/blend, final-model/native
terminal SHA binding, and live owned-process-group qualification. Eight focused
CPU tests with18 subtests passed before execution, alongside existing scorer
checks; all four actual GPU scorers and the CPU blend returned exit0.

[PR53](https://github.com/VladKochetov007/WunderFunder/pull/53) receives results
as SemyonEpanov. `GPU_RELEASE_RECEIPT.json` records no tabdata GPU clients and
preserves foreign NIER/MPS. Large raw JSONs are losslessly gzipped with original
and compressed hashes. Immutable scientific source capsules and old failures
are retained.
