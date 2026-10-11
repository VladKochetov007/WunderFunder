# H382 local clean crossfold scratch confirmations (prepared, not admitted)

This packet stages four bounded comparisons on folds 2 and 3: seed 3 from scratch, h256 two-layer GRU, Muon-all, propensity power 0 with each fold’s own propensity model, batch 32/chunk 1000, wMSE/chunk normalization, LR 0.001, WD 0.01, onecycle, fp16. The e3 arm selects fixed checkpoint epoch 2; the e4 arm selects fixed checkpoint epoch 3. No teacher, initializer, early selection, or fit-time evaluation is used.

The four arms are `h382_muonall256prop0_f{2,3}_s3_e{3,4}`. Their exact trainer and export argument vectors are in `contracts_r1/`; `SEQUENCE_PLAN_R1.json` contains the durable one-at-a-time queue successors. Nothing is queued or launched by this packet. Root admission remains a separate step.

Fold 2 and fold 3 evaluation components have been inspected earlier in the campaign. Their fitting groups are disjoint from the declared evaluation groups by the copied source-qualified manifests, and each fold’s propensity model was fit against that same fold’s fitting membership. These runs provide crossfold confirmation, not a fresh global holdout. Export writes predictions for all 1,873 valid sequences; grading must subset to `evaluation.valid` (443 rows for fold 2; 505 for fold 3). Use `global_wp_clip2_v1` and the frozen evaluator.

The 50-file source snapshot matches the current parent source contract exactly and preserves the historical inherited snapshot note without using it as current authority. `INPUT_AUDIT_R1.json` records full content hashes for small/NPZ/propensity inputs, producer footer-plus-eight-byte hashes for the two parquet files, component-level membership checks, propensity ancestry and actual full-shape resource evidence. It does not claim a full parquet whole-file hash recomputation or a measured full-fold runtime.

The matched fold-1 reference is `r3_muonall256prop0_f1_s3_h380_e6`, fixed checkpoint epoch 5, scored 0.66885 on its already-exposed fold-1 evaluation; its exact terminal receipt is copied under `control_receipts/`. It is contextual evidence, not a clean-fold result or a score forecast for folds 2/3.

The measured h256/B32/chunk1000/fp16/Muon-all/prop0 profile used one loader worker and observed 9,500,770,304 bytes peak RAM and 3,570,401,280 bytes peak VRAM with no OOM. It was a 25-batch resource profile; it is not a speed estimate for any fold. Each fit contract reserves 10.5 GB from that receipt; root must refresh aggregate RAM, available GPU memory, and output space before admission.


The exact local profile contract and measured review are copied under `resource_receipts/`; their original external paths and hashes remain recorded in `INPUT_AUDIT_R1.json`.
