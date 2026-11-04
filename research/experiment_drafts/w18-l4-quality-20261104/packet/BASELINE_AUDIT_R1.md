# Independent Baseline, Ancestry, and Evaluator Audit R1

Date: 2026-11-04  
Scope: independent eligibility review for the W18-L4 quality contract. Read the current contract, ADR0044, historical TL3/W18 recipes and results, and the received node evidence. No SSH, fitting, profiling, inference, source edits, or science approval.

## Disposition

The strongest eligible cached train-only comparison is the fixed canonical TL3 blend at pooled official valid WP 0.6923774245652725. Its W18 member alone scores 0.6914165059769162. Current receiving evidence closes the previously identified ancestry gap: archive and byte audits pass; all 17 run fit sets equal the exact chronological fit membership; the W18 mixed-target arrays contain only the 10,325 eligible train groups; actual direct teacher predictions, producer weights, sidecars and declared closures are hash-bound; and R2 closes the remaining FBC/TLead teacher-prediction and model_ep3 bytes.

The distinct epoch-3 files are now resolved, not a mismatch: W18 provenance records its initializer as FBC model_ep3.pt SHA256 840966894144bca5aafbf0a8e21ad0daab07a8503ba80e30d96668ef1bc3a496. The W18 mixed-target cache uses FBC model.pt SHA256 759165147d99b33034b74f4af63a30e8a202dafc78fd5e5699d05b6cfe9bef90. Both are separate artifacts listed at selected epoch 3 in the FBC provenance, and both actual files were hash-checked. Keep the distinct role and hash binding; do not require these serialized files to match.

Recommendation for the new 8-fit cohort: use the frozen current W18 control recipe at seeds 1001/1002 for paired per-seed deltas. Keep the cached canonical TL3 blend as the strongest fixed historical anchor, not as a seed-matched control. Its validation is exposed and it does not estimate training variance.

FBC is an eligible simpler single-model comparator, but it is still distilled from fin256_s3 and fin256_s4. Its WP is 0.6913549829232479, below W18 and canonical TL3. It does not avoid the teacher ancestry question; the current R2 checks close its two teacher inputs.

## Official data boundary and exact fit membership

ADR0044 requires official earlier train -> complete later valid, whole sequences, no order inferred from seq_ix. Current source audit and chronology manifest verify:

- Physical train: 10,607 row groups/sequences, train parquet SHA256 4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81.
- Eligible fit_train: 10,325 sequence IDs after excluding 282 train_component_overlap groups. Preserve this separately from the full physical train count.
- Complete valid: 1,873 row groups/sequences, valid parquet SHA256 2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782.
- Scored valid rows: 3,528,273.
- Chronological manifest SHA256 0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c.
- All 17 historical model provenance fit_row_groups sets compare equal as sets to manifest fit.train; all record 10,325 fit groups and zero fit_valid groups. Different ordered digests reflect run sampling order, not membership differences.
- W18 main_source_train has shape 10,325 x 20,000 x 2; families_train has shape 10,325 x 20,000 x 8. Both target row-group lists equal manifest fit.train. Both valid teacher target arrays have zero rows. Source prediction sidecars cover all 10,607 physical train IDs and are keyed to the target rows; there is no positional prefix slice.

The current source audit hashes all train/valid parquet bytes. Current train_index.identity.json SHA256 is d7093810c08a8aa2248a27edb8d6b66f3be7eb203b7cc12bd20508ee9beb704e; valid_index.identity.json SHA256 ff9833d27e723dcc21adc2593b1fcbfec038cf8a8c2e465488ca82b66937e2ab. Their corresponding index archive hashes are train 9ffac9a45c943cdaa2b8deb75986fad27489f163fe2b40919095637877334d0b and valid 89d2da2589d224a0de9f20e8aa06551b6e817540fddd8fba6eaeb147ce04d1b7.

The 1,873-sequence validation set has been repeatedly inspected and selected against. These are exposed development comparisons, not fresh confirmation.

## Complete official valid scores

Metric: global_wp_clip2_v1, official pooled clipped-label-magnitude-weighted Pearson over both targets on every official scored valid row.

| Fixed predictor | target 0 Pearson | target 1 Pearson | pooled WP |
|---|---:|---:|---:|
| W18 seed 5, fixed epoch 1 | 0.6878701879593053 | 0.6949628239945271 | 0.6914165059769162 |
| Tiedlead seed 5, fixed epoch 3 | 0.6878379819344286 | 0.6949925755598049 | 0.6914152787471167 |
| Tiedlead seed 6, fixed epoch 3 | 0.6868911592991259 | 0.6950066763906596 | 0.6909489178448928 |
| Canonical fixed TL3 blend | 0.6887196906212466 | 0.6960351585092984 | 0.6923774245652725 |
| FBC seed 5, fixed epoch 3 | 0.6878423809933999 | 0.6948675848530959 | 0.6913549829232479 |

Canonical coefficients were fixed as 0.34*W18 epoch1 + 0.33*Tiedlead seed5 epoch3 + 0.33*Tiedlead seed6 epoch3. No learned coefficients were fitted on primary valid.

An independent R3 full-array CPU rescore matched the production scorer for 25 supplied outputs: maximum absolute difference 5.551115123125783e-15 at tolerance 1e-10. Current valid parquet SHA matches the rescored input. This establishes arithmetic for those arrays, not significance, training variance, or public/private transfer.

Key bindings:

- W18 model SHA256 a66488e99ed49373722c206df0fa8fd97c179246a72369b69732bb32029baae6; prediction SHA256 642475ab1e26f18001312a8680dba621a0df079fe314c531eaf214308624aacb; provenance SHA256 3a67347d5c1133860c5cd3f910e296a0a5ee956aca54ec68290a763cf2946457.
- Tiedlead seed5 model SHA256 f447271a524fde75779f171c1e796cf9827fda6bad8e3525f05def466393a13c; prediction SHA256 1b42f30f61554286b68d15a5f640f346b94e988c69099f0f03fd7a4c4ab5aa3d; provenance SHA256 fae7831bb2549162fe0c5d90e0f97f94619e7f9fd391e8b9efb71ecb821f6340.
- Tiedlead seed6 model SHA256 ef654318599b835c4b69426b56680335d11605f0f0a82557d52bccfcc3a5ed41; prediction SHA256 9aa551910cc453bde7525a1a0e9bdabdf6f2909b0cd1a860d4454f36d85fd19e; provenance SHA256 ad95ccc860b89c2bdb8f508a3fe74a05176a9e3226e8b9126eb65bc518161a52.

## Actual W18 recipe and execution receipt

Frozen node recipe: research/experiment_drafts/heterogeneous-tl3-20261030/receiving/R1/w18-node-r1.json, SHA256 148cd202970215a76e25051d8975b106072f28a13cdfada15e47fd0f2731af5a. The run executed from /data2/lc-epanov_s/tabdata/runtime/heterogeneous-20261030-r1/source-r1 with the train-only project root /data2/lc-epanov_s/tabdata/runtime/launch-20261028/tl3-trainonly-20261028-r1.

Recorded argv:

    --name tl3c_w18final_distilled_s5 --seed 5 --hidden 256 --layers 2 --dropout 0
    --train-source train --train-seqs 10325 --eval-source dev --full-valid
    --split-manifest /data2/lc-epanov_s/tabdata/runtime/launch-20261028/data-preparation-r1/chronological_primary.json
    --precision fp32 --checkpoint-policy fixed --loss-norm chunk
    --sched onecycle --warmup-frac 0.05 --wd 0.01 --clip 1 --beta2 0.999
    --sync-free --fused-adam --pin-memory --grad-stats --save-training-state --train-eval-seqs 32
    --aux-next 0.3 --aux-struct 0.3 --aux-weight 0.5
    --batch 16 --chunk 2000 --epochs 2 --lr .0001 --loss wmse --optim adamw
    --init tl3c_fbc1k_s5@ep3 --inherit-init-preprocessing
    --distill-cache-root /data2/lc-epanov_s/tabdata/runtime/launch-20261028/tl3-trainonly-20261028-r1/w18-mixed-targets
    --distill-alpha .7 --distill-family-weight .1

The actual receipt reports exit 0, 1,283.6448 seconds, 12,900 successful updates, fixed endpoint zero-based epoch 1. W18 provenance binds initializer SHA256 840966... to FBC model_ep3.pt; inherited normalization hashes match FBC. The actual W18 checkpoint, provenance, predictions, request hash 3123d179944d012494af673a21d4f0d8331ccd6c9d745c1a7798bce9339e163a, command hash a18eb181ea1347c5d5fbf80744f4c9b1009e9512cd6aa48adfaaab52a8dd1678, and source aggregate 7abc9b7e18679bc39723178099dd3859e93c414556ed3b06d829d61a5e43d3a4 are in the receiving packet.

Declared W18 cache sources: fbc1k_s5 strong coefficient 0.30 at epoch3; ffull_s3/s4 full_finetuned coefficient 0.10 each at epoch1; frawW_s3ft/s4ft raw_wmse coefficient 0.125 each at epoch1; orgbase samefold_baseline coefficient 0.0625 at epoch3. Distill alpha .7, family weight .1. The mixed-target metadata SHA256 is f9e983f311589f3510263cf5f75dbfd113b799693d2e69e078989aa313469536.

## Parent ancestry and actual byte closure

The 49-file receiving package manifest has archive SHA256 0b51480d84fe12f572e49a703382ba03078119e4d41a5c7141d0c749d2fda130; all 49 local entries match manifest sizes and hashes. The R1 actual-byte audit passes all 45 listed files (audit SHA256 4976055a1849eb5642d92560a4971a31c0c35523d9106a7aea16cac540931bd1). R2 closes the 12 added prediction/sidecar/model checks and confirms three actual model_ep3 hashes (audit SHA256 6a5a551ce1fe919f73383dbc9ad7337cdaf6554f83bf4cbbb009f802c9dbaf3a).

The complete W18/TL3 ancestry is now pinned as follows:

1. W18 initializer: actual FBC model_ep3.pt hash equals the W18 provenance initial_checkpoint hash. The W18 cache's FBC teacher source uses separately pinned model.pt and its train prediction array/sidecar.
2. W18 teacher sources: all six train prediction arrays and sidecars pass, with exact checkpoint/provenance and epoch; their declared direct closure weights/provenances pass. W18 target arrays pass and their row-group list equals the exact fit set; valid target arrays are empty.
3. FBC ancestry: fin256_s3/s4 prediction arrays, sidecars, model.pt weights and provenance match FBC distill_provenance; R2 hashed the arrays and weights. Both are scratch roots without teachers or initializers.
4. Fine-tuned teacher ancestry: fin256ft_s3/s4 prediction arrays, sidecars, model.pt weights and provenance match both TLead distill_provenance records; R2 hashed these inputs and weights. Their fin256_s3/s4 raw epoch3 initializer weights also match the fine-tune provenance and R2 hashes.
5. Other W18 direct parents: ffull_s3/s4 inherit fmu128_s3/s4 fixed epoch3; frawW_s3ft/s4ft inherit frawW_s3/s4 fixed epoch3; orgbase is a scratch root. Their relevant fixed checkpoint bytes/provenances and W18 consumed train prediction bytes/sidecars pass the R1 audit.
6. Tiedlead seed5/seed6 final checkpoints and full-valid prediction arrays/sidecars pass R1. All 17 runs have the same fitting ID set and empty valid ancestry. No submitted full-corpus artifacts are in these chains.

R2's 12 checks include the fin256_s3/s4 and fin256ft_s3/s4 train prediction arrays, sidecars, and model.pt weights. Its three epoch-indexed checks are FBC model_ep3.pt and fin256_s3/s4 model_ep3.pt. All match the exact hashes recorded by their consuming/producer provenance. This closes the actual-byte gap previously called out in INDEPENDENT_REVIEW_R1.json.

## New cohort control and clean fallback

For paired per-seed comparison, use the current W18 common-control recipe at seeds 1001 and 1002, under the current admitted source and verified split. The old seed5 W18 and fixed TL3 ensemble are historical reference anchors, not matched training-seed controls.

If an engineering or membership gate rejects W18, use two fresh clean-scratch GRU256x2 controls at seeds 1001/1002 inside the existing eight-fit allocation. Historical clean roots fin256_s3 and fin256_s4 score WP 0.6882596673782984 and 0.6871360997574087; their simple prediction mean is 0.6886081210674475. These roots have no initializer or teacher ancestry and provide clean context, but their scores are exposed and not fresh matched controls.

Clean matched-root recipe:

    --name tl3c_clean_gru256_s{1001|1002}
    --seed {1001|1002} --hidden 256 --layers 2 --dropout 0
    --train-source train --train-seqs 10325 --eval-source dev --full-valid
    --split-manifest [current verified manifest]
    --precision fp32 --checkpoint-policy fixed --loss-norm chunk
    --sched onecycle --warmup-frac 0.05 --wd 0.01 --clip 1 --beta2 0.999
    --sync-free --fused-adam --pin-memory --grad-stats --save-training-state --train-eval-seqs 32
    --batch 32 --chunk 2000 --epochs 4 --lr .001 --loss wmse --optim adamw
    --aux-next 0.3 --aux-struct 0.3 --aux-weight 0.5
    --ema-teacher 1 --ema-decay .999 --eval-ema --since-weight 32 --since-floor .25

No initializer, distillation cache, teacher, or learned blend. Historical root endpoint is zero-based epoch3. Keep the new endpoint fixed from the baseline audit and actual-device profile before candidate outcomes. Do not refit hidden teachers.

## Evidence references

- Contract: research/experiment_drafts/w18-l4-quality-20261104/CONTRACT_R1.json.
- ADR: specs/adr/0044-chronological-primary-validation.md in the GALAM reconstruction worktree.
- Current receiving package: C:/Users/admin/Documents/Codex/reviews/tabdata-w18-l4-20261104/BASELINE_ARCHIVE_RECEIVING_R1.json and baseline-evidence/.
- Actual node byte audits: C:/Users/admin/Documents/Codex/reviews/tabdata-w18-l4-20261104/BASELINE_ACTUAL_BYTE_AUDIT_R1.json and BASELINE_ACTUAL_BYTE_AUDIT_R2.json.
- Canonical score: research/experiment_drafts/heterogeneous-tl3-20261030/receiving/CANONICAL_BASELINE_INDEPENDENT_R1.json, SHA256 9725a82eb94a23a8be3f92348ea761166c2cf5a4d59ff2dacc0f3ee81235dd73.
- Independent full-array rescore: research/experiment_drafts/heterogeneous-tl3-20261030/results/R3/INDEPENDENT_RESULT.json, SHA256 0bc067efd291af9937a6e4846ebe12e422ff4c03e0a5a71f0aa76ee7f006cb24.
- Independent review: research/experiment_drafts/heterogeneous-tl3-20261030/INDEPENDENT_REVIEW_R1.json.
