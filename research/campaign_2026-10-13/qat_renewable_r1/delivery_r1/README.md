# QAT delivery preparation R1

This packet freezes the source-qualified F0 seed-3 parent ancestry, whole-component fit subsets, and the matched one-epoch FP32 / uniform-W4 / mixed-W4-W8 command matrix for E-QAT-DEPLOYMENT-20261013-R1. It is preparation evidence only: this actor ran no fits, GPU profiles, predictions, or scores. Root owns source integration, admission, and execution.

The seed-3 parent is clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1 at epoch 5, an H256 two-layer GRU. Its checkpoint SHA-256 is e30577ff...776eda; full file/provenance hashes and the parent's complete recorded argument object are in [the ancestry receipt](ANCESTRY_VERIFICATION_R1.json). CP5 is a new optimizer continuation from c20261003_combo_s3_f0 at epoch 3, not a scratch fit: P1 starts from that initializer with fresh optimizer state, and P2/P3 resume the recorded optimizer-state chain. The combo initializer itself used two scratch-initialized F0 seed-3/4 teachers. Its complete checkpoint, provenance, prediction, and fitting/evaluation membership links are verified by the receipt.

CP5 used the clean-T3 homogeneous teacher cache cache/clean_t3_f0_h390_t3r5_a9_r2, alpha 0.9, auxiliary structural and next losses 0.3 each, propensity c20260929_prop_f0, and source-qualified F0. Cache metadata SHA-256 is 1e4c5c49...f2be3dc8; its pinned train/valid target files have 1,685,600,128 and 224,800,128 bytes. The validator checks the metadata and source checkpoint/provenance hashes, membership, file sizes, and the target SHA values recorded in metadata. It deliberately does not reread or rehash the two large array payloads. The F0 split hash is 09263ad4...f9316a3d: 10,535 fit-train groups, 1,405 fit-valid groups, and 468 fixed evaluation groups.

The frozen component manifests are nested at 10%, 30%, and 100% of the 3,148 eligible fit components, selected with seed 20261013. Each keeps complete sequences and both train/fit-valid partitions; all share the same full 468-group evaluation panel. Because CP5 and its teacher targets saw the full F0 fit membership, the 10% and 30% runs measure adaptation from that full-fit initializer, not small-data training from scratch. All F0 evaluation groups have prior exposure and remain a repeated development panel.

An equivalent F0 seed-4 H256 CP5 exists at clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1, initialized from c20261003_combo_s4_f0 at epoch 3; the receipt pins both checkpoints and source-qualified provenance. There is no equivalent clean-T3 F3 H256 CP5/cache: vt_f3_gru256 is a four-epoch FP16/Muon scratch run without that teacher cache, and vt_f3_h256_e3 is a separate three-epoch FP16/Muon scratch run. The October 13 F3 clean-T3 H384 teacher packet is preparation-only and has no completed checkpoint or target cache yet. Therefore this packet scopes seed-3/F0 fits only.

[The command matrix](RUN_RECIPE_R1.json) contains three 2-warmup/3-step actual-shape profiles, nine one-epoch fits, and fresh full-valid feature-only prediction plus fixed-panel scoring commands. Within each subset, all conditions share the same CP5 model/preprocessing initialization, batch/order seed, target/cache inputs, FP32 AdamW (max lr=1e-4, wd=.01), one-cycle schedule, batch 16 and chunk 1,000. Optimizer state is new in all three conditions. Fit commands save completed-epoch training state; there is no partial-epoch resume vector, so preserve an interrupted attempt and use a new run identity for any re-admitted retry. Profiles omit state saving because the trainer rejects profiling with continuation enabled. ema_teacher=0, swa_frac=0, and the final artifact is raw model_ep0.pt; no averaged checkpoint is selected. QAT is weight-only: recurrent activation quantization is not simulated.

The unchanged float continuation uses --qat none; uniform W4 uses --qat recurrent_w4; mixed W4/W8 uses --qat input_w4_recurrent_w8. Their policies are detailed in [the quantization spec](../QUANTIZATION_SPEC_R1.md). The recipe expects the subset-support commit from this worktree and QAT policy commit f0376b9c5e3671b94c80be94506153c347244842 to be integrated before running.

Run the bounded preflight after integration with the real input checkout and this packet:

    WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
    WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=180s \
    scripts/limited .venv/bin/python \
      research/campaign_2026-10-13/qat_renewable_r1/delivery_r1/validate_qat_delivery_r1.py \
      --repo-root /home/vlad/development/wunder/wnn_connectome_starterpack \
      --delivery-dir /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-13/qat_renewable_r1/delivery_r1 \
      --output research/campaign_2026-10-13/qat_renewable_r1/delivery_r1/ANCESTRY_VERIFICATION_R1.json

This preflight hashes small checkpoint/provenance/manifest files in bounded blocks, validates exact source-qualified membership and nested subset integrity, and checks that the recipe conditions differ only in their declared run, subset, and QAT fields. It does not read parquet rows, teacher-array payloads, or evaluation labels.

The prior TPU bridge failure is preserved in [the V7 registry-schema diagnosis](TPU_V7_REGISTRY_SCHEMA_DIAGNOSIS_R1.json). V7 reached host_admission and rejected the pinned portable registry because it demanded an absent dataset_ref; no actual-data hash, optimizer update, checkpoint reload, or prediction export ran. Root assigned a separate actor to fork V8. Preserve V7 and retain the independent dataset-slug, source hashes, and byte-length checks in that repair.

## Files

- [Ancestry verification tool](validate_qat_delivery_r1.py) and [verification receipt](ANCESTRY_VERIFICATION_R1.json) bind checkpoint chains, teachers/cache metadata, source split membership, subsets, and command vectors.
- [Matched run recipe](RUN_RECIPE_R1.json) gives exact profile, fit, prediction, and score argument vectors.
- [Frozen subset manifests](subsets/f0_s3/) preserve selected complete-component membership and its hashes.
- [TPU V7 failure diagnosis](TPU_V7_REGISTRY_SCHEMA_DIAGNOSIS_R1.json) preserves the unsupported-field failure and narrow V8 repair requirement.

The QAT plan and parent campaign record are in [the campaign entry point](../README.md) and [the experiment plan](../PLAN.md). This packet is indexed from [unified operations](../../../unified/operations/README.md).
