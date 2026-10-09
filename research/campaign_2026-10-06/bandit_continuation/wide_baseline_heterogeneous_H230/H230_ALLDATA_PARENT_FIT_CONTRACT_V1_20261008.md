# H230 all-data parent fit contract V1

- Attempt ID: `h230-all-data-parent-fit-b16-ep3-v1-20261008`
- Status: `awaiting_root_recipe_review_and_fresh_resource_admission`; root review and fresh admission are required before launch.
- Recipe: GRU256×2, seed 3, B16/chunk 2000, fp16, 4 epochs, fixed epoch 3, own `c20261003_combo_s3_f0@ep3` initializer.
- All supplied train and valid groups are fitting data. The exposed validation groups are not held out, and this fit makes no heldout score claim.
- Worker reservation: 8 GB RAM, 2 CPU cores, zero data-loader workers, no swap; aggregate parent limits remain 18 GB high / 20 GB max.
- Latest measured parent current was 11,561,598,976 bytes, so it plus the requested reserve exceeds the 18 GB high limit. The contract is not launchable on that snapshot; fresh controller admission must pass.
- Runtime cap: 10,800 seconds owned by the exact `/usr/bin/timeout` command. The one-update profile is not used as a throughput estimate.
- No scientific argument, checkpoint, prediction, evaluation or training code was changed.
- Machine-readable contract: `research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_ALLDATA_PARENT_FIT_CONTRACT_V1_20261008.json`
