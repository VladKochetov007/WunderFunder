# H233 V2: source-ancestry and grade-mask successor

V2 supersedes the reviewed but unadmitted V1 recipe only to bind the own-teacher fit provenance and to make evaluation membership explicit at export. The exact V1 recipe, source, test, and README bytes are preserved in `H233_EXECUTED_SOURCES_V1/`; do not edit that snapshot. The scientific treatment remains unchanged.

The loader pins both teacher `provenance.json` files by SHA-256, requires selected epoch 3 and the pinned teacher checkpoint, checks the same fold-0 manifest, and compares each teacher's train and fit-development sets with the exact frozen split sets. `CampaignSplit.assert_fit_groups` then rejects any evaluation-component overlap. Teacher evaluation-row-group sets are also checked against the same manifest. The V2 recipe contains the ordered evaluation group IDs and their digest, plus the digest of the excluded fit-development group IDs.

Validation export carries all valid rows in source-cache order to preserve the prior prediction interface. It writes `evaluation_group_mask.npy` aligned to `row_groups` and `selected_mask.npy`, which is already that group mask AND the supplied scored mask after the 99-row warm-up. **A fold-0 held-out grader must use only this `selected_mask.npy`; it contains no selected fit-development rows.** Export provenance binds both masks, their hashes, row order, evaluation group IDs, and counts (468 evaluation, 1,405 fit-development). This is exposed fold-0 evidence only.

The V2 prepared commands remain worker argv templates, not admission. Root retains source review, profile, fit, export and grading ownership; all recipe authorization flags are false. No GPU/profile/fit/export or score is claimed by this code correction.

```sh
PYTHONPATH=src .venv/bin/python scripts/train_conditional_readout.py profile \
  --warmup-chunks 2 --steps 8 \
  --output research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/profiles/H233_C1S3_B8_PROFILE_V2.json

PYTHONPATH=src .venv/bin/python scripts/train_conditional_readout.py fit \
  --run-name bd26_h233_c1s3_rank32_f0_v2 \
  --output-dir artifacts/runs/bd26_h233_c1s3_rank32_f0_v2

PYTHONPATH=src .venv/bin/python scripts/train_conditional_readout.py export \
  --checkpoint artifacts/runs/bd26_h233_c1s3_rank32_f0_v2/model_ep1.pt \
  --split valid \
  --output-dir research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/exports/H233_C1S3_E1_VALID_V2
```
