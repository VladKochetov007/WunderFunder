# Old Z4 F0 student CP1 export gate R5

R5 preserves the R4 scientific/export contract byte-for-byte and binds it by
SHA-256. The R4 validator failed on the actual producer because split and
exposure fields are nested under `fit_recipe_pins`, `grade_semantics`, and
`export`; R5 reads those declared locations. R5 also loads the actual
`training_state.pt`, validates its epoch-2 continuation boundary, optimizer
state and finite tensors, reconciles update diagnostics, and verifies that its
16 model tensors are bitwise identical to CP1.

The R4 producer receipt remains historical evidence. This R5 packet is still
code-only: it does not launch the export or grade. Root must review the passing
R5 gate and actual export profile, then create a distinct R5 admission binding
both `STUDENT_EXPORT_CONTRACT_R4.json` and `RUNNER_CONTRACT_R5.json` before the
runner may execute.

`CP1_TERMINAL_GATE_R5_PRODUCER_TEST.json` is an earlier successful R5 check
from before the optimizer-state and bitwise CP1/training-state equality checks
were added. `CP1_TERMINAL_GATE_R5_PRODUCER_TEST_R2.json` is the successful
actual-producer check before the validator self-hash field was added; the
current implementation result is `CP1_TERMINAL_GATE_R5_PRODUCER_TEST_R3.json`.

From the repository root, the bounded CP1 gate command is:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=1200000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python \
research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/oldz4_student_cp1_gate_r5/validate_student_export_r5.py \
checkpoint-gate --project-root /home/vlad/development/wunder/wnn_connectome_starterpack \
--receipt-out research/campaign_2026-10-06/bandit_continuation/runtime/receipts/oldz4-r5-f0-s3-student-export/CP1_TERMINAL_GATE_R5.json
```

The only export runner is `run_student_export_r5.py`. It imports
`validate_student_export_r5`, requires the R5 gate schema and unique attempt
ID `oldz4-r5-f0-s3-student-export`, and validates the new root admission's
base-contract hash, runner-contract hash, checkpoint-gate hash and CP1 hash
before importing the exact bound trainer or starting the full-valid predictor.
It still checks the actual imported H230 loader helper with
`WNN_LOADER_WORKERS=0`; the scientific model, predictor, full-valid path and
fixed fold-0 grading semantics come from the unchanged R4 contract.

After root admission, the runner invocation is:

```sh
WNN_LOADER_WORKERS=0 WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% \
WNN_THREADS=1 WNN_MEM_MAX=6800000000 WNN_RUNTIME_MAX=1800s \
scripts/limited .venv/bin/python \
research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/oldz4_student_cp1_gate_r5/run_student_export_r5.py \
--project-root /home/vlad/development/wunder/wnn_connectome_starterpack \
--checkpoint-gate research/campaign_2026-10-06/bandit_continuation/runtime/receipts/oldz4-r5-f0-s3-student-export/CP1_TERMINAL_GATE_R5.json \
--root-admission research/campaign_2026-10-06/bandit_continuation/runtime/receipts/oldz4-r5-f0-s3-student-export/ROOT_EXPORT_ADMISSION_R5.json \
--export-receipt research/campaign_2026-10-06/bandit_continuation/runtime/receipts/oldz4-r5-f0-s3-student-export/EXPORT_VALIDATION_R5.json
```

The limits and invocation are a frozen handoff template, not a resource
admission; root must bind the actual profile receipt in the R5 admission.

The frozen expected prediction is `(1873, 20000, 2)` float32 at
`cache/pred_oldz4_h230_student_f0_s3_r3_valid.npy`. Export and scoring remain
separate root-admitted steps. Fold 0 is previously inspected development
evidence, not a fresh holdout or public result.
