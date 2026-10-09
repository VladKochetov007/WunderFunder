# Root admission fields for the one recovery attempt

The R2 gate remains inert until root creates an allocation after the original
attempt is terminal. The request SHA covers the original terminal attempt
receipt, epoch-boundary state SHA/contract SHA, bounded dataset identity receipt,
exact resume command, source pins and resource envelope. The authority must
bind the resulting request exactly. This example is illustrative; use values
from the generated request.

```json
{
  "schema": "wnn.fold1_gru384_resume_authority.r2",
  "status": "conditional_exact_fold1_seed3_resume_authorized",
  "request_sha256": "<SHA-256 of RESUME_REQUEST_R2.json>",
  "packet_sha256": "<static recovery packet SHA>",
  "recovery_gate_source_sha256": "<gate source SHA from request>",
  "original_postfit_gate_sha256": "<frozen R1 gate source SHA>",
  "original_postfit_gate_packet_sha256": "<frozen R1 gate packet SHA>",
  "original_queue_sha256": "<frozen R1 queue SHA>",
  "original_allocation_sha256": "<frozen R2 allocation SHA>",
  "original_attempt_id": "fold1-gru384-s3-fit-r1",
  "original_attempt_record_sha256": "<terminal attempt receipt SHA>",
  "original_training_state_sha256": "<post-terminal state SHA>",
  "original_training_state_contract_sha256": "<request contract SHA>",
  "original_next_epoch": 1,
  "dataset_identity_receipt_sha256": "<bounded current identity receipt SHA>",
  "source_identity_helper_sha256": "<pinned helper SHA>",
  "resume_attempt_id": "fold1-gru384-s3-resume-r2",
  "resume_run_name": "bd26_w384raw_s3_f1_resume_r2",
  "resume_output_path": "artifacts/runs/bd26_w384raw_s3_f1_resume_r2",
  "resume_timeout_seconds": 14400,
  "command_sha256_canonical_json": "<request command hash>",
  "source_pins_sha256": {"...": "..."},
  "resource_request": {
    "cpu_cores": 2,
    "memory_max_bytes": 9000000000,
    "swap_max_bytes": 0,
    "observed_total_vram_max_bytes": 5000000000,
    "maximum_new_bytes": 500000000,
    "exclusive_local_gpu": true
  },
  "maximum_new_bytes": 500000000
}
```

The timeout, state fields, original receipt SHA, data-identity receipt,
request/command/source hashes, and resource limits must match the request
exactly. Dispatch uses the exact authority file SHA. The packet does not create
or authorize a seed4 admission.
