# Root admission fields for the one recovery attempt

The gate remains inert until root writes a separate allocation with all of the
following exact fields. The request file must first be created after the
original attempt is terminal. `status` is an explicit root admission; the
static packet and this document are not authority.

```json
{
  "schema": "wnn.fold1_gru384_resume_authority.r1",
  "status": "conditional_exact_fold1_seed3_resume_authorized",
  "request_sha256": "<SHA-256 of RESUME_REQUEST_R1.json>",
  "packet_sha256": "<static recovery packet SHA>",
  "recovery_gate_source_sha256": "<gate source SHA from request>",
  "original_postfit_gate_sha256": "<frozen R1 gate source SHA>",
  "original_postfit_gate_packet_sha256": "<frozen R1 gate packet SHA>",
  "original_queue_sha256": "<frozen R1 queue SHA>",
  "original_allocation_sha256": "<frozen R2 allocation SHA>",
  "original_attempt_id": "fold1-gru384-s3-fit-r1",
  "original_training_state_sha256": "<post-terminal state SHA>",
  "original_training_state_contract_sha256": "<request contract SHA>",
  "original_next_epoch": 1,
  "resume_attempt_id": "fold1-gru384-s3-resume-r1",
  "resume_run_name": "bd26_w384raw_s3_f1_resume_r1",
  "resume_output_path": "artifacts/runs/bd26_w384raw_s3_f1_resume_r1",
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

The values above are illustrative except for fixed IDs/envelope. The actual
`original_next_epoch`, state/contract/request/command/source hashes and chosen
timeout must match the terminal request exactly. The exact authority file SHA
is then supplied to the dispatch command. No seed4 authority is created by
this recovery packet.
