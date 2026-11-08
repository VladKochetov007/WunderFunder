# Transformer profile and fit receiving notes

The exact two-cell argv manifest is
[`TRANSFORMER_TEACHER_PANEL_R1.json`](../../../../scripts/l4_teacher_capacity_20261104/TRANSFORMER_TEACHER_PANEL_R1.json).
It is a command record, not source/data admission. Its null source, data, and fit
member hashes must be populated from the root-approved receiving admission before
dispatch. The staged tree must be `/data2/lc-epanov_s/tabdata/runtime/l4-teacher-capacity-20261104-r1/source-tf-r1`; keep it separate from the frozen GRU `source-r1`.

## Profile dispatch

Root verifies the current physical UUIDs for historical slots6/7 before use.
The prior panel maps these slots to GPU-004cdb0f-3bf7-0003-702e-0e2fb87215e8
and GPU-cd069633-c451-769e-3727-cb1fe310e407. Do not dispatch if current
inventory differs or either slot has an unexpected owner. Use one visible UUID
per process, eight assigned CPU threads and 48GiB host RAM per worker. Keep the
combined GRU plus Transformer allocation within 64 CPU cores, 400GiB RAM, the
6.5GB output ceiling, and the 3GB `/data2` free-space reserve.

Read the existing
`/data2/lc-epanov_s/tabdata/runtime/l4-teacher-capacity-20261104-r1/GRU_STATE_R1.json`.
Do not create or reset it. Its `first_profile_started_epoch` and
`hard_deadline_epoch` govern both families. Each stage timeout is the smaller
of the stage cap and `hard_deadline_epoch - now - 2700` seconds. A profile
dispatch requires the shared state and root-approved source/data admission to
exist, and all request/log/receipt/output paths in the manifest to be absent.

Run each `profile_argv` once through
`wnn.multistage_search_owned_command.run_owned_command`, with `progress` omitted:

- `cwd=source_root`
- `env=base_environment(gpu_uuid, source_root)` plus the bounded worker values
  already used by the GRU runner
- `log_path=R/logs/transformer/profiles/<cell_id>.log`
- `receipt_path=R/receipts/transformer/profiles/<cell_id>.json`
- `owner=current_process_owner()` extended with panel, cell, config, argv, split,
  source and data identities
- `stage=0`, `trial_number=GPU slot`, `mode=l4_teacher_capacity_transformer_profile`
- `gpu_key=gpu_uuid`, `output_dir=profile_output`, timeout at most1800seconds

Hold `hold_gpu_slot(R/control, gpu_uuid)` for the full call. The command writes a
trainer `profile_receipt.json` inside its output directory; the owned-command
receipt is a separate process/exit/ownership record. A profile runs exactly50
real-data optimizer updates and does not evaluate valid or save a scientific
checkpoint. It reaches the configured 50-step diagnostics callback.

## Receipt admission before fit

The sequence CLI's profile admission is exact identity equality, not a loose
name match. Its trainer receipt must have:

- `receipt_schema=wnn.sequence-profile.v1`, `status=profile_complete`, then
  independent review may set `approved=true`;
- exact `identity` equality, including `config_sha256`, `data_sha256`,
  `source_sha256`, `source_files`, `runtime`, `world_size=1`, model parameter
  counts and optimizer contract;
- `updates=50`, positive `forwarded_rows`, `real_rows`, `valid_rows`, positive
  elapsed time, finite nonnegative objective losses, `errors=[]`,
  `error_count=0`, and `scientific_checkpoint_written=false`;
- one `memory_by_rank` entry for rank0 with positive allocated/reserved/device/
  host memory values; `peak_reserved_bytes <= device_total_memory_bytes`, and
  device UUID/total memory equal `identity.runtime.devices[0]`.

The independent review receipt must bind the panel SHA, approved source/data
admission receipt SHA, both profile process-receipt SHAs, both trainer profile
receipt SHAs, exact configs/argv, split and fit membership hashes, source/data
closures, actual GPU UUIDs, and profile memory/finite-update evidence. It must
approve both profiles before either fit starts. Preserve each trainer receipt's
identity and evidence when changing only its `approved` field to true.

## Fit dispatch

Use each cell's `fit_argv` once, with the same physical UUID as that cell's
profile, `stage=1`, `mode=l4_teacher_capacity_transformer_fit`, a separate log
and process receipt under `R/logs/transformer/fits/` and
`R/receipts/transformer/fits/`, and timeout clamped by the shared deadline minus
the collection reserve. Fit preconditions are: both process profile receipts
exited0 without timeout, both trainer profile receipts independently approved,
the independent combined review approved and hash-bound, and the shared
deadline still has more than the 45-minute collection reserve.

The existing fit monitor records the fixed32-sequence train curve and a full
1,873-sequence official-valid report after every epoch. `monitor_events.jsonl`
also samples gradient/activation/resource diagnostics every50 steps and ordinary
training telemetry every500. Validation is exposed development evidence. The
fixed endpoint is zero-based epoch3; no score claim is independent confirmation.
