# Fold1 GRU384 seed3 recovery R3

R2 terminated at `01:28:11Z` when the attempt supervisor sampled `9,000,030,208`
bytes against its `9,000,000,000`-byte reserve. The R2 systemd unit peak was
`8,999,997,440` bytes; every attempt, parent and hierarchical OOM/kill delta
was zero. This is a bounded supervisor-reserve failure, not a CUDA OOM or a
scientific failure. Root preserved the event and reclaimed only campaign
file-cache under the exclusive GPU lock; the exact receipt is pinned in the R3
packet.

R2 atomically saved its epoch-boundary `training_state.pt` after epochs 0, 1 and
2. The state hash is `de4823c79d32efe8f9394f5f45823a25687539d598e928ccdd138e038103874d`,
`next_epoch` is 3, and it includes model, AdamW, OneCycle, FP16 scaler, data
loader and RNG state. R3 therefore replays the final epoch from the start of
that epoch. It preserves the four-epoch scheduler horizon and all original
source, data, membership, optimizer and loss inputs; the partial uncheckpointed
R2 fourth-epoch prefix is discarded by restoring the boundary snapshot.

The R3 request uses `--name bd26_w384raw_s3_f1_resume_r3`, resumes from the
R2 checkpoint, and keeps `--epochs 4`. Its attempt envelope is 2 CPUs,
9,000,000,000-byte reserve/maximum, 5,000,000,000-byte observed VRAM ceiling,
zero swap, 500 MB output cap, 2 GB free-disk floor and 7,200-second wall limit.
Root requested an exact-unit `MemoryHigh=8,000,000,000`-byte drop-in before
dispatch; the terminal auditor requires the live unit to report that property.
No fit is launched by this package. The prior `STOP_ADMISSION.json` is retained
and its exact hash/content are recorded in the pending request; root must issue
and record any resume/admission before dispatch. The request remains pending
root review and admission.

Run the actual checkpoint and source preflight under the shared support lock
and the current CPU/RAM support cap:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=2000000000 WNN_RUNTIME_MAX=120s \
python scratch/fold1_gru384_successor_20261009/recovery_r3/fold1_gru384_recovery_gate_r3.py \
  --packet-sha256 PACKET_SHA prepare --check-only
```

After root review, a request receipt can be written with `prepare --output
.../RESUME_REQUEST_R3.json`. Root creates a separate authority whose fields
match `verify_authority()` in the gate, then uses the existing campaign resource
controller to launch the exact prepared command. `audit` is for the terminal
R3 output and independently requires successful unit state, zero OOM counters,
real `MemoryHigh`/`MemoryMax`/CPU/swap properties, exact source pins, the R2
state ancestry and the frozen fold1 output verifier.

`launch_seed4_after_r3.py` is the separate conditional successor. It requires a
successful R3 audit and an explicit `--launch-seed4` invocation before it uses
the original seed4 command and the original two-fit allocation. It never
changes the frozen R1/R2 gates or launches R3 itself.

The reviewed candidate packet SHA-256 is
`c80c8b2946ad3d6102ac8c74de723c796553c4425f6aaf851160f3a819280526`; the gate
source is `ad462d2c2888e1f3acb5c366dec946cfa25545ecf72dad7b5046a7e65a4f37ae`.
The prepared, not-admitted request receipt is
`RESUME_REQUEST_R3.json` (SHA-256
`5665744651fed633c6040f0b9c927a656388535ae9661ce784920b54c7de2e90`). Its
command hash is
`a5641445619a8b3bf04f47ec5701d55ff8231dcc38aa7d4b5d2a0d209d37d293`. The
focused R3 tests pass (`3 passed`), and Ruff check passes for the gate,
conditional seed4 wrapper and tests. The request records the existing STOP
barrier; only root can record its supersession and admit R3.
