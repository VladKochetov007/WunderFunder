# Native device comparison R4 contract

R4 preserves the R3 native TPU FP32 recipe and R5 GPU control binding. It adds
resource admission between the actual full-shape profile and any long fit. It
does not change the model, optimizer, loss, data, membership, precision, or
comparison.

The FP32 profile must report memory statistics for every actual TPU device
after the synchronized batch-32, chunk-2000, input-width-211 WMSE and Huber
profiles. For each device, the gate compares `bytes_limit` with the larger of
`peak_bytes_in_use` and `bytes_in_use`. Missing device count, capacity, usage,
or full-shape profile evidence blocks fitting while preserving the profile
receipts.

Before fitting, the runtime also reads host `MemAvailable`, cgroup memory
current use and the effective finite ceiling (`memory.high` and `memory.max`,
or the cgroup-v1 equivalent), and free bytes on the output filesystem. The
root admission must bind positive byte reserves for device free memory, host
available memory, cgroup headroom, and disk free space. A finite cgroup ceiling
must meet its reserve. If the cgroup sample explicitly reports no finite
ceiling, the gate records that basis and enforces the host-available reserve;
missing or incomplete cgroup samples block fitting. R4 does not infer reserves
from the laptop or reuse local RAM limits for Kaggle TPU capacity.

The outer worker repeats host, cgroup, and output-filesystem checks while the
matched fit phase is active. A failed or unavailable live sample records a
stop receipt and terminates the exact packet child process group. The gate
does not raise limits or contact a provider. Root admission still separately
controls push and fit authorization; a passing profile alone cannot start the
fit.

The fit membership remains 350 real whole-component training sequences, with
the final 30-real-sequence batch padded by two zero-weight rows, and 468 fixed
evaluation sequences. The source ancestry and shared artifacts remain bound
to the completed, root-reviewed R5 GPU export-recovery contract.
