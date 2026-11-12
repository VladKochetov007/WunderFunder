# Server handoff: fixed TL3 + LSTM CPU ZIP

Status: **PACKAGED_NOT_EXECUTED**. These commands are for a Linux x86-64 server,
not the Mac. No GPU or training is needed. The prepared archive is a candidate,
not a validated submission. Do not upload before reviewing the receiver outputs.

## Transfer

Copy `cpu-receiver-handoff-20261110-r2.tar.gz` to the server using your SSH alias:

```bash
scp cpu-receiver-handoff-20261110-r2.tar.gz USER@SERVER:/physical-disk/wunder/
```

On the server, choose a physical-disk directory with at least2GB free for
receiver outputs/scratch in
addition to the existing5.62GB validation file and Docker image storage.
Keep Docker image space separate from the output allowance. Do not use
`/tmp`, `/dev/shm` or a RAM-backed volume. A cgroup-v2 Docker host is required.

```bash
cd /physical-disk/wunder
tar -xzf cpu-receiver-handoff-20261110-r2.tar.gz
cd cpu-receiver-handoff-20261110-r2
sha256sum --check SHA256SUMS
export WNN_DELIVERY_ROOT="$PWD"
export WNN_VALID_DATA=/absolute/path/to/valid.parquet
export WNN_CPU_ID=0  # Replace with one available CPU allowed on this host.
```

The complete validation source must be the canonical file: 5,615,509,128 bytes,
1,873 groups of 20,000 rows, 3,528,273 scored rows. The receiver hashes the
10,686,926-byte Parquet footer plus its 8-byte trailer and checks the pinned
identity in INPUTS.json. This footer check is not a full hash of all data pages.

## Build the receiver on the server

The Docker build downloads packages; subsequent model execution has no network.
It uses Python3.11, CPU Torch2.9.1, NumPy2.2.6, SciPy1.15.3 and PyArrow18.1.0.
Torch is only used by the receiver's LSTM reference check, never by the ZIP.

```bash
packet=research/submission_candidates/tl3-lstm-complement-20261110
docker build --platform linux/amd64 \
  -f "receiver/$packet/Dockerfile.receiver" \
  -t wnn-cpu-receiver:20261110 receiver
export WNN_IMAGE_ID="$(docker image inspect --format '{{.Id}}' wnn-cpu-receiver:20261110)"
printf '%s\n' "$WNN_IMAGE_ID" > receiver-image-id.txt
```

Keep this exact image ID and CPU for both stages. The launcher supplies one
logical CPU and one-CPU quota, 16,000,000,000-byte RAM/no-swap limit, no GPU
mounts, no network, read-only image and persistent writable output/scratch.
The image is a reproducible receiver approximation, not the portal's exact
installed wheel inventory. The reference portal CPU is AMD EPYC9224; timings
on another CPU are a proxy, even under the same one-core limit.

## 1. Bounded profile and parity

```bash
bash launch_server.sh profile
cat outputs/profile/profile.json
```

By default this processes two predetermined complete sequences (first/last groups) and bounded
state/parity diagnostics. Set `WNN_PROFILE_SEQUENCES=8` before both commands for
a wider timing sample. The timeout is 30 minutes. It checks native ABI,
finite2-vector outputs, hidden-state updates on warm-up/unrequested rows,
A-B-A and explicit resets, suffix causality, exact blend arithmetic and the
LSTM against one-step/chunked Torch references at absolute tolerance1e-4.
The recipe, source manifest and ZIP are externally hash-bound; dependency and
native member bytes are checked. CPU, memory, versions and timing are recorded.

Proceed only if the command succeeds and `profile_passed` is true.
Before the full run, inspect callback time and estimated full-validation cost;
if it is incompatible with the two-hour bound, investigate that bottleneck first. Review the
per-row time before spending time on full validation. A profile receipt does
not approve quality or the 60-minute whole-test limit. It always says
`ready_to_submit:false`. Failures preserve logs in outputs/profile; diagnose
and retain that attempt before using a new delivery directory for a rerun.

## 2. Full validation of the exact callback

```bash
bash launch_server.sh full
cat outputs/full/full.json
```

The launcher requires the passing profile receipt for the exact same ZIP,
recipe and source inputs. Full validation uses every row in all1,873 sequences,
the official callback/mask and global clipped weighted Pearson accumulator.
It saves only the3,528,273 scored rows: served-base and actual callback
predictions, targets, seq_ix and step_in_seq. These five aligned NPY arrays
retain inputs for paired sequence-level uncertainty analysis. Their raw payload
is119,961,282 bytes (about120MB), versus599,360,000 bytes for the previous
two full-row prediction arrays (about80% less; headers excluded). Every raw row
still advances both recurrent models. WP, paired point difference, resource
readings and receipt hashes are returned. Timeout is two hours;
this is a validation allowance, not the competition's one-hour inference limit.

`WNN_TEST_ROWS` is deliberately unset: the official data overview does not
publish the hidden test cardinality. Do not substitute validation rows or invent
a bound. If the organizers supply a defensible upper bound, set it **before**
full validation:

```bash
# export WNN_TEST_ROWS=ACTUAL_JUSTIFIED_INTEGER_BOUND
```

Without that bound the runtime gate stays `PENDING_TEST_ROWS`. With it the
receiver gives a conditional extrapolation with25% margin and a3,600-second
limit; it still does not guarantee portal runtime. Model loading is included in the projection; other initialization overhead and
CPU differences need review. Positive point delta is a screening gate, not a
significance claim; inspect per-sequence results/uncertainty before release.
Neither stage uploads the ZIP or starts a subsequent fit automatically.

## Decision and next experiments

- Reject this deployment if the serving change removes the LSTM benefit, parity
  fails or the measured cost is incompatible with the portal envelope. Preserve
  the result; do not retune mixture weights on repeated portal feedback.
- If quality survives and the resource envelope is justified, refresh submission
  history and review the same immutable ZIP for one later upload. Historical
  teacher-cache ancestry is incomplete and remains explicitly disclosed.
- Subsequent research should address the measured bottleneck: native LSTM serving
  parity/performance if latency binds; new clean-fold matched teacher/student
  experiments if quality binds. Do not infer a better teacher or a first-place
  result from this previously exposed +0.000125 research difference.

Final ZIP: `kotiki-tl3-w18-lstm10-cpu-20261110.zip`,14,011,657 bytes.
SHA-256: `766bab1f3205697819af994acbb54cfe60682651977f393d76bec69608a75e3e`.
The handoff also contains BUILD_RECEIPT.json, DEDUP.json, INPUTS.json, recipe.json,
STATIC_REVIEW.md, all receiver sources, the exact utils.py and SHA256SUMS.
