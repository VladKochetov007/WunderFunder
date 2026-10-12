# Why recent processes were killed, and how we prevent repeats

2026-10-12 user reinforcement: OOM is avoidable wasted work. Future admission targets17decimalGB committed working memory, leaving1GB below the existing18GB allowance; hard20GB/no-swap limits are unchanged. Count currentparent+unusedworkerreservation+newtaskbound, and verify host available memory before starting material allocations. Existing healthy jobs drain normally. Size discovery uses bounded metadata/real-writer fixtures or representative profiles, never intentional OOM.

The R25 paired-blend scorer completed without OOM but measured1,402,112KiB peak RSS (about1.436GB), too close to its1.5GB supportcap; hierarchical max events increased7 and cannot alone attribute the boundary. Preserve the successful result and current counter observations. For this same legitimate paired analysis, future support budget should include measured arrays plus margin (approximately1.9GB), admitted only if the aggregate17GB target holds; otherwise use streaming/smaller complete panels or wait. This is not permission to raise limits after a malformed parser/staging failure. Do not rerun a completed score to manufacture a lower peak.

2026-10-12. Scope: support work for the continuing Challenge2 campaign. This is
not a reason to stop healthy GPU jobs or raise the campaign's resource limits.
The user requested this reflection while the clean F0 student fit was running.

## What the evidence says

| Incident | Actual boundary and cause | Scientific consequence |
|---|---|---|
| Python PIDs4143414/4144286,10:43Z |650MiB support scopes. The first NPY reader interpreted magic bytes as a version/header length and requested a huge read; kernel anonymous RSS was about664MiB. Both failures are preserved; the second stdin command's exact source must remain attached by its owner. | No model or objective comparison completed. |
| Python PID52127,11:19:30Z | A1.69GB cache was copied into RAM-backed `/tmp`. Kernel reported about641.29MB file/shmem and7.08MB anonymous memory, with zero dirty/writeback bytes. The bounded scope exhausted memory; exit137 was confirmed by the kernel OOM record. | Asset staging failed before any provider upload or fit. |
| Current control P1 | Own5.5GB maximum; peak about4.95GB, observed VRAM about2.64GB after an epoch transition, no worker max/OOM/kill. Its first epoch completed7460 attempts,7457 updates and3 AMP skips, then continued. | Healthy training. Intermediate checkpoints are not the planned final epoch5 result. |

The staging failure violated an existing instruction against large `/tmp`
files. The header failure bypassed existing format-validation primitives.
These were avoidable implementation/preflight errors, not a lack of permission,
insufficient aggregate resources, or evidence against the model family.

The earlier competition `SCORE:TIMEOUT` is a different failure: a60-minute CPU inference deadline, not these local memory kills. Current C completed36m36s. Serving candidates need measured one-core runtime headroom; increasing research RAM cannot repair a scorer timeout.

A low process RSS did not protect the staging task: cgroups charge RAM-backed
files too. `memory.current` is the total to budget; `memory.stat` explains its
composition. Shmem is included in file accounting, so do not add it twice.
Likewise, a parent's hierarchical OOM counter includes child scopes. Attribute
kills to the kernel's exact cgroup/PID before blaming the live training worker.
The initial broad “dirty cache” diagnosis was corrected to tmpfs by these
measured counters.

## Immediate repairs already made

- The NPY inspection now verifies magic/version and bounds header length before
  reading. Metadata inspection does not read prediction payloads.
- Kaggle asset staging uses a short private physical `/home` directory. Its two
  immutable cache arrays are same-filesystem hardlinks, verified by inode/device,
  content identity and size. No chmod on a hardlink and no mutation of the source.
- The failed staging source, logs, exit and partial-file identity are preserved.
  Only the regenerable incomplete `/tmp` copy may be removed with a cleanup
  receipt; original datasets, caches and weights remain intact.
- The healthy P1 GPU job continues. Its own counters are distinguished from the
  staging scope's OOM and from MemoryHigh reclaim/throttling counts.

## Mandatory preflight for the next support task

1. Classify the work: header/metadata, streaming copy/hash, array analysis,
   checkpoint inspection, or GPU fit. Set an explicit task limit and timeout
   inside the user campaign slice. Never fall back to the wrapper's old defaults.
2. For staging, resolve the physical path and filesystem before writing large
   files. Reject `/tmp`, `/dev/shm` and RAM-backed filesystems for model/data
   staging. Prefer hardlinks for frozen same-filesystem inputs. If copying is
   necessary, use bounded chunks, disk writeback and cache advice; never a whole
   large-file `read_bytes()` or a temporary-directory default.
3. Derive array bytes from the actual header and dtype, including the header.
   Use the established reader/validator rather than another handwritten parser.
   For analysis, budget conversions, mask selection copies, labels, gradients,
   library imports and working arrays—not just the source file sizes. Use memmaps,
   streaming and separate calibration/evaluation phases where appropriate.
4. Check fresh aggregate memory and CPU, including unused headroom reserved by
   the healthy active worker. Use `memory.current` and the proper **user** systemd
   manager. Stay within5CPU/18decimalGB working/20GB hard/no swap and5decimalGB
   observed VRAM. Small metadata tasks retain small caps; legitimate larger
   analysis gets a separately justified bound within the same aggregate.
5. Exercise the actual consumer against an actual producer fixture before
   spending on the task: NPY header, cache metadata, checkpoint format, receipt
   schema, source-qualified groups and final run name. Synthetic fixtures alone
   do not establish compatibility when a producer already exists.
6. Derive pins from the trusted frozen manifest rather than manually retyping
   hashes or guessing receipt fields. Use absolute MAIN Python paths and the
   correct working directory in isolated worktrees. Compare the complete launcher
   argv and environment, including `WNN_PROJECT_ROOT`, not just scientific flags.
   The P2 omission resolved `/root/wnn` before CUDA; its corrected successor keeps
   the same validated P1 state and adds P1’s exact environment. Freeze the final packet,
   then validate that exact packet.
7. Observe actual PID, phase, heartbeat and resource counters after launch.
   MemoryHigh events mean reclaim/throttling; exit137 alone does not prove OOM.
   Use kernel/systemd evidence to distinguish OOM, timeout and manual termination.

## On the first failure

Preserve the command, source, stderr, unit/cgroup identity, limits, peak memory,
OOM deltas and last phase. Diagnose the first failure before retrying. A second
attempt needs a concrete changed mechanism and a new identity: bounded parsing,
physical disk staging, hardlinks, streaming or smaller working arrays. Do not
blindly repeat or increase the cap to conceal a parser or staging error.

Reuse completed training/checkpoints after a downstream failure. Never refit,
relax ancestry/metric checks, delete irreplaceable data or stop a healthy worker
to make an infrastructure failure disappear. Record the correction and its
actual producer fixture; no new general controller or permission layer is needed.

## Ownership and acceptance

- Root owns allocation, independent review and these admission conditions.
- Each implementer owns task-specific filesystem/byte/schema checks and the first
  failure's diagnosis. Reuse existing tools; do not redesign the infrastructure.
- The cloud builder must seal its physical-stage/hardlink and actual mounted
  payload fixture receipts before an upload or push.
- The numerical analyst must freeze component membership, scope and a peak-memory
  estimate before loading payloads, then run a bounded shape/resource profile.
- The fit operator validates actual full training state before continuation and
  retains model/optimizer/scheduler/scaler/RNG/loader state and exact ancestry.

Acceptance is concrete: corrected commands pass their actual-producer fixture,
then complete within the existing limits. The absence of a future OOM is not
promised from a checklist; actual phase/resource observations remain required.

Evidence: `ops_continuation_20261011/monitor_20261012_1044_r21/NPY_HEADER_OOM_*`
and `ops_continuation_20261011/monitor_20261012_1107_r22/KAGGLE_STAGE_TMPFS_OOM_*`
under the active diversity campaign; actual P1 attempt/log in continuation runtime.
