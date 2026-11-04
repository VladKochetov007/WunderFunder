# Independent execution critic R1

**Review timestamp:** 2026-11-04 11:04 UTC  
**Scope:** read-only execution-path review of the W18 L4 cohort controller, the frozen R2 panel, process ownership/deadline helper, and fit/export handoff. No source or test files were changed. No SSH, GPU, profile, or fit was run.

## Review outcome

The current R2 panel passes local structural validation and has the intended eight cells, four distinct recipes, and paired seeds. The current controller stops remaining cells after a worker error. I found no immediate blocker in the frozen panel's cell count, recipe matching, fixed endpoint, CPU affinity, or tracked output paths.

Before profile admission, root must finish the separate execution-source binding and source/data/parent admission receipt. The numeric source manifest is intentionally numerical-only and does not bind the new controller modules. The controller also treats admission and profile-review receipts as external authorization: it checks their existence/hash, but does not validate their approval schema or compare their contents to the panel itself. The root-owned gate must perform those checks.

## Findings

### 1. Error-triggered child shutdown can wait for the full remaining cell timeout

**Location:** `src/wnn/multistage_search_owned_command.py:103-105`

When a progress callback fails because the shared STOP event was set, the deadline reserve is reached, or a budget check fails, the helper calls `terminate_owned_process(child, grace_seconds=remaining)`. Here `remaining` is the time until this child’s hard timeout; for a fit it can be as long as three hours. If the child does not exit on SIGTERM, the helper can delay SIGKILL and cohort collection until that time expires.

The normal timeout path is bounded: it sends SIGTERM near the per-child deadline and SIGKILL at the deadline, after checking that the recorded PID, process birth ticks, and process group still match. The progress-error path retains the same ownership check but uses an unnecessarily long grace.

**Smallest correction:** use a fixed short termination grace (for example, at most 10 seconds) on callback failure, then recheck PID/birth/PGID before SIGKILL. This is a slow-cleanup contingency rather than a foreign-process safety issue; ordinary Python children usually exit on SIGTERM.

### 2. Controller bytes require a separate execution-source binding

**Location:** `NUMERIC_SOURCE_BINDING_R1.json`; `FROZEN_PANEL_R2.json:identity.source_binding_sha256`

The numeric binding manifest contains 315 entries, including `scripts/train_rnn.py` and `scripts/predict_run.py`, but it contains none of `scripts/w18_l4_quality_20261104/{contract,runner,runtime}.py`. This is consistent with a numerical-only manifest, but the controller code is not covered by that hash.

Root reports that `EXECUTION_SOURCE_BINDING_R1.json` and a published source commit will be included in the actual admission receipt. Treat that as a required pre-profile gate: it should hash the controller modules/entrypoint and be attached to the detached launch and phase receipts. The panel’s current numerical source hash alone is not proof of controller identity.

### 3. Profile telemetry reports sampled VRAM, not exact peak allocated/reserved

**Location:** `scripts/w18_l4_quality_20261104/runner.py:194-221`; `src/wnn/training/profile_results.py:15-103`

The W18 profile logs include step throughput and finite loss/gradient update evidence. The controller samples `nvidia-smi` every 30 seconds and records free memory/utilization. The canonical trainer only emits exact CUDA peak allocated/reserved values from its activation-backend evidence branch, which is not enabled in this strict-FP32/no-quantization cohort.

Use the minimum sampled free VRAM as coarse operational headroom, and do not report it as an exact peak allocated/reserved measurement. If exact peaks are needed for admission, use a separately receipted device monitor or add a small non-quantized profile measurement before fitting.

### 4. The controller exports predictions only; grading remains a separate required phase

**Location:** `scripts/w18_l4_quality_20261104/runner.py:117-153`

The export phase calls `predict_run.py` for the fixed epoch-1 checkpoint and writes full-valid feature-only predictions. It does not calculate the official pooled Pearson score or the fixed 90/10 blend. Root confirms the canonical grader is a separate one-shot step. Require its row-group, prediction-sidecar, scored-mask, checkpoint, and blend-weight checks before making any quality claim.

## Verified behavior and recipe table

| Area | Required match | Reviewed behavior |
|---|---|---|
| Cell allocation | One control plus three mechanisms, seeds 1001/1002 | R2 has 8 cells, 4 distinct normalized recipes, and both seeds per recipe. |
| Precision and epoch | FP32, no training quantization, fixed epoch 1 of 2 | Panel args use `--precision fp32`, `--epochs 2`, fixed policy; the contract rejects non-`none` QAT/activation backends. Export explicitly loads `model_ep1.pt` with `--epoch 1`. |
| Chronological validation | Official complete valid; no random/seq_ix substitute | Fit args bind the official split manifest, `train` source, and `--full-valid`; export requests full-valid feature-only groups. The canonical grader must verify exact row order/scored mask. |
| Matched candidates | One factor per config, identical recipe across paired seeds | Current R2 has four unique configuration recipe hashes; within each configuration the paired seeds differ only by name/seed. The frozen `--batch-order seeded` setting is shared across all cells. |
| GPU/CPU use | One fit per physical GPU; bounded disjoint CPU sets | Panel has eight unique L4 UUIDs and non-overlapping eight-core affinities; `command_prefix` checks affinity against the controller’s allowed CPU set and applies `taskset`. |
| Deadline | One 12-hour wave from first profile; reserve last 45 minutes | Durable state captures the deadline once. Each profile/fit/export timeout is bounded by the remaining wave before reserve; the owned helper uses a monotonic per-child deadline. |
| Process ownership | Signal only owned descendants | Child starts a new session; receipt records child PID, birth ticks, PGID, argv, cwd and output path. Stop/timeout uses PID/birth/PGID checks before process-group signals. |
| Disk/RAM | 8 GB output cap, 3 GB free reserve, 400 GiB owned RSS | Runtime accounting includes runtime, profile/fit directories, prediction final/partial/sidecar paths and deduplicates files by device/inode. Budget checks run before launch and during worker progress. RSS is aggregated only for PGIDs in current phase receipts. |

## Checks executed

- Loaded and validated `research/experiment_drafts/w18-l4-quality-20261104/FROZEN_PANEL_R2.json` through the actual `load_panel` function: **pass**, SHA-256 `951cefd42a35b86e1d408bdbfbe5e55d165e765977173ec4abef9e0fe4f02bf8`.
- Confirmed 8 cells, 4 unique configuration recipes, seeds 1001 and 1002 for each configuration, and common `--batch-order seeded`.
- Reviewed process creation, receipt writing, timeout, progress-callback error, and signal paths in the current local source. No real child was launched: this workstation is Windows, the production helper explicitly requires POSIX/procfs, and SSH/receiver execution is outside this critic assignment.
- Did not run fit, profile, GPU, SSH, or test-suite commands.

## Reviewed snapshot hashes

These hashes identify the local snapshot reviewed and should be refreshed if the builder changes any implementation after this note.

| File | SHA-256 |
|---|---|
| `FROZEN_PANEL_R2.json` | `951cefd42a35b86e1d408bdbfbe5e55d165e765977173ec4abef9e0fe4f02bf8` |
| `scripts/w18_l4_quality_20261104/contract.py` | `0d4cff02ec8bef29f39db4b807d9f9265d5fe7d9ebbd8827aa89c485122092c8` |
| `scripts/w18_l4_quality_20261104/runner.py` | `0ad6d5eec1ab788fc6de1b1dd5b4a74875eafb7838abec7ed6ddcc54572a87c3` |
| `scripts/w18_l4_quality_20261104/runtime.py` | `619096e9876795c26da85973b32847af3c30b15956652916922cdb7cd4b4062c` |
| `src/wnn/multistage_search_owned_command.py` | `7769b9fac5b702c7ec3dfd74cc941e397f3d5897c7413363ebda9848d380d292` |
| `tests/test_w18_l4_cohort_contract.py` | `e72b5f57619af6a54f7c17845f9e2483e04108529ca53335cf7c3dc0c6a8eb2a` |

The root-owned execution-source binding/admission and canonical grader are outside this review file’s scope and remain preconditions for actual execution and any result claim.
