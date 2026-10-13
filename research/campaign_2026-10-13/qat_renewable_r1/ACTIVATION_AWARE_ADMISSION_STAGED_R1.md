# Activation-aware QAT staged admission — 2026-10-13

**Status:** prepared for root review; no compile, Torch test, or GPU run was performed for this note. Root retains all execution and GPU-admission ownership. **Experiment:** E-QAT-A8-BACKEND-20261013-R1; H-QAT-A8-20261013. This is an operational cost plan, not result evidence.

The source implementation is frozen at worktree commit `60daafba7b6bb9fc55b62ecf3f656b48eb90726e`. Static inputs checked were `src/wnn/activation_qat.py` (SHA-256 `9ccdfb864222ef77c139cd64ca1adf420e6939dea652fd37e59a93c484a397b7`), `scripts/profile_activation_qat.py` (`9d5680059f6d6104e0f2ca82a2ab93941cc44ad45c77b77de951c74b6944c901`), and installed Torch `torch/_higher_order_ops/scan.py` (`6bb980d8ceecf0455a6f45400105fc1d3e524c6c7c1f93af7543aea6ad92d87b`).

## What static source says about scan memory

The installed scan backward is a reverse scan (`ScanAutogradImpl.call_backward`, around lines 859–998). It initializes gradients for scan `additional_inputs` as zero tensors and accumulates each timestep's gradient into that carried tensor using `prev + cur`. Thus the feared `T × selected-parameter-count` stack of recurrent-weight gradients is not how this implementation accumulates those gradients. It does retain scan forward intermediates and returns per-timestep gradients for scan inputs; activation and graph/compiler memory still scale with sequence work and are not measured here. The previous 3.8 GB parameter-gradient estimate should therefore not be used as a predicted allocation, and this source inspection does not establish a peak-memory bound.

The prior B16/C1000 fused/eager profiles are comparison anchors only: FP32 peaked at 4,952,092,672 host bytes / 2,169,503,744 observed VRAM bytes; recurrent-W4 at 3,982,680,064 / 2,173,698,048; mixed W4/W8 at 4,528,828,416 / 2,173,698,048 (`LOCAL_10PCT_EXECUTION_R1.json`). None used the compiled scan backend. For scale, one FP32 `[T,B,3H]` input-gate tensor at B16/C1000/H256 is 49,152,000 bytes per layer and one `[T,B,H]` sequence tensor is 16,384,000 bytes per layer; saved intermediates, backward temporaries and compiler memory are additional and unknown.

## Bounded profile ladder

1. **Compiler/backend smoke, first.** Existing `synthetic_smoke --mode both` exercises compiled scan forward/backward in both activation modes with B2/T32/F112/H64/L2. It uses random inputs, a synthetic square objective, no optimizer updates, no actual cache/data/checkpoint, and is not representative of actual model or sequence memory. Admit only as a 4,000,000,000-byte host-reservation / one-CPU GPU smoke, with a 300-second child timeout. At the live preflight, `parent memory.current + unused active-worker RAM reservations + 4,000,000,000` must be at most 17,000,000,000 bytes. Require the root's separate aggregate VRAM gate at no more than 5,000,000,000 observed bytes and the controller's configured free-memory reserve. The 4 GB is a containment reservation for this small smoke, not an assertion that the compiler has a measured 4 GB peak. If the compiler exceeds it, stop and retain the first failure; do not retry blindly.

   Use a fresh physical cache directory and verify its filesystem with `findmnt` before launch. The synthetic mode does not enforce the physical-cache check itself. Controller stdout/stderr is durably captured at `research/campaign_2026-10-06/bandit_continuation/runtime/logs/qat-a8-compile-smoke-r1.log`; the printed JSON distinguishes `synthetic_smoke_complete` from a backend failure. The separate profile receipt is a status marker only.

2. **Actual-recipe B16/C1000 paired backend profile, only after root reviews stage 1 and remeasures the aggregate after cache handoff.** The frozen profile wrapper starts two fresh trainer subprocesses, `scan_a8` and `scan_fp32`, each using the same actual CP5 parent, fitting subset, actual first B16 batch, validated teacher/propensity caches, frozen WMSE/auxiliary loss and fresh AdamW path. Each mode is bounded to one compile/warmup update and one measured update on rows 0:1000, for four optimizer updates total. It is a backend profile, not a fit-fidelity or quality comparison.

   Proposed worker containment is 8,000,000,000 host bytes and one CPU. This is a conservative *proposed cap*, not measured scan headroom: it gives about 3.05 GB over the largest same-shape eager anchor, but compile/intermediate memory is still unknown. Do not admit unless the controller preflight satisfies `current + unused active reservations + 8,000,000,000 ≤ 17,000,000,000`, the GPU is otherwise idle, the same ≤5 GB aggregate VRAM gate holds, and the root explicitly admits this stage. If the preflight does not fit, wait for cache handoff or use a separately reviewed smaller actual-data profile; do not lower caps or launch on a projected overcommit. Child timeout is 900 seconds with a 10-second TERM-to-KILL grace. The wrapper records each subprocess log and partial mode receipt; a failure in mode two must leave mode one's evidence intact.

   The scan compiler's persistent cache is `/home/vlad/.cache/torch/inductor-qat-activation-aware-r1` on physical storage; re-check `findmnt -T` immediately before the run. One CPU is set for the durable worker and the profile source sets Inductor compile threads to one. Both modes share this disk cache, but each subprocess still gets independent model/optimizer state and compile timing. The weight-cache invalidation, changing represented-weight fingerprints, finite-after-update checks, and paired sampler/checkpoint/subset/normalization/cache/source identities are required by the frozen wrapper receipt.

## Exact root-owned commands

The following commands are prepared, not run. Paths under `wnn-agent-worktrees` refer to the frozen source above; artifact and runtime paths intentionally remain in the main project checkout. `scripts/limited` bounds only the short controller process; the durable worker owns its separate reservation.

### Stage 1 — small synthetic smoke

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=500000000 WNN_RUNTIME_MAX=120s \
/home/vlad/development/wunder/wnn-agent-worktrees/qat-activation-aware-20261013/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
/home/vlad/development/wunder/wnn-agent-worktrees/qat-activation-aware-20261013/scripts/bandit_resources.py \
--root /home/vlad/development/wunder/wnn-agent-worktrees/qat-activation-aware-20261013 \
--state /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/runtime launch \
--attempt qat-a8-compile-smoke-r1 --kind gpu --cpus 1 --reserve-bytes 4000000000 \
--profile ACTIVATION_AWARE_ADMISSION_STAGED_R1.md \
--contract /home/vlad/development/wunder/wnn-agent-worktrees/qat-activation-aware-20261013/research/campaign_2026-10-13/qat_renewable_r1/CONTRACT.json \
--admission-timeout 30 -- \
timeout --signal=TERM --kill-after=10s 300s env \
TORCHINDUCTOR_CACHE_DIR=/home/vlad/.cache/torch/inductor-qat-activation-aware-smoke-r1 \
PYTHONPATH=/home/vlad/development/wunder/wnn-agent-worktrees/qat-activation-aware-20261013/src \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
/home/vlad/development/wunder/wnn-agent-worktrees/qat-activation-aware-20261013/scripts/profile_activation_qat.py \
--profile-kind synthetic_smoke \
--worktree-root /home/vlad/development/wunder/wnn-agent-worktrees/qat-activation-aware-20261013 \
--mode both --device cuda \
--receipt /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-13/qat_renewable_r1/ACTIVATION_AWARE_SYNTHETIC_SMOKE_R1.json
```

### Stage 2 — actual first-chunk paired profile

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=500000000 WNN_RUNTIME_MAX=120s \
/home/vlad/development/wunder/wnn-agent-worktrees/qat-activation-aware-20261013/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
/home/vlad/development/wunder/wnn-agent-worktrees/qat-activation-aware-20261013/scripts/bandit_resources.py \
--root /home/vlad/development/wunder/wnn-agent-worktrees/qat-activation-aware-20261013 \
--state /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/runtime launch \
--attempt qat-activation-aware-profile-r1 --kind gpu --cpus 1 --reserve-bytes 8000000000 \
--profile ACTIVATION_AWARE_ACTUAL_PROFILE_PLAN_R1.json \
--contract /home/vlad/development/wunder/wnn-agent-worktrees/qat-activation-aware-20261013/research/campaign_2026-10-13/qat_renewable_r1/CONTRACT.json \
--admission-timeout 30 -- \
timeout --signal=TERM --kill-after=10s 900s env \
TORCHINDUCTOR_CACHE_DIR=/home/vlad/.cache/torch/inductor-qat-activation-aware-r1 \
WNN_PROJECT_ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack \
PYTHONPATH=/home/vlad/development/wunder/wnn-agent-worktrees/qat-activation-aware-20261013/src \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
/home/vlad/development/wunder/wnn-agent-worktrees/qat-activation-aware-20261013/scripts/profile_activation_qat.py \
--profile-kind actual_recipe \
--project-root /home/vlad/development/wunder/wnn_connectome_starterpack \
--worktree-root /home/vlad/development/wunder/wnn-agent-worktrees/qat-activation-aware-20261013 \
--mode both --device cuda \
--receipt /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-13/qat_renewable_r1/ACTIVATION_AWARE_ACTUAL_PROFILE_R1.json
```

## Failure and collection policy

After any nonzero exit, reservation refusal, timeout or worker termination, first read the exact attempt at `runtime/attempts/<attempt>.json`, its `runtime/logs/<attempt>.log`, and any mode-specific `ACTIVATION_AWARE_ACTUAL_PROFILE_R1_{a8,no_a8}.log` already written. Then use the existing `bandit_resources.py status` and, only if the supervisor is terminal/uncertain, its `reconcile` path. Keep a partial profile and terminal controller receipt; classify setup/compile/resource/backend failure separately from quality, which remains `not_evaluated`. Do not reuse an attempt ID or run a retry until root has identified the first failure, reconciled controller state, and explicitly re-admitted a diagnosed next step. No full-data fit, long-sequence fallback, or scientific QAT-gain claim is authorized by either profile.
