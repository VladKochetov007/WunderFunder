# B300 execution handoff, 18 October 2026

## Current status

Latest access check at16:42UTC: the user explicitly requested one new connection after reporting restored access. SSH returned its banner and password prompt, but rejected the single password submission with `Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)` and exit1. No authenticated session, remote command or B300 fit occurred; no retry is scheduled. [Latest access receipt](ACCESS_CHECK_20261018T1642Z.json). The earlier timeout below is retained as history; the current blocker is authentication, whose exact cause remains unknown.

The user authorized execution of the reviewed backlog on all eight B300s through one persistent SSH session. Root made one 90-second attempt to `epanov_s@10.14.14.26`; SSH timed out during banner exchange before authentication. No password was submitted, no remote command ran, no fit started, and no retry is scheduled. See [AUTHORITY.json](AUTHORITY.json) and [ACCESS_CHECK.json](ACCESS_CHECK.json). Remote access is the immediate blocker; current host allocation, UUID availability, topology, driver/CUDA/PyTorch stack, data paths, storage and scheduler remain unverified.

The previous read-only inventory reported eight visible B300s, about 275 GB per device, and three GPUs busy with vLLM processes. That snapshot is not an allocation or current availability receipt. Preserve those workloads. Do not start queue jobs from the example files: [allocation.example.json](../b300_queue_preparation_r1/allocation.example.json) has execution disabled and no UUIDs/resources, and [bound-arms.example.json](../b300_queue_preparation_r1/bound-arms.example.json) has no arms. Every PR in [queue.json](../b300_queue_preparation_r1/queue.json) has `launch_ready=false` and `fit_admitted=false`.

## First source-bound work after access

The only currently source-bound fit recipe is the registered rank-one labels-only comparison, F2/seed1001: fresh GRU H512×1 versus H384×2, two fits. Its exact pointers are [cluster handoff](../priority25_execution_r4/cluster/ISSUE_RANK01_F2.md), [recipe](../priority25_execution_r4/RANK01_RECIPE.json), [trainer argv and staged/resume commands](../priority25_execution_r4/RANK01_COMMANDS.json), source commit `a266b036b02e8e012c6eb31072291dad618a169d`, official grader `6f9fd05e491a9700cce1adf50e58b56cdc5e65bb`, recipe SHA-256 `9292b545fd088df40c4cc54066a745025ee4d30c2bf1574bdb4df341eb16937c`, and F2 manifest SHA-256 `56b0e2ecb6bbda1f34569206cbf2c276544c275bac025793ae450488cf5d17a2`.

Before any profile or fit, bind the actual clean source checkout, cluster-readable data and fold paths, input/cache hashes, output/scratch paths, two free physical GPU UUIDs, software versions and bounded CPU/RAM/GPU/time limits. Verify the exact data hashes in the cluster handoff; those expected values are `train.parquet=4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81`, `valid.parquet=2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782`, `train_targets.npz=ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9`, and `valid_targets.npz=dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955`, with target-index hashes and canonical fold binding from the same handoff. The old `/home/grinchenkoeg/...` paths, Beleriand UUID, 4-GB device cap, systemd wrapper and resource limits are not cluster bindings.

After those values are verified and two devices are actually admitted, run the two exact-shape profiles serially or on distinct admitted UUIDs. This is the RANK01 profile argv with only run name, fold and split path bound to F2; do not add `torchrun`:

```bash
cd "$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT/src"
export WNN_PROJECT_ROOT="$DATA_ROOT"
export NVIDIA_TF32_OVERRIDE=0

CUDA_VISIBLE_DEVICES="$WIDE_UUID" "$PYTHON" "$SOURCE_ROOT/scripts/train_rnn.py" \
  --name rr18_r01_wide_f2_s1001_profile_v1 --hidden 512 --layers 1 \
  --valid-fold 2 --split-manifest "$SPLITS/fold2.json" --device cuda \
  --arch gru --cell gru --enc linear --dropout 0 --train-seqs 10607 \
  --train-source train+valid --valid-train-fraction 1 --valid-repeat 1 \
  --n-valid-folds 4 --full-valid --eval-source vfold --batch 8 --chunk 2000 \
  --epochs 6 --checkpoint-policy fixed --checkpoint-epoch 5 \
  --no-eval-during-fit --seed 1001 --precision fp32 --loss wmse \
  --loss-norm chunk --batch-order seeded --deterministic --sync-free \
  --fused-adam --grad-stats --optim adamw --lr 0.0001 --beta2 0.999 \
  --wd 0.01 --sched onecycle --warmup-frac 0.05 --head-lr-mult 1 \
  --clip 1 --profile-steps 3 --profile-warmup 1

CUDA_VISIBLE_DEVICES="$DEEP_UUID" "$PYTHON" "$SOURCE_ROOT/scripts/train_rnn.py" \
  --name rr18_r01_deep_f2_s1001_profile_v1 --hidden 384 --layers 2 \
  --valid-fold 2 --split-manifest "$SPLITS/fold2.json" --device cuda \
  --arch gru --cell gru --enc linear --dropout 0 --train-seqs 10607 \
  --train-source train+valid --valid-train-fraction 1 --valid-repeat 1 \
  --n-valid-folds 4 --full-valid --eval-source vfold --batch 8 --chunk 2000 \
  --epochs 6 --checkpoint-policy fixed --checkpoint-epoch 5 \
  --no-eval-during-fit --seed 1001 --precision fp32 --loss wmse \
  --loss-norm chunk --batch-order seeded --deterministic --sync-free \
  --fused-adam --grad-stats --optim adamw --lr 0.0001 --beta2 0.999 \
  --wd 0.01 --sched onecycle --warmup-frac 0.05 --head-lr-mult 1 \
  --clip 1 --profile-steps 3 --profile-warmup 1
```

`SOURCE_ROOT`, `DATA_ROOT`, `SPLITS`, `PYTHON`, `WIDE_UUID` and `DEEP_UUID` are intentionally unset until measured from the actual admitted host; this block is the exact recipe binding, not evidence a remote run occurred. Require finite profiles, strict FP32, memory headroom and an ownership check before fits. Then follow the pinned stage-1/continuation argv in `RANK01_COMMANDS.json`; keep the fixed epoch-5 endpoint, original labels only, matched runtime and source-qualified F2 membership. The saved commands are exact argv references, not portable Beleriand launch wrappers.

The new `src/wnn/cluster_queue` can schedule independent **one-GPU** jobs after complete arm/source/input/resource/output bindings and a real enabled allocation. It rejects multi-GPU fits. The canonical trainer has no DDP process group, rank/device binding, sequence sharding, global objective reductions, collective metric path or rank-state resume. Eight cards therefore mean at most eight independent one-card jobs, never a `torchrun` wrapper around this trainer. `materialize` currently cannot run against the empty example arm file.

## Reviewed proposals and admission gates

The 16 original PRs plus issue #16 are inventoried below. “Current first gate” is the missing prerequisite before a distinct scientific cell can be materialized; every item still needs actual host/data/source/profile admission. PR submitter metadata does not prove idea authorship; all PRs list the same repository account.

| Item | Proposal / planned fits | Implementation and first missing gate |
|---|---:|---|
| PR1 | Causal Transformer output AR; 60 | Proposal only; implement real causal teacher, own-prediction AR/KV path, exporter and matched direct-attention control. No DDP. |
| PR2 | Cold GRU width/depth compute match; 150 | Existing GRU factory is reusable, but freeze five exact configs, matched measured update/compute budget and threshold; current five one-seed screens do not establish scaling. Distinct from R4’s near-parameter pair. |
| PR3 | GRU vs Transformer teacher families; 120 | Student trainer reusable; complete Transformer and fold-clean teacher/cache producer first, then identical student recipe and target calibration. |
| PR4 | Per-field age vs seven group clocks; 60 | Prior short profile/partial evidence only; add rank/optimizer-geometry confound controls and clean FP32 teacher/cache/parent lineage. |
| PR5 | Function-preserving GRU widening; 60 | Initializer parity and short profile exist; bind clean fold parents, fixed growth implementation, full-budget matched continuation and threshold. |
| PR6 | Frozen-parent output anchor; 60 | Resource profile only; validate FP32 parent export and separate teacher-function/calibration effects. |
| PR7 | Causal learned initial-state family; 120 | Subset recipes only; expand beyond 104 sequences, validate prefix timing/gradient semantics and rebuild clean fold parents. |
| PR8 | Grouped Transformer projection; 60 | Grouped starter only; add flat parameter-matched control, freeze groups/temporal positions and implement/profile Transformer. |
| PR9 | Past-only smoothing plus raw input; 60 | Helper only; wire feature/cache/stream path, fit coefficient on train only, test prefix/reset invariance. |
| PR10 | Trailing circle residual; 60 | Coordinate meaning unresolved; freeze pairs/window/epsilon and validate degeneracy, cancellation and causal-prefix behavior. |
| PR11 | Chronos/TimesFM transfer; 60 | No eligible checkpoint/adapter binding; resolve task-rule eligibility, license/hash/pretraining scope and causal multivariate target mapping. |
| PR12 | Clean heterogeneous teacher cohort; 60 | One F3 teacher/export exists; rebuild clean FP32 cohort per fold and bind each ancestry/cache and fixed mixture coefficient. |
| PR13 | Fixed checkpoint trajectory average; 60 | Two F0 averages constructed, export/grade pending; rebind fixed epoch/norm across all fold/seed endpoints. |
| PR14 | Causal innovation residuals; 60 | Prior pair blocked on cache schema; fix producer/consumer fixtures and bind fold-clean FP32 predictors/inputs. |
| PR15 | Causal geometry/discrete features; 120 / 15 theoretical waves | Four arms and feature ranges are specified; first gates are exact source/data manifests, clean source binding, 8-card profile and 5,371,320-update resource projection. No B300 update measurement exists. |
| PR17 | Causal smoother GRU teachers + distillation; 330 / 42 theoretical waves | No PyTorch smoke/profile/teacher registry/cache. Test full 112-sequence H1024×2 Python-loop trainer, bind fold-clean producer/checkpoint/cache identities and full-cost budget first. |
| Issue #16 | R4 rank-1 labels-only F2/seed1001 wide/deep; **2** | Closest runnable basis: exact trainer, recipe, grader, fold and target-cache hashes above. Still blocked by SSH/data/ownership/actual B300 profiles; one pair is not CP30. |

### Ranked-25 items and mapping

Ranks order hypotheses, not executable jobs. “Adjacent” or “scaffold” does not mean a PR implements the hypothesis; items without PRs require new implementation and matched controls.

| Rank | Idea | PR relation | Admission dependency |
|---:|---|---|---|
| 1 | Wide/shallow GRU convergence | PR2 exact family; R4 is a separate near-parameter recipe | Matched measured-compute recipe and full FP32 budget. |
| 2 | Causal prefix statistics/normalization with raw bypass | PR9/14/15/17 adjacent but distinct | Explicit raw → raw+statistics → normalized+raw control sequence. |
| 3 | Strong causal teacher to compact student | PR1/3/12/17 broad direction | Official student endpoint, clean teacher ancestry and calibration. |
| 4 | Intermediate teacher hidden-state hints | None | Freeze teacher/student and projected hint control; audit semantic query. |
| 5 | Output-component capacity | None | Parameter match and score reconstructed outputs. |
| 6 | Shared transforms with feature IDs | PR8 adjacent | Within-group sharing + ID/escape is unimplemented. |
| 7 | i1 reliability / inter-instrument lags | PR4 adjacent only | Compare no-i1, i1 and fixed lag basis. |
| 8 | Local causal attention plus recurrence | PR1 adjacent only | Same backbone with and without fixed left-causal window. |
| 9 | Covariance orientation | PR15 adjacent only | Causal covariance vs trace/activity controls. |
| 10 | Mamba-3 MIMO vs SISO teacher | None | Streaming/reset/causal parity, matched capacity and same student. |
| 11 | xLSTM matrix-memory teacher | None | Scalar/matrix memory match and same student transfer. |
| 12 | Nonlinear metric calibration | None | Low-cost calendar candidate, fit-only bins and frozen weighted objective. |
| 13 | New observable future queries | None | Define novelty beyond existing queries and censoring. |
| 14 | Soft write to primary hidden state | PR7 adjacent, different mechanism | Source-bound replay and soft-write vs dense recurrence. |
| 15 | Bounded group-robust objective | None | Fit-only group definition and one bounded objective change. |
| 16 | CUSUM slow-memory adaptation | PR4 adjacent only | CUSUM vs age/EW activity on fixed shifts. |
| 17 | Cross-stream event order | None | Antisymmetric area vs fixed lags and count/sum-preserving shuffle. |
| 18 | Masked current-feature reconstruction | None | Control mask, normalization, skip paths on same student. |
| 19 | CPC future input states | None | Same-sequence negatives, persistence control and same student. |
| 20 | Distribution/dependency distillation | PR3 scaffold only | Output-only KD vs fixed distribution summary; preserve two outputs. |
| 21 | ARTBP compensated truncated gradient | None | Exact BPTT toy parity first; high compute cost. |
| 22 | Content-based prior-state retrieval | None | Bound memory; recency/taps/shuffled-key controls. |
| 23 | Transition/cycle topology | PR15 adjacent only | Occupancy features do not establish cycle topology. |
| 24 | Adaptive fast memory teacher | None | Reset semantics and final fixed student vs ordinary teacher. |
| 25 | Input world model/internal rollouts | None | Beat persistence/one-step; control compounded error and student transfer. |

The separate `MAXIM-MINIMAL-FAITHFUL-KD` entry also remains blocked on exact recipe and fold-clean ancestry; preserve raw family targets, coefficient sum 0.8125, seed-specific parents and organizer provenance. Do not call ordinary KD a faithful replication.

## Queue order after access is restored

1. Root verifies worker identity, live device UUIDs/busy processes, topology, driver/CUDA/PyTorch/NCCL, physical storage/scratch, cgroup delegation, scheduler and actual data availability. Preserve foreign jobs and existing F0/F1 claims.
2. Complete the two-shape R4 F2 actual FP32 profile and issue the source/data/owner/resource receipt. If admitted, run only the two F2/seed1001 fits and fixed endpoint/grade from the pinned recipe; keep both arms matched.
3. Bind one implemented experiment at a time into complete immutable one-GPU arm templates, controls, dependencies, source/input hashes, finite budgets, expected artifacts and profile receipts. Begin with independent, code-ready GRU mechanisms; do not batch all PR titles into job JSON.
4. Add PR15 only after source/data manifests and an eight-device profile/projection. Add PR17 only after its PyTorch smoke, complete-profile and per-fold teacher/cache registry. Implement and test missing architectures/features before admitting their scientific fits.
5. Preserve the CP30 30-cell-per-arm rule for confirmation, clean outer-fold ancestry, strict FP32, fixed endpoint and unchanged official grader. Small first-seed screens are exploratory only. The eight-GPU user authorization does not turn the 25-rank hypothesis list into 25 ready implementations.

No external publication, rental, billing/account change or foreign-job mutation is part of this handoff. The current measured state is still zero remote commands and zero B300 fits.
