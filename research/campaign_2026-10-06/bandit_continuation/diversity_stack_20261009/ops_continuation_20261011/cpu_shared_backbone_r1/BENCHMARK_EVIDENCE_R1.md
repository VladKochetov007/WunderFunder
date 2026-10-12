# Exact shared F0 Q8VNNI backbone: benchmark evidence R1

## Decision

The isolated exact-sharing implementation passed bitwise prediction parity in
both authorized callback pairs and was materially faster on this local Intel
i9-11900H: the shared path was 48.76–49.31% lower in measured callback time.
This is benchmark-only evidence for this CPU and this own F0 pair. It is not a
quality claim, public-score forecast, production promotion, or AMD EPYC 9224
runtime certification. Root review and target-hardware/package/quality gates
remain open.

## Frozen identities

- Code/spec freeze: commit `70749a11b6e71209f4874e44e115e8bb78e5bdee`, branch
  `codex/shared-hft-backbone-20261012`, from base
  `759c1450cd4852899ac518c2db1fdbe997af6ba1` in the isolated worktree.
- The final release wheel was built from that source commit for CPython 3.11+
  abi3. Wheel SHA-256:
  `76f55324d302c28325626d4e571e80b349b33849e29e1c75b215022e72e27151`.
  Imported native module SHA-256 in both pair receipts:
  `0cbf62fddaa57ad9aae99f62e03cd21af51050a02ec44f55e1027df320e05eae`.
- Callback runner SHA-256 is
  `8bc57f520b9b284ca863fb8a798be3a0e923ad9242c0fb7ffa58abc2e2a0af53`; its
  `utils.py` SHA-256 is
  `e8e1034d61be359ebe943015e4e6bd2f76716303eff520857b7d07eeeb6144a6`.
- Student source is
  `artifacts/runs/oldz4_h230_student_f0_s3_r3/model_ep1.pt`, SHA-256
  `664643b2653a85d192f74a976d76afa933ae56c02b2621387d2f576fd61695eb`.
  HFT source is
  `artifacts/runs/oldz4_h230_hft_f0_s3_r5/model_ep0.pt`, SHA-256
  `2bde556119ededc3693733280444a42f0f64138a2e3759bed6b068cb72f18ec8`.
  They have 211 inputs, hidden width 256 and two GRU layers. Producer preflight
  found equal normalization, input projection and recurrent tensors; only the
  four head tensors differ. The raw final head rows are 31 and 23 respectively;
  both are served as their first two rows by the frozen predictor. The benchmark
  keeps the heads separate and averages their outputs at exactly `0.5/0.5`.
- The source grade used FP16. Its exposed fixed-half grade was `0.6834706264664301`
  versus `0.6831418133898598` for the student, with conditional paired interval
  `[-0.0007169960064610543, 0.001326790542246119]` over 468 inspected groups / 141
  components. Those values motivated the fixed pair only; this run loaded no
  labels and did not regrade it.
- The callback panel is the features-only file
  `/tmp/wnn_submission_exploit_runtime_r1/valid_first10_complete_groups.parquet`,
  SHA-256 `e5e6472cedfcb20ecf01195777ab986b3560379c5e81caea71973632dde0d6a3`.
  It contains ten complete 20,000-row sequences. Each method made 400,000
  callbacks across two explicit reset passes. The runner measures every row's
  callback and only retains/hash-checks rows requested for prediction; no labels
  were read.

The full actual-checkpoint/config preflight and API self-test are preserved in
`PRODUCER_PREFLIGHT_R1.json` and `API_SELF_TEST_R1.json`. Both benchmark pairs
record the source commit, checkpoint and grade-receipt pins, panel and runner
pins, per-sequence hashes, runtime ABI/CPU/affinity and cgroup snapshots.

## Results

| Pair/order | Separate baseline | Shared candidate | Runtime reduction | Output hashes |
|---|---:|---:|---:|---|
| 1: separate → shared | 56.18693 µs/row | 28.47984 µs/row | 49.3123% | Exact match on all 10 sequences |
| 2: shared → separate | 58.51550 µs/row | 29.98067 µs/row | 48.7646% | Exact match on all 10 sequences |

Both methods also reproduced identical predictions across their two reset
passes. Pair order alternated as frozen; no third pair or broader performance
search was run. The native runtime selected `q8vnni` for both methods. Each
command used one pinned CPU (`CPU 0`) with `cpu.max=100000 100000` and the
120-second command limit. The measured host was `11th Gen Intel(R) Core(TM)
i9-11900H @ 2.50GHz`, Python SOABI `cpython-313-x86_64-linux-gnu`.

The implementation reduces logical dynamic-Q8 weight payload read per callback
from 2,142,888 to 1,209,692 bytes, saving 933,196 bytes; over 400,000 callbacks
that is 373,278,400,000 logical bytes. This is payload accounting from the
loaded Q8 matrices, not a claim of measured DRAM traffic. Resident streamed
weights remain 2,143,104 bytes for either endpoint. Both still allocate 4,096
bytes of recurrent state and 23,728 bytes of scratch, so this prototype makes no
resident-memory or package-size claim.

Each benchmark child cgroup had a 1,199,996,928-byte memory max. Pair 1 peak was
537,505,792 bytes and pair 2 peak was 514,256,896 bytes; both recorded zero
`high`, `max`, `oom` and `oom_kill` events. Before the pairs, the parent campaign
slice reported respectively 14,003,716,096 and 13,744,144,384 bytes current,
18,000,000,000 high, 20,000,000,000 max, no swap, and 500% total CPU quota.

The local i9 result does not certify the target AMD EPYC 9224. It only shows
that exact sharing is a promising quality-per-runtime candidate on this tested
host. No target-quality grade, full stream/package gate, submission ZIP,
recommendation, leaderboard score, or production source change was produced.

## Verification commands and outcomes

The final committed source passed these bounded commands, each using the
absolute main-repository `scripts/limited`, `wnn-campaign.slice`, one thread,
1.2 GB memory max and 120-second runtime max. The build used one Cargo job.

```sh
LD_LIBRARY_PATH=/home/vlad/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib \
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1200000000 WNN_RUNTIME_MAX=120s CARGO_BUILD_JOBS=1 \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
cargo test --lib --features python shared_backbone
```

Result: 2 passed: exact shared-Q8 output equality across two reset passes, plus
constructor preflight accepting distinct heads and rejecting a changed trunk
and a head that does not expose exactly two outputs. Python API self-test passed
default-versus-explicit-disabled equality, shared Q8 parity and incompatible
trunk rejection. The actual producer preflight passed with both pinned
checkpoints, matching input configuration and feature panel.

The source document `IMPLEMENTATION_AND_PREBENCH_R1.md` records the frozen
build/unpack commands. These were the only two callback pair commands:

```sh
WNN_SOURCE_COMMIT=70749a11b6e71209f4874e44e115e8bb78e5bdee \
PYTHONPATH=/tmp/wnn-shared-f0-package-r1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1200000000 WNN_RUNTIME_MAX=120s \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/cpu_shared_backbone_r1/benchmark_shared_f0.py \
--pair pair1 --order separate-shared --output \
research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/cpu_shared_backbone_r1/PAIR1_R1.json

WNN_SOURCE_COMMIT=70749a11b6e71209f4874e44e115e8bb78e5bdee \
PYTHONPATH=/tmp/wnn-shared-f0-package-r1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1200000000 WNN_RUNTIME_MAX=120s \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/cpu_shared_backbone_r1/benchmark_shared_f0.py \
--pair pair2 --order shared-separate --output \
research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/cpu_shared_backbone_r1/PAIR2_R1.json
```

## Attempt ledger

- A targeted Cargo test was first invoked from the worktree root rather than
  `native/`; it exited 101 with “could not find `Cargo.toml`” and compiled
  nothing. The corrected invocation above passed.
- The first Python-feature unit-test executable could not load
  `libpython3.13`; rerunning with the explicit `LD_LIBRARY_PATH` above passed.
- The venv has no `pip`; an attempted target install stopped with
  `No module named pip`. No shared environment was changed. The abi3 wheel was
  extracted with `unzip` into `/tmp/wnn-shared-f0-package-r1` instead.
- A repository-wide Cargo formatter check reported pre-existing formatting
  drift in native sources. Targeted formatter inspection also showed unrelated
  existing drift; only the new snippets were manually aligned. No whole-file
  formatter rewrite was applied. `git diff --check` was clean and the benchmark
  script passed `ruff check` before source freeze.
- One preflight command used a mistyped
  `diversity_stack_20260909` directory and exited 2 before opening Python or
  reading a checkpoint. The corrected command produced the committed
  `PASS_PREFLIGHT` receipt.
- A release wheel was built before the last source tightening but never used
  for the callback benchmark. The wheel was rebuilt from frozen commit
  `70749a11`; only its SHA-pinned module was used in the two recorded pairs.

The full receipts are `API_SELF_TEST_R1.json`, `PRODUCER_PREFLIGHT_R1.json`,
`PAIR1_R1.json`, `PAIR2_R1.json`, and `BENCHMARK_SUMMARY_R1.json`. All evidence
is isolated and benchmark-only pending root's independent review.
