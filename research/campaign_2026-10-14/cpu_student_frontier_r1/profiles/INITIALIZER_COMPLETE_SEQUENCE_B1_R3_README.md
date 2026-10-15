# Complete-sequence initializer resource profile R3

R3 repairs the concrete pre-update failure from the root-admitted R2 attempt. The preserved [R2 failure record](INITIALIZER_R2_FAILURE_R1.json) reports that the harness requested `torch.device("cuda")` (index `None`) and then compared it directly with parameter devices resolved as `cuda:0`. R2 therefore rejected correctly placed parameters before its first optimizer update; it produced no resource or quality evidence.

The [R3 contract](INITIALIZER_COMPLETE_SEQUENCE_B1_R3_CONTRACT.json) keeps the same frozen H256×2 CP0 backbone, clean homogeneous F0 targets, common normalization, source-qualified `(valid,37,44346154)` fitting sequence, B1/C2000 FP32 profile and three learned initializer arms. At execution entry, R3 resolves the current CUDA index once and passes that explicit device through the input, propensity, backbone, initializer and checkpoint-reload paths. Exact parameter and buffer device checks remain enabled. The result records both the requested and resolved device. This changes device identification only; objective, data, initializer definitions and resource ceilings remain fixed.

The [R3 harness](../../../../scripts/profile_initializer_frontier_r3.py) and [focused regression test](../../../../tests/test_profile_initializer_frontier_r3.py) are source-bound by the contract. The standard-library preflight passed with one CPU and a 300 MB cap; the [receipt](INITIALIZER_DEVICE_CANONICALIZATION_R3_PREFLIGHT_R1.json) binds the current source, contract, fitted control, targets, and preserved R2 failure record. The 900 MB CPU regression passed 14/14 tests; its [receipt](INITIALIZER_DEVICE_CANONICALIZATION_R3_CPU_TESTS_R1.json) covers implicit CUDA resolution to the current index, preservation of explicit CUDA indices, non-CUDA devices, meta-device wrapper placement and cgroup page rounding. The preflight output uses a device-specific filename because the older `INITIALIZER_COMPLETE_SEQUENCE_B1_R3_PREFLIGHT.json` is already occupied by R1 evidence and remains unchanged; the [collision record](INITIALIZER_DEVICE_CANONICALIZATION_R3_PREFLIGHT_COLLISION_R1.json) documents the first attempted path.

The frozen pins are contract SHA256 `08b943d63913e5ceebd845f1419feed65af36b375589304ed85fb50ed03fa3e7`, harness SHA256 `b2e01a312b5f7e7a785c5bd92ad3d106e6758e78432ebac0528444cad2c140c9`, test SHA256 `5b8222541d8e038e7335309a9782a52fdcf0d456742cb0ed259a8c67eb89d1c1`, preflight SHA256 `70e5fee614adaddffde31c616b4e522bee9c0ec04e50a751513f9ef8ae8655b7`, and test receipt SHA256 `1f0eff8a545c0e7c4b3be6396274b4ecc9adfa8b175ea64914e2f95e6631fda7`. No GPU work is authorized by this packet; root owns any later admission. R3 remains a one-update resource smoke and carries no fit-quality claim.

The successful standard-library preflight command, run from the project root, was:

```sh
WNN_PROJECT_ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack \
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=300000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python \
  /home/vlad/development/wunder/worktrees/student-initializer-profile-r3-20261015/scripts/profile_initializer_frontier_r3.py \
  --data-root /home/vlad/development/wunder/wnn_connectome_starterpack \
  --contract /home/vlad/development/wunder/worktrees/student-initializer-profile-r3-20261015/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R3_CONTRACT.json \
  --output /home/vlad/development/wunder/worktrees/student-initializer-profile-r3-20261015/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_DEVICE_CANONICALIZATION_R3_PREFLIGHT_R1.json \
  --preflight-only
```

After independent review and a fresh root admission, the prepared controller launch is:

```sh
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/worktrees/student-initializer-profile-r3-20261015
FIT=/home/vlad/development/wunder/worktrees/student-frontier-sequence-20261014
B8=/home/vlad/development/wunder/worktrees/student-frontier-b8-20261014
STATE="$MAIN/research/campaign_2026-10-06/bandit_continuation/runtime"
CONTRACT="$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R3_CONTRACT.json"
OUTPUT="$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_DEVICE_CANONICALIZATION_R3_RESULT_R1.json"

"$MAIN/scripts/bandit_resources.py" --root "$WT" --state "$STATE" \
  launch --attempt student-initializer-profile-f0s3-h256l2-b1-c2000-fp32-r3 \
  --kind gpu --cpus 1 --reserve-bytes 5000000000 \
  --profile 'one source-qualified clean F0 sequence; frozen H256x2 CP0; zero reference plus constant/first-row/prefix; one FP32 checkpointed complete-sequence update each; 3.5 GB allocator and 5 GB observed VRAM ceiling' \
  --contract "$CONTRACT" --python "$MAIN/.venv/bin/python" -- \
  timeout --signal=TERM --kill-after=10s 10m \
  /usr/bin/env WNN_PROJECT_ROOT="$MAIN" \
  PYTHONPATH="$WT/src:$FIT/src:$B8/src" \
  WNN_GPU_MEMORY_BYTES=3500000000 \
  "$MAIN/.venv/bin/python" "$WT/scripts/profile_initializer_frontier_r3.py" \
  --data-root "$MAIN" --contract "$CONTRACT" --output "$OUTPUT" --execute
```

The command above was admitted by root as `student-initializer-profile-f0s3-h256l2-b1-c2000-fp32-r3` and completed successfully. The [R3 terminal record](terminal_r3_r1/README.md) preserves exact resource measurements and output hashes. This remains a resource-only smoke. The concrete native C++ gate has not yet run, and no initializer subset fit, full-panel prediction or quality comparison is authorized by this result.
