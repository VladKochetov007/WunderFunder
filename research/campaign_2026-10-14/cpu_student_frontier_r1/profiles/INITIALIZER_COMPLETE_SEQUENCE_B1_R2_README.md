# Complete-sequence initializer resource profile R2

R2 is a prelaunch repair of the frozen R1 one-sequence resource smoke. It keeps the same H256×2 seed-3 epoch-0 backbone, clean homogeneous F0 targets, normalization, source-qualified fitting row `(valid, row_group=37, seq_ix=44346154)`, FP32 precision, B1/C2000 geometry, optimizer and four-mode scope. It does not alter the cache or claim fit quality. The original [R1 source and preflight](INITIALIZER_COMPLETE_SEQUENCE_B1_R1_README.md) remain immutable.

The independent [R1 review](../../../unified/reviews/INITIALIZER_B1_RESOURCE_PROFILE_REVIEW_20261014_R1.md) found three blockers before any CUDA work: a valid page-rounded cgroup limit could be rejected; newly constructed initializer parameters and the reload wrapper stayed on CPU around a CUDA backbone; and zero-initialized prefix output was compared by exact hash against an unsplit recurrent call. R2 makes the cgroup check accept only the declared 5,000,000,000-byte cap or a smaller value within one 4,096-byte page, places each wrapper on the selected device and asserts every parameter/buffer device, and compares the split/unsplit zero baseline with recorded `rtol=1e-5, atol=1e-5` and maximum absolute delta. Checkpoint reload remains an exact same-call prediction comparison.

The frozen R2 contract is [here](INITIALIZER_COMPLETE_SEQUENCE_B1_R2_CONTRACT.json), SHA256 `c07644f386fd0c6d6a42935d2856c68e17fc52498718ef21920b6783e8c62665`; the harness is [here](../../../../scripts/profile_initializer_frontier_r2.py), SHA256 `f94c3274e2cc16eb1793d0ded7f4a8db39eda20eb7a2085608782e944eb1bdee`. Its [final standard-library preflight](INITIALIZER_COMPLETE_SEQUENCE_B1_R2_FINAL_PREFLIGHT.json), SHA256 `3bc2ad2aba91e754495499c5f2338ab94bd6f3a5ea4be30239138f4d5ea9821d`, binds the R2 source, contract, H256 control, target ancestry and F0 split. It ran with one CPU and a 400 MB memory maximum; no PyTorch import, cache-payload rehash, CUDA allocation or fit occurred. Ruff check/format and Python compilation passed. The [focused CPU regression receipt](INITIALIZER_COMPLETE_SEQUENCE_B1_R2_CPU_TESTS.json) records 10/10 passes under the fresh 900 MB/one-CPU gate, covering wrapper placement on a meta-device backbone and cgroup cap rounding. The profile itself checks parameter/buffer placement before forward.

The first preflight invocation chose a pathname already occupied by the R1 source receipt [preserved here](INITIALIZER_COMPLETE_SEQUENCE_B1_R2_PREFLIGHT.json) and exited before running; its [collision receipt](INITIALIZER_COMPLETE_SEQUENCE_B1_R2_PREFLIGHT_PATH_COLLISION.json) records that no file changed. An earlier R2 source/contract version did pass preflight before the comparison tolerance was finalized; its [superseded receipt](INITIALIZER_COMPLETE_SEQUENCE_B1_R2_CONTRACT_PREFLIGHT.json) binds script SHA256 `5f479ef2b468f23fa8f63f514cf1794dac65c8c254bbe5d40d6287c94973945d` and contract SHA256 `b5a93f1ca2b74d12289c198a5a978f6965e396317bb1eb4cfc195fe077a366f1`. A later invocation chose that same occupied path; its [collision receipt](INITIALIZER_COMPLETE_SEQUENCE_B1_R2_CONTRACT_PREFLIGHT_PATH_COLLISION.json) records that no file changed. Neither receipt nor source was overwritten. The final preflight has a distinct path and matches the current R2 pins.

Root owns the resource admission and launch. After the healthy H192×2 fit is graded and the profile slot is admitted, the exact command is:

```sh
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/worktrees/student-initializer-profile-r2-20261014
FIT=/home/vlad/development/wunder/worktrees/student-frontier-sequence-20261014
B8=/home/vlad/development/wunder/worktrees/student-frontier-b8-20261014
STATE="$MAIN/research/campaign_2026-10-06/bandit_continuation/runtime"
CONTRACT="$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R2_CONTRACT.json"
OUTPUT="$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R2_RESULT.json"

"$MAIN/scripts/bandit_resources.py" --root "$WT" --state "$STATE" \
  launch --attempt student-initializer-profile-f0s3-h256l2-b1-c2000-fp32-r2 \
  --kind gpu --cpus 1 --reserve-bytes 5000000000 \
  --profile 'one source-qualified clean F0 sequence; frozen H256x2 CP0; zero reference plus constant/first-row/prefix; one FP32 checkpointed complete-sequence update each; 3.5 GB allocator and 5 GB observed VRAM ceiling' \
  --contract "$CONTRACT" --python "$MAIN/.venv/bin/python" -- \
  timeout --signal=TERM --kill-after=10s 10m \
  /usr/bin/env WNN_PROJECT_ROOT="$MAIN" \
  PYTHONPATH="$WT/src:$FIT/src:$B8/src" \
  WNN_GPU_MEMORY_BYTES=3500000000 \
  "$MAIN/.venv/bin/python" "$WT/scripts/profile_initializer_frontier_r2.py" \
  --data-root "$MAIN" --contract "$CONTRACT" --output "$OUTPUT" --execute
```

The GPU profile has not run. Even a successful R2 result is resource/implementation evidence for one sequence and one update per learned arm only; it does not admit the proposed subset fit or establish initializer quality. The later fit plan is a separate recipe and requires the measured R2 resource result plus fresh root admission.
