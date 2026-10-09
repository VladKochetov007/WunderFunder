# H230 all-supplied teacher cache build contract V2

V2 supersedes [cache contract V1](./H230_ALLDATA_TEACHER_CACHE_BUILD_CONTRACT_V1_20261008.json) (SHA256 `53bfbdaebf6e3990f4c0df43c067de66703914fadd0045b551650ed6b3513430`). It preserves V1's teacher sources, target construction, supplied-row membership, output schema and exact worker argv. The only requested change is the post-parent resource proposal: **1 CPU and an 8,000,000,000-byte worker reservation**. This is a conservative request, not a measured exact-build profile and not launch approval.

The build remains blocked until retry3 finishes successfully, root verifies the fixed ep3 parent artifact, approves this exact resource request, and fresh controller admission confirms aggregate memory/OOM headroom after the fit reservation is released. Preserve the 18GB memory-high, 20GB memory-max, zero-swap and current aggregate CPU limits. Recheck the output/partial paths and the 40,820,025,600-byte prebuild free-space floor.

The immutable source/input/cache details, teacher ancestry, exact command, 8-array layout, complete train/valid row-order requirements, and declared fold-teacher history remain in the JSON. The cache contains all 10,606 train and 1,873 valid target rows. Since the later student fit includes the 468 exposed evaluation groups, it supports no held-out claim. Complete cache coverage does not imply that a drop-last student epoch consumes every group.

The earlier clean-fold builder run used chunk size 2, finished in 67.07 seconds, and peaked at 6,499,999,744 bytes under a 6.5GB reservation. That is context, not a profile of this all-supplied build. Contract SHA256: `3b571613e806c23056c855d79b71917b4287a631e0a8ea08c5342f5f5ddeb92c`.
