# CPU homogeneous cache admission source

This local-only packet prepares the canonical CPU cache builder under the frozen T3 contract. `REMOTE_SUBMISSION.py` and `REMOTE_SUBMISSION_TEMPLATE.py` are intentionally identical and fail closed before any remote read or write while the S5-valid acceptance, exact stop, producer SHA, sidecar, and invocation pins are unset. Root must bind the actual S5-valid receipt, independently review the final body, and recheck current headroom, source/data identities, all six accepted/stopped inputs, output paths, unit ownership, scratch, and disk before launch.

The source reuses the previously reviewed read-only data-identity and parent-cgroup gates, verifies the 167-file staged source and canonical split, and checks the six prediction sidecars plus array file stats only. It does not read or hash prediction arrays; the canonical builder performs streamed SHA, shape, FP32, and finiteness validation while producing the cache. It launches no model inference or fit.

See [PREPARATION.json](PREPARATION.json) for local pins and [CPU execution review](../../TEACHER_CACHE_CPU_EXECUTION_REVIEW.json) for the reviewed CLI, arithmetic, and resource analysis.
