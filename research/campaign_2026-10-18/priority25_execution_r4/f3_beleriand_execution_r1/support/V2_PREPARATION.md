# F3 deep stage1 support wrappers V2 — preparation only

**Status: UNADMITTED; do not invoke.** This packet adds a separate deep-only root launcher/capture pair. It does not change the frozen V1 wrappers, canonical checkpoint collector/verifier, training plan, or wide stage1 acceptance. No remote launch, staging, checkpoint read, training, inference, or grade was performed while preparing V2.

The purpose of V2 is to collect and structurally verify the deep fold3/seed1001 stage1 checkpoint using the already reviewed `f3_checkpoint_support_r1` helper packet. Its preparation manifest SHA-256 is `7e4fce0d7232a487a769e12d611050bc608f9cc7fcea300bc2063525abfa38dc`, and its independent review SHA-256 is `de709897397d7cd775d2aba84fb71f8f7d91504802814f5bcfc26e48e203763b`. The F3 plan remains SHA-256 `b303146cf9db05e0a3bddf8f245d3c9b880d055ce7e97ebe5dd13510b5f4a2d8`. V2 accepts only `--arm deep` and requires the exact native attempt `computer-vision-homework-r01-deep-f3s1001-ep01-v1` to be finished successfully. At collection time the sequence may have advanced: V2 accepts only continuation index 2 or 3, verifies that `sequence.attempt` equals the `attempt` of that exact frozen `plan.steps[index]`, and accounts for that current worker dynamically. It refuses an unadvanced stage1 pointer, a pointer outside the two continuation steps, or any attempt/index mismatch.

## Files and output names

- `launch_stage1_v2.py` admits one collector or verifier support unit after the native attempt finishes successfully. It uses a V2-suffixed systemd unit and creates V2-suffixed admission and launch records.
- `capture_stage1_v2.py` captures terminal raw bytes for that V2 unit, writes V2-suffixed execution/stop records, then stops only that exact terminal unit.
- The reviewed checkpoint helper outputs keep their canonical unsuffixed paths: `RANK01_DEEP_F3_S1001_STAGE1_COLLECTION.json` and `RANK01_DEEP_F3_S1001_STAGE1_VERIFICATION.json`. Collector and verifier support provenance is separate (`*_COLLECTOR_*_V2.json` / `*_VERIFIER_*_V2.json`).

The launch gate checks the exact F3 plan, fold3 manifest, portable source/archive manifests, all eight reviewed helper hashes, finished attempt, and matching sequence pointer. It requires an empty campaign-slice `cgroup.procs`, accounts for every populated direct child via cgroup v2 `cgroup.events` (which includes processes in descendants), and admits only the expected sequence controller plus the current continuation worker validated against frozen plan index 2 or 3, with their fixed caps. Their hierarchical caps bound descendants. It preserves the existing aggregate CPU, memory, swap, scratch, disk, and output-collision gates. Each support unit remains limited to its reviewed CPU, memory, swap, and two-minute runtime envelope with CUDA hidden.

## Root admission requirements

This preparation is not an admission or execution approval. Before any remote launch, root must independently review these exact wrapper bytes, compute and bind their hashes, and provide the exact reviewed support-manifest/review pins. Root must refresh the native attempt, sequence, source/helper hashes, parent resources, ownership, scratch, disk, and B output collisions immediately before each support operation. Collector and verifier remain separate bounded units; verification only follows a successfully captured collector receipt.

Before sending either launcher to B, root must also perform a fresh create-only check on the local `priority25_execution_r4/receipts` paths below. The capture wrapper repeats its own output checks before any capture SSH call, but that later check cannot replace the pre-launch collision check:

- Collector: `RANK01_DEEP_F3_S1001_STAGE1_COLLECTOR_ADMISSION_V2.json`, `RANK01_DEEP_F3_S1001_STAGE1_COLLECTOR_LAUNCH_V2.json`, `RANK01_DEEP_F3_S1001_STAGE1_COLLECTION.json`, `RANK01_DEEP_F3_S1001_STAGE1_COLLECTION_EXECUTION_V2.json`, and `RANK01_DEEP_F3_S1001_STAGE1_COLLECTION_STOP_V2.json`.
- Verifier: `RANK01_DEEP_F3_S1001_STAGE1_VERIFIER_ADMISSION_V2.json`, `RANK01_DEEP_F3_S1001_STAGE1_VERIFIER_LAUNCH_V2.json`, `RANK01_DEEP_F3_S1001_STAGE1_VERIFICATION.json`, `RANK01_DEEP_F3_S1001_STAGE1_VERIFICATION_EXECUTION_V2.json`, and `RANK01_DEEP_F3_S1001_STAGE1_VERIFICATION_STOP_V2.json`.

Root must preserve any collision or failed/partial output rather than overwrite it or retry into the same path. The local capture script checks its expected output paths with create-only semantics before contacting B; launch and transport receipts also use exclusive creation. The helper receipt paths on B are independently checked before launch.

The exact wrapper SHA-256 values are to be recorded in the root’s independent review/admission after this preparation. V2 does not establish wide/deep parity, a prediction score, convergence, serving quality, or CP30 completion. Stage1 receipt acceptance remains structural and cannot stand in for the subsequent paired terminal checkpoints and grade.

`test_stage1_v2_sequence.py` is a small synthetic guard for the sequence-to-plan binding: a finished deep stage1 attempt with the sequence at either valid continuation is accepted, while mismatched continuation attempts, indices, or native-step identities are rejected. It does not launch a unit or read campaign data.
