# F3 stage1 support orchestration

`launch_stage1_v1.py` is the root's remote admission wrapper for one collector or verifier, for exactly `wide` or `deep` F3 stage1. The [independent review](ROOT_STAGE1_SUPPORT_REVIEW_V1.json) binds its bytes and the local capture wrapper. Root must first verify the reviewed checkpoint packet manifest and pass its eight exact helper hashes, then send the exact launch source on SSH stdin. The wrapper records the supplied wrapper/review hashes; the sending root must verify those bindings before dispatch.

Native `finished` and explicit exit status zero are required. The wrapper returns `WAITING_NATIVE_FINISH` without mutations when that gate is not met. After native completion it rechecks staged helper, plan, fold and source-registry hashes, current sequence ownership, fresh aggregate CPU/memory reservations, physical scratch and collision-free output/unit names. Collector and verifier remain separate serial admissions.

`capture_stage1_v1.py --arm wide|deep --operation collector|verifier` captures exact remote receipt bytes, hashes and terminal systemd evidence to `P/receipts`. Only after capture does it stop the exact owned terminal support unit. Actual computation time comes from the monotonic process start/exit timestamps, not `systemd-run --wait`. A failed operation retains evidence and must be diagnosed before a new version is admitted.

The wide operations completed on 18 October at 18:22 UTC. Their collection and verification must not be repeated. The same reviewed wrappers can handle the deep stage1 when its native attempt has actually finished; this does not authorize a duplicate controller or any change to the frozen fit plan. No terminal or result-collection invocation is implemented here.

The future deep-only [V2 wrappers](V2_PREPARATION.md) have passed [root independent review](ROOT_STAGE1_SUPPORT_REVIEW_V2.json), including actual frozen-plan continuation-index binding. Their four synthetic tests are preparation evidence, not checkpoint verification. Follow the V2 local and remote collision gates on fresh admission; do not reuse the already completed wide V1 outputs.
