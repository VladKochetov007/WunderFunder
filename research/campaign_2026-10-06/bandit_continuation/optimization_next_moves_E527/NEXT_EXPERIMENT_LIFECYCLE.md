# E527 — Finite experiment queue and resource lifecycle

**Status:** design note; no provider, timer, service, or rental state was changed.  
**Owner of future admission:** root controller.  
**Execution rule:** every queued step is already scientifically reviewed, frozen, resource-gated, and bounded before a provider is started.

## Existing control points

Prefer the repository's durable primitives over a new controller:

- `scripts/bandit_resources.py` is the aggregate-limited **local** systemd/cgroup attempt manager: it owns local reservation/launch/status/reconciliation/drain/reopen records, not Vast/Kaggle instance creation. Its current admission barrier and live receipts are authoritative for local work; historical allocation snapshots are not current authorization.
- `scripts/bandit_sequence.py` is a **local finite sequencer** for a hash-frozen list of distinct authorized commands: it checks predecessors/receipts and outputs, stops on a stale heartbeat, and does not retry failures. `STOP_ADMISSION.json` drains the current healthy local attempt and blocks the next one. A command packet may itself submit work to an external provider only when an existing provider-specific path and separate live admission allow it.
- A local stop marker does not destroy, stop, or observe a remote Vast instance. The named cloud operator must separately collect terminal artifacts, query/destroy the provider resource, and verify final billing/instance state.
- `scripts/kaggle_sequence.py` and `scripts/kaggle_gpu_runner.py` provide the existing sealed Kaggle execution and phase-heartbeat path. Keep packet/data/recipe hashes, worker telemetry, output checks, and cloud quota admission attached to the attempt.
- `research/campaign_2026-10-06/cloud_preflight/poll_provider.py` is a read-only Kaggle quota/session monitor. Its existing systemd timer is configured for one-shot polling every 60 seconds with 10% CPU, 512 MiB maximum memory, and a 55-second command runtime. The repository declaration does not prove that the host timer is installed or currently active; check host state before relying on it. This scout did not mutate or activate the unit.
- That Kaggle monitor is not self-terminating: its current target list retains historical kernels, a terminal artifact receipt skips that kernel's status call, but every timer tick still refreshes account quota; an empty target list is rejected. If the current finite queue is empty, root must make an explicit scheduling decision so an active timer does not keep polling an idle account. This scout did not clear targets or change the timer.
- Vast's documented `onstart` callback can wake a collector after the instance starts, but status polling still needs a bounded fallback. A callback is an event hint, not proof that inputs are staged or training has begun.

The repository currently has no Vast-specific API runner, callback listener, or poll target. `poll_provider.py` is Kaggle-only. Do not repurpose its Kaggle timer as a Vast daemon or assume a callback receiver exists. If a later root-authorized Vast attempt needs callback wakeups, first scope a small receipt-writing handoff into the existing attempt lifecycle. Make repeated callbacks idempotent on `(attempt_id, instance_id, event)` and allow them only to wake the controller; callbacks never reserve resources, start successors, or self-retry. Keep callback authentication outside result/log artifacts. Otherwise use bounded status reads in the already authorized cloud-operator session and destroy on terminal failure/timeout.

## Named ownership per queue item

- The **root controller** owns fresh admission, the frozen finite queue, source/recipe identity, stop/drain/resume decisions, and the next allocation after grading.
- The **scientific actor** owns independently implemented code and weights, the exact runnable command, parity gates, and the frozen recipe packet. No teammate code or weights become an initializer.
- The **cloud operator** owns provider-side readiness, bounded phase heartbeats, resource observations, terminal status, output download, and cleanup evidence. Provider state alone never advances the science tree.
- The **independent grader** owns the fixed official metric and declared teacher/student/blend comparison; the actor does not sign its own result.

Every admitted item names these roles and an active owner who remains responsible through a terminal receipt. A prospective successor may be prepared and source-reviewed while the current item runs, but neither a callback nor the exiting actor may admit it. If an owner or artifact is missing at a terminal transition, preserve the last receipt and mark the item for review rather than silently advancing or leaving a resource running.

Do not create an open-ended queue runner or a new always-on service for one experiment. If an existing tool cannot represent a needed transition, first name the missing transition and add only a small reviewed extension to its existing receipt/state machine.

## One finite queue item

Freeze each attempt as a self-contained record containing:

1. A unique attempt ID and parent hypothesis/experiment IDs.
2. Exact source revision and dirty-tree disposition; command, environment, dependency/runtime versions, data/split/membership fingerprints, target-cache ancestry, and output paths.
3. A fixed scientific recipe, scorer, checkpoint selection rule, validation exposure, seed, comparison/control, and intended student/blend endpoint.
4. Resource gates: provider quota/credits, rental type and interruption/resume policy, accelerator UUID/class/count, disk headroom, expected source/cache byte count, host RAM/CPU cap, runtime ceiling, and cleanup deadline. Numeric gates remain unset until live measured evidence supports them.
5. An immutable predecessor relation and a specific successor packet, or an explicit queue-empty terminal state.

The successor packet should be built and hash-checked while its predecessor is running, but it must not be admitted until the predecessor has a terminal receipt and the root controller's current admission/resource checks pass. This overlap avoids idle human preparation time without running duplicate science or evading the shared GPU lock.

## State transitions and bounded wakeups

`draft → frozen → preflight_passed → admitted → provider_starting → worker_ready → profiling → fitting → exporting → collected → independently_graded → tree_reconciled → next_finite_item_or_closed`.

The successor has its own independent `draft/frozen/preflight_passed` preparation states and may be made ready while the current item runs. `next_finite_item_or_closed` is reached only after this attempt is terminal and reconciled; a ready successor still needs a fresh admission check before launch.

- **Draft/frozen:** recipe-audit and source review finish before resource reservation. Hash the packet after all edits; never repair a running packet in place.
- **Preflight:** re-read current stop/admission state, provider quota/credits, exact data identity, free disk, live host limits, and whether an identical attempt is already active. Fail closed on unknown admission facts; record them as unknown, not as negative science.
- **Admitted/provider_starting:** reserve through the existing resource helper. An API accepted response or provider `RUNNING` status does not prove that the notebook started, the GPU is visible, or fitting began. Keep cloud startup and data staging as their own heartbeat phases with explicit wall-time ceilings.
- **Wakeups:** if an authorized callback is available, have it publish a small timestamped wakeup/instance identity to the existing collector path. A bounded read-only poller remains fallback. Polling should stop when the durable terminal evidence receipt proves status plus downloaded output and logs; no timer should poll an empty queue forever.
- **Worker_ready/profile:** enumerate visible device UUIDs and runtime/cuDNN versions; run the frozen resource profile before a fit. Record p50/p90 and profiler-free step wall time. If actual memory, CPU, or quota violates the gate, write an execution/resource failure receipt; do not call it a model failure.
- **Fitting:** heartbeat each phase/update window. Save resumable checkpoints atomically with model, optimizer, scheduler, scaler, RNGs, sampler/order state, epoch/update/chunk/detach position, recipe hash, split/cache identities, and library/hardware provenance. A restart resumes only from a checkpoint whose full provenance matches the frozen packet.
- **Export/collect:** write predictions/checkpoints to a unique attempt directory, verify expected filenames and hashes, download logs and outputs, and persist provider terminal status before declaring success. A provider terminal state without downloaded artifacts is `collection_incomplete`, not a successful experiment.
- **Grade/reconcile:** an independent grader runs the official metric and planned fixed-student/blend comparison, then root appends the hypothesis/experiment and tree receipts. The actor does not approve its own result. A failure receives a new attempt ID for any repaired retry; retain the original packet and failure evidence.
- **Queue empty:** after all authorized work has been graded or marked for review, collect outputs, stop/destroy the compute instance, confirm billing/terminal status, and close the finite sequence. Never leave a GPU running while waiting for a later user decision.

## Failure, interruption, and shutdown behavior

- API/provider timeout, stale heartbeat, missing callback, startup error, failed input hash, quota change, OOM, worker exit, export failure, or missing artifact sets a terminal `needs_review`/failure state for that attempt. Record which phase and expected/observed evidence were present.
- No automatic retries after ambiguous failure. A retry uses a new attempt ID and a corrected reviewed packet, only after the old instance is verified stopped/destroyed and the root reopens admission if needed.
- An explicit stop closes new admission and drains healthy admitted work. Preserve the stop receipt. A later explicit resume is a new decision that supersedes the stop for new work; it must not erase the original record.
- **Destroy vs stop:** collect and hash outputs first. For a paid provider, stopping can retain attached storage charges; destroy/delete after artifacts are verified when no further work requires that disk. If a checkpoint must remain resident for a declared resume, include its continuing storage charge in the admission decision and set an expiry/cleanup action.
- **Rental type:** keep on-demand and interruptible/bid attempts in separately preregistered cohorts. Use on-demand for the first device/profile baseline. Admit a bid attempt only after exact checkpoint-and-resume behavior is tested; an interruption yields a retained execution-failure receipt and a reviewed new attempt ID, never an unbounded self-retry.
- On an empty queue, failed admission, or user stop, shut down rather than spinning an idle GPU. Bound each callback/poll request and each poller's runtime. Keep host timers and Kaggle monitoring independent from cloud-resource lifecycle; changing one must not silently admit training.

## Cost and scheduling comparisons

Keep each measured component separate:

`T_total = T_startup + T_transfer + T_verify/cache + T_profile + T_fit + T_export/pull + T_cleanup`.

`C_attempt_cash = C_compute_active + C_storage_retained + C_transfer`.

Use the provider's verified total active hourly charge **or** its verified component breakdown; do not double-count storage if included. Report interrupted fit work/quota seconds separately as a subset of resources already consumed, not as another cash charge on top of `C_attempt_cash`. Keep `C_attempt_cash`, fit time, useful completed endpoints, score, and observed interruption/completion frequencies null until actually measured. Do not translate a marketplace reliability score into a completion probability.

For two independent GPUs compare completed, independently graded candidates per elapsed hour and all-in dollar against a matched one-GPU cohort, counting shared CPU/RAM, duplicate cache, and host contention. For DDP compare one fit's latency at matched global examples, loss denominator, update cadence, and schedule. Those are different questions.

Use these break-even equations with the fields still unknown until an actual offer and measured run exist:

- `C_i = r_gpu_i * t_active_i + r_storage_i * t_exists_i + r_up_i * bytes_up_i + r_down_i * bytes_down_i`. Use the provider's verified `dph_total` **or** its itemized equivalent; do not count a component twice. Report the portion of `t_active_i` that produced no retained checkpoint as lost work, but do not add that same time again as dollars.
- Faster card A is cheaper for the same fixed recipe only if `C_A <= C_B`. `t_active` includes startup, staging, profile, fit, export, and cleanup while the instance is live. Solve the rate ceiling after measuring these terms; do not substitute peak FLOPS for `t_fit`.
- With empirically observed completion rate `p_i` from comparable identically distributed independent attempts, repeated retries have expected spend per successful artifact `C_attempt_i / p_i`. If `p_i` is not measured, this stays null. A marketplace reliability score is not `p_i`.
- For independent two-card runs compare `graded_artifacts / cohort_wall_hours` and `graded_artifacts / C_cohort` against the matched one-card cohort. Candidate count alone is not score; retain the complete official-metric and fixed-student/blend comparison.
- Keep any later compute cohort within the user-reported $20 planning ceiling only after a fresh read-only balance check. E522 proposed at most $2 for a profile, at most $10 for an initial cohort, and at least $8 held for transfer/startup/failure reserve; this is a planning proposal, not a quote or current spend authorization. No current offer or price is observed in E527.

## Smallest useful next sequence

1. Reconcile H230's live owner/attempt/provider receipts and the active stop/admission record. Do not duplicate it.
2. If no valid same-shape remote profile exists, admit exactly one matched device profile for the H230 recipe, with cold/warm staging and full-shape wall/resource traces.
3. Choose one systems follow-up conditional on its measured bottleneck: data digest/cache, one-worker/prefetch, recurrent dispatch, or graph/compile. Keep exact data order and reference numerical behavior.
4. In parallel with preparation (not compute admission), freeze one independent teacher/architecture hypothesis and its fixed student endpoint from the open tree. Keep its fit conditional on resource and data readiness.
5. Only after a device/library gate, schedule precision and kernel probes. FP8 begins as a bounded eligible-op/parity probe, not a presumed speedup or production recipe.
6. If two devices are actually visible, start with independent own-seed fits for candidate throughput. Try synchronous multi-GPU training only if single-fit latency is the explicit objective and the measured communication case supports it.

This sequence measures where budget can buy more credible experiments toward top-1 and >0.700. It does not optimize for cheap compute in isolation, and it does not forecast a public score.
