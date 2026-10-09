# E529: continuous research implementation

## Actual admission and approach-first allocation — 2026-10-08

Local H230 is fitting; cloud GPU V12 and TPU V5 have actual push receipts and active owners. Provider status alone does not establish scientific progress. Vast H217 packet is built; no rental yet, with owned-lifecycle transport repairs in progress. The original queue is a preparation snapshot, not an automatically running dispatcher. Approach diversity has priority over more seeds. Reviewed source/recipe semantics remain fixed; local caps are not cloud caps. Latest user permits multiple owned Vast workers within the aggregate$4 initial/$20 cumulative budget, including duplicated staging costs.


Authority: [saved approved plan](../../../../specs/plans/2026-10-07-continuous-research.md), [ADR0013](../../../../specs/adr/0013-owned-paid-compute-and-continuous-queue.md). Native roles are Luna/max. Local limits remain 6 CPU / 20 GB RAM / 5 GB observed VRAM; one GPU worker. Vast allowance is $4 initial / $20 total all-in, only on instances created by this campaign. No existing or unrelated machine may be touched.

## Current execution truth

- The user reported C1 single public scores 0.6647/0.6654, below incumbent 0.6660; these are scored endpoints, not unscored candidates. Full H230 student training has not started. See [public receipt](../submission_C1_E524/E524_PUBLIC_RESULTS_20261007_V1.json).
- E528-A2 is an actual completed CPU input-reuse measurement with exact feature parity; see its own lineage. It is not a training-speed or score result.
- H217/H232 have source-bound staged recipes, including own controls and corrected FP16 profile-update evidence. Locking a recipe preserves its comparison; it does not mean the direction is deferred. They are not fit results.
- H218 has an independently written block-memory attention prototype and small CPU causal/chunk/reset/gradient tests; it is not integrated into the production trainer and has no actual-device profile or score.
- GPU/systemd and shell DNS access are unavailable in this managed environment. No Vast instance was rented. The token remains private and unvalidated remotely. Git metadata is read-only.
- Grok session creation is blocked by its read-only state directory; OpenCode's prior abort was under a restrictive virtual-address limit, not a demonstrated model-provider failure; Freebuff's usable interactive route remains unverified. No new external-model proposal is claimed.

## Queue interface

`QUEUE_V1.json` registers 15 finite attempts across four lanes. It is a preparation snapshot, with admission disabled until exact packets and fresh resource/provider gates exist. Existing successful science is reused. The attention exploration slot is independent of the TCN score.

Inspect without launching:

```sh
taskset -c 0 prlimit --as=1500000000 --cpu=60 -- timeout 60s \
  .venv/bin/python scripts/research_dispatch.py \
  --queue research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/QUEUE_V1.json \
  --runtime research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/runtime
```

To make a later queue runnable, freeze a new identified queue with each ready entry's literal **guarded** `launch_command`, exact expected artifact names and source-bound recipe. Root issues a separate admission receipt with owner `root`, `authorized:true`, matching attempt ID and canonical job SHA-256, source hashes and timezone-aware creation/expiry at most 120 seconds apart. Admission must follow actual provider/local resource gates. Add `--execute` only from the authorized host execution path; it dispatches once and exits, with at most one launch per free lane. The called provider adapter owns durable execution and collection. It is not permission to bypass `bandit_resources`, a cloud packet's review, or Vast ownership/budget checks.

The dispatcher durably writes intent before calling a provider. A failed or timed-out submit keeps its lane reserved, including across restarts or a later queue omitting that attempt. It never retries an ambiguous launch. A completion receipt must bind the exact job, contain every expected artifact exactly once, and match actual project-local file hashes. Paid completion additionally requires verified teardown. Provider status alone is insufficient.

To release a diagnosed failed attempt, supply `--reconcile-terminal <receipt>` separately from dispatch. Root's receipt must bind attempt/job, confirm the provider worker has terminated, and hash terminal evidence. Vast additionally requires verified teardown of the owned instance. The journal preserves the failure; a repaired experiment receives a new ID. No bulk cleanup or adoption of account-listed machines is supported.

## Scientific and system checks

The initial dispatcher tests cover uncertain submit restart, source drift, absent/mismatched artifacts, empty queues, lane reservation outside the current queue and terminal release. The profile instrumentation reports optimizer attempts, completed updates, AMP skips and loss/gradient validity. H218 tests cover future-prefix isolation, reset, arbitrary chunk boundaries, memory learning and undetached full/chunk gradient equivalence.

Independent review found that a stored complete status could outlive its evidence and that symlinked state paths were not fully confined. The successor dispatcher rechecks hashed completion/reconciliation evidence before releasing even an omitted attempt's lane, stores completion bindings, and confines state and reconciliation files to the project. Regression checks include concurrent dispatcher processes (one launch), lost receipts and escaping symlinks. Dispatcher plus FP16 profile-evidence tests: 20 passed. H218's independent CPU review passed four tests, with approximately 648 MB peak process-tree RSS; this remains a protocol check, not a fit.

H230's candidate/control audit found a substantive comparison issue: setting family loss weight to zero also removed eight outputs and skipped family-cache reads. The [portable source successor](h230_source_v2/README.md) now keeps the same 31-output architecture, cache and main target while changing only the family-loss coefficient. Three focused tests with the actual own initializer and bounded real-cache rows passed, at 787 MB peak sampled RSS. Earlier test-envelope failures were diagnosed and preserved. The full candidate remains eligible; its student fit has not run.

H233's [V2 recipe](recipes/H233_C1S3_CONDITIONAL_READOUT_V2.json) pins teacher provenance and makes the valid export's ordinary scoring mask exclude all 1,405 fit-development groups, retaining only the 468 evaluation groups. Six focused CPU tests passed. [Independent original review](reviews/H233_INDEPENDENT_REVIEW_V1.md) and [controller repair review](reviews/H233_V2_CONTROLLER_REPAIR_REVIEW.json) cover the change; the V1 source/recipe snapshot is preserved. No GPU profile, fit, export or score has run. Its C1 parent has since scored below the public incumbent; any readout gain remains an untested new treatment.

H218's prototype defaults are input211, width256, four layers, eight heads, 256-row blocks and two outputs. Each layer attends to a completed-block recurrent summary plus causally masked current-block history; its GRU summary updates only after a complete block. Partial blocks are carried across arbitrary chunk boundaries. A future training integration must preserve explicit TBPTT detach boundaries, independently fit normalization on allowed groups, profile real shapes and compare against the frozen teacher/control endpoints. This CPU prototype does not establish GPU throughput or a useful teacher.

## Market selection and ownership

Search the wider live market; user-supplied 3090/A2000 listings are reference observations only. No offer is selected. Include CPU, disk, startup, source transfer, export and storage in the first cohort's $4 estimate. The approximately 35 GB source data makes tiny-disk offers unsuitable without an explicitly costed staging method. Do not infer GRU/TCN speed from marketplace TFLOPS or interpret reliability as a measured completion probability.

Only a durable campaign creation receipt can establish an instance's ownership. Labels alone, account visibility and user-supplied offer IDs cannot. Other machines remain outside all SSH, collection, mutation and teardown operations. `.env` is ignored; never include its content in packets, prompts or commit staging.

## Remaining handoff

Complete independent reviews of new implementation and packet transports, prepare actual scientific worker packets and collectors, then use fresh device/quota/cost receipts in an execution-capable environment. Reassess allocation every two hours; no background scheduler or new timer is installed by the current preparation snapshot. Preserve all failed and superseded attempts. Maintain tree/index/current-state and commit scoped changes when Git writes become available.
