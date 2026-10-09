# H230 execution contract v2

Status: static execution-binding candidate only. H230 cache, profile, fit,
prediction export, and grading remain unexecuted by this actor and are not
admitted by this contract. Root owns stage admission. The E521 producer
receipt and every checkpoint/export hash must be refreshed and sealed after all
required exports finish; the older V1 status snapshot is historical only.

V2 keeps the H230 recipe, targets, masks, folds, initializers, seeds, checkpoint
policy, and metric unchanged. It supersedes V1 only to remove ambiguity between
the main-root resource controller and the pinned worktree worker, and to record
root's cache-reservation reconciliation. V3's reviewed model output-width
contract remains unchanged at 31 outputs.

## Source binding and path semantics

The source implementation is the persistent worktree
`/home/vlad/development/wunder/worktrees/wnn-bandit-r1-familyloss-20261007`,
commit `dde32d0f5e39efe65c0497fec06449d38e18b369`. The resource controller,
durable attempt state, split, source data, checkpoints, teacher cache, student
cache, predictions, and results use the main project root
`/home/vlad/development/wunder/wnn_connectome_starterpack`.

Launch `scripts/bandit_resources.py` with `--root` equal to the main project
root and its campaign runtime state path. The controller sets the service
`PYTHONPATH` to main `src`; therefore every worker child command explicitly
starts with `/usr/bin/env PYTHONPATH=<absolute worktree>/src`, then executes the
worktree `.venv/bin/python` and an absolute worktree script path. The exact
post-timeout worker argv arrays are recorded in this JSON contract. Do not
substitute a main-tree trainer or an unqualified relative script path.

The worktree's `.venv`, `artifacts`, `cache`, and `datasets` paths currently
resolve to the corresponding main-root paths. Before every stage, verify the
worktree `HEAD` is the pinned commit and each of those four `realpath` results
matches the main root. Abort if either check differs. This makes trainer and
predictor outputs derived from the absolute worktree script land in the shared
main `artifacts/runs` and `cache` locations. Cache-builder input/output paths
are explicit absolute main-root arguments. The resource controller itself
continues to run from main so its code/source manifest, ledger, and resource
admission are rooted consistently.

The worker argv prefix is:

```text
/usr/bin/timeout --signal=TERM --kill-after=20s <stage_cap_seconds>
/usr/bin/env PYTHONPATH=/home/vlad/development/wunder/worktrees/wnn-bandit-r1-familyloss-20261007/src
/home/vlad/development/wunder/worktrees/wnn-bandit-r1-familyloss-20261007/.venv/bin/python
/home/vlad/development/wunder/worktrees/wnn-bandit-r1-familyloss-20261007/scripts/<pinned-script.py>
```

Use `scripts/bandit_resources.py --root <main-root> --state
<main-root>/research/campaign_2026-10-06/bandit_continuation/runtime launch`
with the root-approved attempt ID, kind, CPU and memory reservation, resource
profile receipt, contract path, and the corresponding exact worker argv from
the JSON. The stage wall cap is enforced by the explicit `timeout` wrapper;
resource-controller and cgroup gates remain independently active. The worker
commands in this contract are templates for later admission, not permission to
run them.

## Recipe, data and cache gates

The source-family recipe remains exactly the fixed eight-teacher mixture in
`SOURCE_FAMILIES_DRAFT_V2.json`, coefficient sum 0.8125, with the own GRU384
seed pair as the samefold baseline family. The main cache target keeps the raw
unnormalized sum. The family head learns raw per-family targets. Student width
is `2 + 19 + 2 + 8 = 31`; raw-family behavior and all H230 fit settings are
unchanged.

Before any cache attempt, root must verify a fresh E521 terminal-success
receipt and seal both seed-3/seed-4 fixed-epoch-3 checkpoints, provenance, and
train/valid prediction arrays plus sidecars. Verify the split, complete row
membership/order, target identities, shape, dtype and finite values from actual
mounted bytes. The fold-0 evaluation groups remain excluded from fit/cache rows.
The earlier live receipt is only a pointer and must not be treated as a frozen
source identity.

The builder estimate is exactly 13,372,800,000 bytes from 11,940 fit groups ×
20,000 rows × 4 bytes × 14 cache channels. Root reconciled the draft's
24,200,000,000-byte planning reservation as advisory, superseded for the live
disk check by this exact payload estimate plus the 25-GiB free-space floor:
40,216,345,600 bytes. This reconciliation is conditional on refreshing actual
fit-row counts and all 14 float32 output channels before build. The builder's
30-GB output budget, all source and membership checks, and the 25-GiB floor
remain hard gates. If any shape/channel/count changes, stop and return for a
new reviewed contract; do not silently enlarge or weaken the budget.

## Fixed stages and unchanged scientific limits

The cache is a single serial CPU build, chunk size 2, zero loader workers,
6.5-GB reservation, and 3,600-second cap. Profile uses the exact H230 student
shape and cache with 3 warmup and 20 measured batches, no checkpoint, and a
900-second cap. Each two-epoch student fit is a serial one-GPU job with a fixed
zero-based epoch-1 checkpoint, no fit-time evaluation, 5,400-second cap and
6.5-GB worker reservation. Each matching valid prediction export is a separate
serial stage with streaming/memmap output, zero loader workers and 1,800-second
cap. Use the allocation's global memory/CPU limits and single-GPU lock. The
recipe and model remain as audited in V1/V3; this contract changes no objective,
label treatment, family coefficient, mask, seed, checkpoint or grader.

No score is produced here. The exposed fold-0 comparison remains development
evidence and requires the independently frozen grader and root admission.
Failures preserve logs, outputs, and receipts; no same-ID retry, cap relaxation,
fold/mask change, or score interpretation follows automatically.

## Review and admission state

This V2 is a new static candidate for independent review. The H230 V3 static
review is a conditional PASS only for the earlier contract/source shape. It is
not approval of V2's invocation binding or live cache/profile/fit. After
independent V4 review, root still must seal fresh E521/export hashes and issue
per-stage resource admission. No provider, cloud, fit, profile, cache, export or
grade action is authorized by this document alone.
