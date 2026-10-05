# Local resume review — 2026-10-05

## Recommendation

E464 is approved as the first local-GPU queue. Do not launch E427 first. E464
has all eight sealed fit identities, a fixed sequential controller, successful
actual-parent scope preflights, a passing controller source-check packet, and
the user's explicit resumption record. Current queue startup must use
`--resume`, since its durable snapshot is nonterminal and the installed
`ExecStart` omits that flag.

## E464 readiness and required resume repair

The fit recipe is fixed across all eight parents: one head-only epoch from
parent epoch 3, seed 3, batch 16 with drop-last over 1,505 groups, 94 batches
and 940 2,000-row optimizer attempts. The preregistered comparison set retains
all eight architecture/routing/adapter arms and 16 paired contrasts. Actual
parent scope preflights passed for all eight arms; eleven controller fixtures
passed. The amendment corrected the earlier 93-batch accounting error and
requires exact ordered pair and FP32 `[1873, 20000, 2]` export checks.

Admission identity check: controller
`checks/run_e464_sequence.py` matches SHA-256
`f7483720e4b361620340791a47a1abc876d5f68036947ebf4d7544a53509421e`; staged
service unit matches
`b405d5c132a23281918b1c95bcb830ec4386f9c79825443831f713599d7e8ec6`. All
eight sealed fit-spec bytes match the hashes in
`checks/e464_queue_root_admission_v1.json`. I hashed each unique code/input
path once across those specs: 81 paths, zero missing files, size mismatches or
SHA-256 mismatches. The exact current E425 user service is loaded/inactive with
exit 0, and its terminal sequence plus three required successful attempt
records satisfy E464's predecessor condition.

The durable queue snapshot is `waiting_for_e425`, with no completed E464
attempts and `predecessor_seen_active=true`. I previously reported that the
E425 unit was absent based on a system-level `systemctl` query; that was the
wrong scope. The correct user-unit query shows `LoadState=loaded`,
`ActiveState=inactive`, `ExecMainStatus=0`. The durable E425 sequence says
`fits_and_exports_complete`, and all three required runner attempts have exit
0 and verified teardown. No predecessor-controller change is needed.

I independently rechecked the current root admission bindings once across all
eight sealed fit specs and their deduplicated referenced files: all eight
sealed-spec hashes match `e464_queue_root_admission_v1.json`; all 81 unique
code/input paths match their recorded SHA-256 and byte sizes; there are no
mismatches. The current runner has zero E464 fit attempts. This verifies exact
fit identities and code/input closure at review time.

The resumed supervisor uses this command (the installed service's original
`ExecStart` omitted `--resume`):

```bash
systemd-run --user --no-block \
  --unit=wnn-e464-sequence-resume-v2 \
  --slice=wnn-campaign.slice \
  --working-directory=/home/vlad/development/wunder/wnn_connectome_starterpack \
  --property=Type=oneshot \
  --property=CPUQuota=25% \
  --property=MemoryMax=512M \
  --property=MemorySwapMax=0 \
  --property=OOMPolicy=kill \
  --property=KillMode=control-group \
  --property=TimeoutStartSec=12h \
  --property=RuntimeMaxSec=12h \
  /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  research/campaign_2026-10-04/checks/run_e464_sequence.py --resume
```

The first worker is the existing sealed `e464-h093-learned-head-s3-final-v1`;
the queue admits each next fit only after preceding artifact checks. Do not run
the controller directly in a shell for a long-lived queue.

At the later live snapshot during this review, the supervised `--resume` unit
had advanced to `phase=fit:h093_learned`; that exact first attempt was active
with `charged_seconds=0` in its initial running record. Root launched this
authorized queue after explicit resumption; this confirms the current
predecessor check passes. No controller code change was required.

## Watcher false positive and acknowledgment evidence

The live watcher incident is
`27010ecdfe9f30867f02c22a75e2bf3d346030d61945275cf24379e35390e491`. Its only
claim is the journal message `wnn-e464-sequence-v1.service: Failed with result
'signal'.` This matches the explicit user stop of the waiting controller, not
a training failure. `checks/E464_USER_PAUSE_20261004.json` classifies it as an
intentional pause, records `last_snapshot_before_stop=waiting_for_e425`, and
records `new_fits_started=0`. The queue snapshot has an empty completed list;
there are no E464 fit attempts. The watcher status reported no active GPU jobs;
the refreshed device query showed 75 MiB used. Preserve the incident and
journal record, acknowledge this exact watcher ID with the explicit-resumption
reason if releasing the separate legacy watcher queue; do not misclassify the
event as an E464 training failure. The historical acknowledgment helper is
not appropriate for this newer event.

The user pause was archived at
`research/campaign_2026-10-05/pause_archive/runner_PAUSE.json`; the explicit
resume is recorded in `research/campaign_2026-10-05/RESUMPTION.json`. The older
10/03 watcher barrier and 10/01 pause remain for their own queues, consistent
with the resume record. They do not alter the E464 queue's root `READY.json`
or E464-specific startup guard. The false-positive SIGTERM is classified as
the intentional stop of a waiting controller by `E464_USER_PAUSE_20261004.json`;
it is not evidence of a fit failure. Keep the incident and watcher-owned pause
in the historical record.

## E427 and objective/cadence review

E427 is further from admission. Its source-only overlay, CPU fixtures and
input-only context statistics are useful preparation, but all three
production-shape GPU smokes are pending, all three fit specs remain drafts, and
no E427 fit/export scorer is sealed. The context path materializes a
`[32,20000,28]` float32 feature tensor (about 72 MB) plus a row-index stream and
temporary cumulative-sum workspaces whose peak has not been measured. The
existing 4.8 GB device stop threshold leaves little room against the recent
4.49 GB baseline until the real smoke proves otherwise. E427's deployment
export/reconstruction is also pending.

Do not conflate the objectives: E464 deliberately uses the inherited
head-fine-tuning `gcorr` recipe; E427's amended full-combo continuation uses
`wmse` with structural and next-target auxiliaries at 0.3 each, matched across
aligned, stale and no-op arms. Its fixed one-epoch `chunk=1000` recipe is a
cadence intervention only in the sense of preserving the parent recipe; it
does not identify cadence or credit-length effects because there is no matched
`chunk=2000` arm. The dedicated cadence comparison should retain pooled
loss numerators/denominators and report optimizer/EMA counts, per the paused
plan.

## Resource and scientific limits

Current local snapshots show the campaign slice at 12 CPU and 36 GiB RAM cap,
about 1.58 GB charged before the resumed fit, with GPU usage at 75 MiB. The
resumed E464 worker is now the sole authorized local GPU job. E464 worker specs request 8 CPU, 24 GiB RAM, a 4.2 GB
allocator ceiling and 4.8 GB stop ceiling. Keep one GPU worker, preserve the
24-hour local block and three-hour E464 reserve, and refresh resource counters
immediately before queue admission. The current scientific anchor remains the
user-reported fixed seed-4 public result at 0.6660. E464 uses a historically
exposed validation panel, so its paired intervals are conditional comparisons,
not fresh generalization or public-score evidence.
