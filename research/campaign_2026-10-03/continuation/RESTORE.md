# Recovery snapshot and paused restore

This is a portable backup of source, runtime configuration and recovery evidence.
It is **not a standalone training restore**: datasets, checkpoints, predictions,
caches and excluded large research files remain external references. No recovery
command starts services, installs units, invokes training or clears a live marker.

The controller creates the small snapshot after all analyst agents have stopped
writing. Run from the repository root using the existing bounded CPU wrapper:

Healthy GPU training can continue during capture. Job records, scope logs and
watcher counters are individually stable observations at the timestamps in the
archive manifest; they are not one atomic runtime state. Source and configuration
must remain unchanged throughout capture. Restore stays paused and reconciles
these observations with actual services before resumption.

```bash
research/campaign_2026-10-03/continuation/run_cpu.sh archive-fix \
  .venv/bin/python -B research/campaign_2026-10-03/continuation/save_state.py \
  --archive --hash-budget-mib 256 --hash-seconds 60
```

Each published snapshot has a unique name under `continuation/snapshots/`.
`LATEST_SNAPSHOT` changes only after success. A failed capture leaves an
unpublished `.partial` directory and `FAILURE.json`; the preceding snapshot
remains usable. `--output NEW_DIRECTORY` creates an independent snapshot and
does not change `LATEST_SNAPSHOT`. Never use an existing destination. Metadata
without `--archive` does not contain portable source bytes.

The tar includes continuation scripts, configuration, state, pause/failure
markers, incident logs and acknowledgement state; Python/Rust/submission source;
the split metadata and queued shell recipes; and root dependency specifications
including `uv.lock`, `requirements.txt`, Python version and Cargo metadata.
Prior snapshots, review fixtures, lock owners, native build output, virtualenvs, private
Claude/Codex session trees, agent transcripts and credential files are excluded.
Top-level review source/reports and opt2 implementations are included. Small
selected research source/metadata is included; other research and bulk
artifacts are listed in the external inventory. Large archives and binary
checkpoints are never included. The archive enforces an 8 MiB per-file limit,
128 MiB payload limit and 20,000-member limit. Missing required runtime source
or suspected credentials cause capture failure, not a misleading success.

`_recovery/` contains the repository revision, scoped index/worktree patches,
status, installed dependency versions, sanitized campaign systemd definitions
and effective properties, the external inventory and an archive integrity
manifest. Captured unit definitions include persistent drop-ins and transient
definitions through `systemctl cat`. Only resource environment variables are
retained. Diagnostics retain command exit codes, so missing tools or inaccessible
units remain explicit. Definitions are historical evidence; do not install or
replay them. Host paths and Python environments must be rebuilt for a new host.
The archived files already contain worktree/index changes; **do not apply the
patches again** to the restored tree. A base revision checkout is useful only
when recovering additional repository files absent from this small snapshot.

External references use repository-relative paths, byte counts and nanosecond
modification times. `hashed_now` means a streamed SHA256 was computed and file
identity/size/timestamps were unchanged across the read. `reused_unverified`
means a prior inventory's hash was reused with matching size/time; it does not
prove current content. `unverified` includes references beyond the byte/time
budget, unavailable files and files that changed while hashing. To reuse a
previous JSONL inventory, add `--hash-manifest PATH_TO_ARTIFACT_INVENTORY.jsonl`.
Before any resumption, recover needed external files from their retained backing
store and independently hash them. These snapshots neither certify the backing
store nor contain those files. Missing external roots are recorded explicitly.

Perform the restore check in a **trusted staging checkout with a functioning
CPU wrapper and resource slice**, separate from the destination. The restored
tree is paused and cannot provide its own initial CPU admission. A new host
needs that trusted bounded runner configured before this step. Do not start the
archived queue or copy its old systemd units to bootstrap the runner.

```bash
snapshot_dir=research/campaign_2026-10-03/continuation/snapshots/REPLACE_WITH_SNAPSHOT
read -r snapshot_sha snapshot_file < "$snapshot_dir/SHA256SUMS"
research/campaign_2026-10-03/continuation/run_cpu.sh archive-fix \
  .venv/bin/python -B research/campaign_2026-10-03/continuation/save_state.py \
  --restore-dry-run "$snapshot_dir/research-session.tar" \
  --expected-sha256 "$snapshot_sha" \
  --restore-dir research/campaign_2026-10-03/continuation/review/archive_fix_restore_NEW
```

Use a new destination each time. This command verifies the supplied trusted
archive checksum, rejects unsafe paths, links, duplicate members and size
overflow, checks each member against the embedded manifest, then extracts only
source/metadata. It never executes extracted code. Working memory is bounded
by small file buffers; run it under the 1-core/2-GiB wrapper as shown.

Fresh `RESOURCE_FAILURE`, `QUEUE_PAUSE` and `STOP_NEW_JOBS` pause markers are
present in the destination. Archived admission markers, including `READY`, are
saved under `_recovery/original_control_state/`, never activated. Original
`LATEST_SNAPSHOT` is also kept there because prior snapshots are excluded.
Fresh empty CPU lease files are created; no old lease ownership is restored.
Original incidents and acknowledgements remain evidence. `_recovery/dry_run.json` records
the result and states that no external artifacts or services were restored.
If any step fails, leave the destination paused and inspect it; do not merge a
partial copy into the active checkout.

After a successful dry-run the controller can inspect dependency metadata,
rebuild the environment/native extension in a bounded runner, verify required
external hashes and split/checkpoint identities, and reconcile running job
records against actual units. Captured "running" records are historical, not
proof of live work on the restored host. Confirm the resource contract and
acknowledged incidents before any separately authorized resumption. The dry-run
intentionally provides no automatic resumption or training command.

The same snapshot directory also contains `progress-artifacts.tar`, holding new
EDA/prototype/reproduction arrays, figures and notebook HTML created after the
initial private archive. Its checksum is the second entry in `SHA256SUMS` and
its embedded `progress_manifest.json` hashes every member. The bounded companion
restore checks the archive checksum, all member hashes, paths, links and payload
limits, and requires the destination to remain paused. It refuses to overwrite
different staging artifacts. After the source restore above, run:

```bash
read -r progress_sha progress_file < <(sed -n '2p' "$snapshot_dir/SHA256SUMS")
research/campaign_2026-10-03/continuation/run_cpu.sh archive-fix \
  .venv/bin/python -B research/campaign_2026-10-03/continuation/restore_progress_artifacts.py \
  "$snapshot_dir/progress-artifacts.tar" --expected-sha256 "$progress_sha" \
  --restore-dir research/campaign_2026-10-03/continuation/review/archive_fix_restore_NEW
```

This is the same paused staging destination used by the first command. No services
or extracted code are executed. `_recovery/progress_dry_run.json` records results.

Keep `snapshots/2026-10-03-153300/research-session.tar` as well: that initial private
archive retains raw Claude transcripts and the older large EDA artifacts omitted
from the bounded portable snapshot. Datasets, prediction caches and trained
checkpoints remain external dependencies recorded by the inventories.

Snapshots can contain additional `progress-artifacts-002.tar` (and later parts).
Look up each filename's checksum in `SHA256SUMS` and invoke the SAME bounded
companion restore for every part into the same paused destination. Do not assume
the context archive is always the third checksum line; look it up by filename.
`PROGRESS_PARTS.json` records the member count and bytes of each part.
Research-local `.pt` state captures under the explicit research roots are also
included by later captures. Restore treats them as opaque bytes, verifies hashes
and never loads or executes them. This does not duplicate all training models in
the external artifacts tree. Source snapshots now include research/frontier.md.
Later multipart captures use960MiB payload parts and a1GiB bounded restore policy
so a roughly600MB full-batch state capture fits. Tar creation now streams and
hashes each member without read_bytes/BytesIO copies of large state files;
restoration continues to use1MiB buffers. The CPU scope remains1core/2GiB/900s.
The old single-part helper still rejects payloads above512MiB and directs callers
to the streaming multipart helper. Older512MiB archives retain their original
checksums and verification evidence.

The18:31
backup required two parts; each was actually hash-verified and restored paused.
The original single-part capture rejected708,095,082 bytes before writing any
archive. Partitioning preserves every artifact without enlarging per-part bounds.

The latest snapshot also contains private `session-context.tar`: original agent
briefs, the Claude session index and the final chain handoff. These are excluded
from the portable source policy and saved separately; the initial raw archive
predates their creation. Its named `SHA256SUMS` entry and embedded member manifest
cover every file. Restore these context files only into the paused staging tree;
they do not activate any runtime admission marker or service.
