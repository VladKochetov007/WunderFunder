# R1 result collector terminal capture

This record preserves the terminal outcome of the already-admitted result collector. It is capture evidence only; it makes no independent scientific acceptance claim.

## Exact unit and terminal state

- Unit: `computer-vision-homework-r01-f3-result-collection-r1.service`
- Invocation: `0beed2d4d96c492b814ad6969170526c`
- Launch: `2026-10-19T03:14:51Z`; explicit process exit: `2026-10-19T03:15:00Z` (about 9 seconds)
- Captured at `2026-10-19T03:19:13.453520Z`: `Result=success`, `ExecMainStatus=0`, `ActiveState=active`, `SubState=exited`, exact invocation.
- Before stop, the unit showed `CPUQuotaPerSecUSec=750ms`, `MemoryHigh=448000000`, `MemoryMax=512000000`, `MemorySwapMax=0`, `RuntimeMaxUSec=2min`; full `ExecStart` is preserved in `attempt_02/SYSTEMD_SHOW.raw` and selected properties in `attempt_02/SELECTED_PROPERTIES.raw`.
- A guarded stop ran at `2026-10-19T03:20:15.994411Z` after raw capture verification. The pre-stop invocation, exited state, exit status, and caps matched exactly. `systemctl --user stop` returned 0; post-state was `inactive/dead`. Exact raw stop receipt: `attempt_02/STOP_RECEIPT.raw`, SHA-256 `fe491ec6b4865d4e0546c8382c86f5b3133252b26486ff2518178f6fc6dced59` (3,651 bytes). Post-stop systemd clears the invocation ID and resource properties; the preserved pre-stop properties are authoritative for ownership/caps.

## Result and source bindings

The exact producer receipt is preserved as `attempt_02/RESULT_RECEIPT.json`, 466,940 bytes, SHA-256 `d99135b40bf968e3b9dbe8ab1ea5dc6d022af7e487385948f3648cbd915da0f0`. It reports schema `priority25.f3_result_collection.v1`, `status=collection_passed`, and states that it did not open checkpoint payloads, rerun grading, deserialize prediction payloads, train, or export. Its collector source pin is `collect_f3_results.py` SHA-256 `2a61de37d4fb9260a69e553aca2b789d427ad4719cc516a8a45d10fb745723b8`; support pins are in the exact receipt. The captured producer metadata records 505 evaluation groups, 145 components, and prediction shapes `[505, 20000, 2]`; prediction files were not transferred or independently rehashed here.

The published snapshot is `/home/grinchenkoeg/computer-vision-homework-20261018/research/campaign_2026-10-18/priority25_execution_r4/f3_result_collection_r1/records/F3_S1001_RESULT_R1`. `attempt_02/SNAPSHOT_INVENTORY_STAT_FINAL.json` records stat-only metadata for all 59 receipt-declared paths: 59 present, 0 missing, with file sizes matching producer metadata. It does not read or rehash snapshot contents. The producer receipt records streaming prediction SHA-256 and finiteness checks as part of its own scope.

Exact raw root admission and launch receipts are `attempt_02/F3_RESULT_COLLECTION_ROOT_ADMISSION_V1.json` and `attempt_02/F3_RESULT_COLLECTION_ROOT_LAUNCH_V1.json`; the bounded producer log tail is `attempt_02/F3_RESULT_COLLECTION_R1.log.tail.raw`. `attempt_02/REMOTE_CAPTURE_02.raw` is the byte-preserving transport envelope (722,107 bytes, SHA-256 `cecb48275f9cc52e3f5e1a070cbba8ad0b37245a3006e24424c17528c1433295`). Each captured base64 payload was locally decoded and checked against its recorded byte length and SHA-256 before extraction. The command status, journal, and full systemd outputs are also preserved as raw bytes under `attempt_02/`.

## Resource observation

At capture, the parent campaign cgroup was the admitted path `/user.slice/user-1061.slice/user@1061.service/wnn.slice/wnn-campaign.slice`; the unit's `ControlGroup` was empty, so no host-root cgroup fallback was used. Parent `memory.events` remained `high=2179`, `max=561`, `oom=0`, `oom_kill=0`, and `memory.events.local` remained all zero, matching the inherited admission baseline. The parent cgroup had no populated processes or populated descendants at the capture observation. Parent `memory.peak` (`6506029056` bytes) is a parent-scope lifetime peak; unit `MemoryPeak` and `MemoryCurrent` were `[not set]`, and no unit RSS is inferred. These are separate scopes.

## Capture-attempt note

The first read-only transport attempt is retained at `REMOTE_CAPTURE_RAW.json` and `SSH_CAPTURE_STDERR.raw`. It returned zero stdout and the exact error `NameError: name 'root_adm' is not defined` at remote stdin line 115; it did not invoke a stop. The successful attempt is isolated under `attempt_02/`. A corrected stat-only inventory used the published `snapshot_directory` from the already captured producer receipt; the first inventory within the envelope had followed temporary staging paths, which no longer existed after publication. No compute was re-run, restarted, or otherwise changed during capture.
