# E520 page-cache hash repair candidate v1

Status: **isolated code candidate only; not reviewed, packetized, admitted, or run on Kaggle**. This directory does not edit or replace E520 packet v1/v2 or the reviewed source under its parent directory.

## V2 failure diagnosis

The collected v2 receipt is `research/campaign_2026-10-06/cloud_preflight/e520_profile_terminal_v2.json` (SHA-256 `e31f265c65056ea10bfea62b6c4d14da358891ba9b932430a01fec484c717615`). The worker exited at 830.075 seconds with SIGTERM 143 while full-file train/valid hashes were incomplete. The abort snapshot was pre-CUDA idle, RSS 27,422,720 bytes, host available 32,563,232,768 bytes, and cgroup headroom 6,410,838,016 / 32,212,254,720 bytes (19.90186%), 31,612,928 bytes below the unchanged 20% guard. No profile batch, fit, checkpoint, prediction, or score began.

The observation points to cgroup-accounted file-cache pressure during sequential parquet reads: process RSS was small and host memory remained abundant while cgroup headroom crossed its gate. The receipt does not include cgroup `memory.stat`, so page cache is a plausible mechanism, not proven attribution. The current streaming helper reads 1 MiB blocks but leaves cache advice to the kernel's defaults.

## Candidate change

`profile_worker_candidate.py` is a copy of the reviewed E520 worker with a narrow change: raw train/valid parquet SHA-256 uses 1 MiB sequential reads and invokes `os.posix_fadvise(fd, aligned_offset, aligned_length, POSIX_FADV_DONTNEED)` after each completed page-aligned range. The digest still consumes the same bytes, in the same order, through SHA-256. Input assets, fold membership, target order and every expected hash remain inherited unchanged from the base worker at SHA-256 `e055d47d8950d211541298a37d237c1b7d76b5a5dc3121a0c9793ce7c74f2107`.

The durable receipt `e520_profile/hash_io_advice.json` records platform capability, page size, per-file byte progress, advised ranges, successful call counts and final digest. It is included in `profile_result.json` after a successful input preflight. Progress is atomically refreshed every 256 MiB and on completion/failure, so a resource-gate SIGTERM leaves a recent offset. No prediction or data array is created by this change.

## Fail-closed behavior and limits

The candidate fails before reading a raw parquet file if POSIX advice or a compatible page size is unavailable. Any `posix_fadvise` or file-read error writes a failure record and aborts; there is no fallback to unadvised hashing and no retry. An accepted `POSIX_FADV_DONTNEED` call is only a kernel hint and does not prove eviction. The existing strict caps remain unchanged: host and cgroup each need at least 4,294,967,296 bytes and 20% available; process RSS stays at or below 8,000,000,000 bytes; CUDA allocated/reserved peaks stay at or below 2,000,000,000 / 2,500,000,000 bytes; process GPU memory stays at or below 3,000,000,000 bytes; free disk stays at or above 8,000,000,000 bytes; and the worker requires one visible GPU. The 1,800-second wall cap and no-retry path also remain unchanged. If page cache still reduces measured cgroup headroom below either strict gate, the worker must abort as before.

This candidate does not authorize another provider run. Independent code review, candidate-specific packet and transport review, and fresh controller admission are required before any retry. The v2 quota/resource receipts remain evidence of the prior aborted attempt only.

## Bounded local verification

The synthetic tests use small temporary files only. They check byte-for-byte SHA-256 equivalence, page-aligned advice ranges, durable unsupported/error records and fail-closed behavior. They do not read mounted parquet, allocate model tensors, launch a fit, create a packet, or call Kaggle.
