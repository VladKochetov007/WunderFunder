# Owned-worker signed-object prefetch review

This change adds an **optional stage-time transport optimization** for the two
frozen inputs `datasets/train.parquet` and `datasets/valid.parquet`. It does
not create or adopt an instance. `run` still enters the systemd-supervised
`VastCampaign.run_attempt → supervise → _stage_and_launch` path; `supervise`
uses the same receipt, account-scope, provider-instance, and persisted SSH
host-key checks before staging. Direct `create` remains disabled. After
prefetch, the existing remote SHA-256 verification remains authoritative, and
the normal rsync path still stages the packet and handles a completed,
sanitized prefetch error. An ambiguous SSH transport timeout and any invalid
plan/receipt or unsafe destination fail closed to avoid overlapping writers.

The hook is disabled unless both explicit `--prefetch-plan` and
`--prefetch-admission` paths are supplied to the supervised `run` or
`supervise` command. No service unit or campaign command was changed here, so
the existing production invocation remains on rsync until a reviewer wires
the two paths into the already-supervised invocation. The public admission is
an auditable binding, not a cryptographic signature; the root/operator must
freeze its content hash and the matching service arguments before activation.

The private plan is a mode-0600 JSON file beneath `.runtime`:

```json
{
  "schema_version": 1,
  "inputs": [
    {
      "remote_path": "datasets/valid.parquet",
      "sha256": "<frozen-input-sha256>",
      "size_bytes": 123,
      "url": "<private-signed-https-url>"
    }
  ]
}
```

It must contain exactly the eligible frozen parquet members for that attempt.
Each URL is restricted to HTTPS on `storage.googleapis.com`, without URL
credentials, explicit ports, fragments, or control characters. The public
admission, also beneath `.runtime`, has exactly these fields:

```json
{
  "schema_version": 1,
  "attempt_id": "<attempt>",
  "campaign_label": "<label>",
  "private_plan_sha256": "<exact-private-file-sha256>",
  "source_binding": [
    {
      "remote_path": "datasets/valid.parquet",
      "sha256": "<same-frozen-input-sha256>",
      "size_bytes": 123
    }
  ]
}
```

The controller verifies both files are owned regular files beneath a protected
runtime directory, rejects symlinks and duplicate JSON keys, checks the
private file's exact mode, and compares attempt, label, plan-file SHA, and
each path/hash/size against the admitted worker-input records. Only then does
it construct the remote destination from the frozen output root, label, and
the two allowlisted relative paths. The URL is present only in the private
file and the encrypted SSH stdin payload: it is never put in argv, ledger,
receipt, or exception text.

The standalone remote helper uses only the Python standard library. It
disables environment proxies, checks redirects and the final response URL,
streams at 1 MiB per read, adjusts socket read timeouts to the remaining
absolute deadline, hashes incrementally, and verifies exact byte count and
SHA-256 before an atomic replacement. It opens every destination directory
without following symlinks and writes a unique mode-0600 temporary file with
exclusive/no-follow flags. The outer SSH deadline is at most 1,200 seconds,
with 20 seconds reserved for remote startup; the public receipt includes the
private-plan, admission, and remote-program hashes plus per-file path, digest,
size, elapsed time, and status, never a URL. Before starting it reserves at
least 300 seconds for the remaining upload/launch path and estimates additional
packet/input staging time from their frozen byte count.

## Validation

Run from the isolated worktree:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s \
scripts/limited /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/ruff check \
  scripts/kaggle_vast_prefetch.py scripts/kaggle_vast_prefetch_remote.py \
  scripts/vast_campaign.py tests/test_kaggle_vast_prefetch.py tests/test_vast_campaign.py

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s \
scripts/limited /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/pytest -q \
  tests/test_kaggle_vast_prefetch.py tests/test_vast_campaign.py
```

Result: Ruff passed; 78 focused tests passed. Tests use fake plans, fake HTTP
responses, and a mocked SSH transport; they cover ownership-gated stage
ordering, invalid-admission fail-closed behavior, plan mode/symlink/source
binding, argv-vs-stdin secrecy, redirect rejection, bounded streaming, hash
mismatch cleanup, symlink-parent refusal, no-URL receipts, and skipped rsync
for a completed prefetched input. A separate local-only stdlib socket smoke
check verified that the worker HTTP response exposes the socket handle used by
the per-read deadline update. No signed-URL file was opened, no provider or
worker was contacted, and no data transfer or fit was attempted. Actual Kaggle
URL compatibility, worker-image runtime behavior, and end-to-end throughput
remain unverified; the root reviewer must review this isolated change before
activation.

## Frozen file hashes

These SHA-256 values identify the implementation and focused tests for this
isolated commit:

| File | SHA-256 |
|---|---|
| `scripts/kaggle_vast_prefetch.py` | `abb52a582de4ef8073d829af5a4e376a850ac21a84cc972a0d2137201f6e422a` |
| `scripts/kaggle_vast_prefetch_remote.py` | `25f1e776636ca30e733b955986bd2da3f1aaea932f3a57660d43783cdb5c3b7d` |
| `scripts/vast_campaign.py` | `1e91fff7041a99ed959e72f45a99c367e75c70bdb2d24ee100d25e80bc5528e0` |
| `tests/test_kaggle_vast_prefetch.py` | `07af544334829efbbb592d6a6b01b0a7f2bd6aea40584857c2dcf45b72efa31e` |
| `tests/test_vast_campaign.py` | `fef4bfbc643a34bb5d7a5cf95248c74a9f7018726cc76c44e7ac8713276c98ad` |
