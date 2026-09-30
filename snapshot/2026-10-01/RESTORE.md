# Restore the remote experiment archive

All parts must be in one directory. Each part is at most 45 MiB for ordinary GitHub storage. The uncompressed payload is about 2.3 GB.

```bash
sha256sum -c PARTS.sha256
cat experiments-remote-20261001.tar.zst.part* | sha256sum
# Compare the preceding digest with ARCHIVE.sha256.
cat experiments-remote-20261001.tar.zst.part* | zstd -d -c | tar -xf -
cd snapshot
sha256sum -c SHA256SUMS
```

The stream can be restored without creating a duplicate compressed archive. Per-run indexes and SOTA_STATE.md are also provided separately for search. Bulk prediction caches and organizer datasets are inventoried but excluded. Source-state has a Git bundle, HEAD source snapshots, and uncommitted patches; no home configuration or credentials are included.
