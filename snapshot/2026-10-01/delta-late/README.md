# Late Transformer results supplement

This supplements, and does not rewrite, the fixed main snapshot. See `readable/inventory.json` for the exact UTC cutoff and active prediction jobs. Main archive:266run directories; late snapshot adds 2run directories (fttfm_f1, fttfm_f1h), for 268 total. Six checkpoint files are in this supplement; five are new checkpoint paths and one replaces the parent best-model file. Main860 unique checkpoints +5new =865. All scores in histories/logs are ordinary fold evaluation, not the pure-chain paired metric.

Verify and unpack:

```bash
sha256sum -c PARTS.sha256
cat experiments-late-20261001.tar.zst.part* | sha256sum
# Compare with ARCHIVE.sha256.
cat experiments-late-20261001.tar.zst.part* | zstd -d -c | tar -xf -
cd payload
sha256sum -c SHA256SUMS
```

Overlay the `payload/WunderFunder` files onto the restored main snapshot to obtain the captured late state. Review the fresh dirty-worktree patch relative to archived HEAD; do not blindly apply it on top of an already-applied older patch. Predictions are inventoried only; bulk cache remains on beleriand.

The pipeline finished before this supplement was copied: see `LATEST_RESULTS.md` and `verification.json`.
