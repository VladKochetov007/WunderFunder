# MIDAS handoff

[The saved research route](route-preview.json) comes from the MIDAS CLI at `dec62b91120bfb6aaf4b0d214f4c4e2ac88bc54d`. The preparation returned `READY_LOCAL_HANDOFF` with `launchable_remotely: false`. The B300 stages in the parent packet are the receiver's explicit actions. MIDAS does not schedule them.

Once [the paired readout](../readout.py) has accepted the four canonical E4 grades, package each grade with its original metadata, split and terminal receipt. Use a provisioned checkout at the MIDAS pin. The exporter invokes MIDAS's own receipt validator before making the new bundle visible. It binds the cell and hashes to the paired comparison. Checkpoints are external hash references; no weights or parquet files are transferred.

```bash
export WF_MIDAS=/path/to/pinned/midas-checkout
test "$(git -C "$WF_MIDAS" rev-parse HEAD)" = dec62b91120bfb6aaf4b0d214f4c4e2ac88bc54d
test -z "$(git -C "$WF_MIDAS" status --porcelain -- midas)"
CELL=heads6_s207301
PYTHONPATH="$WF_MIDAS" timeout 600 "$WF_PYTHON" "$PACKET/midas/export_endpoint.py" \
  --run-root "$WF_RUNS" --split-manifest "$WF_SPLIT" --cell "$CELL"
```

Copy the cell's `midas-evidence` directory to the designated evidence store. On the MIDAS host run the following for that directory, substituting the project's provisioned workspace and persistent MIDAS home.

```bash
PYTHONPATH="$WF_MIDAS" timeout 600 "$WF_PYTHON" -m midas training-evidence import \
  --receipt "$BUNDLE/receipt.json" --evidence-root "$BUNDLE" \
  --workspace "$WF_REPO" --home "$WF_MIDAS_HOME" --json
```

The MIDAS schema calls the endpoint scope `outer_validation`. In this competition the partition has repeatedly been used for development. Failed and partial attempts remain in their receiver stage logs and model journals and have no complete-panel score. A negative completed endpoint is exported on the same terms as a positive one.
