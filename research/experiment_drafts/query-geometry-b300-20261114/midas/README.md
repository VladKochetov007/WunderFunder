# MIDAS preparation and evidence handoff

MIDAS was used for the preparation route, using the clean source pin
`dec62b91120bfb6aaf4b0d214f4c4e2ac88bc54d` at
`/Users/alfa/Documents/MIDAS_worktrees/training-evidence`.
[route-preview.json](route-preview.json) is the exact CLI output:
`READY_LOCAL_HANDOFF`, route `research`, `launchable_remotely: false`.
[route-invocation.json](route-invocation.json) records the actual argv, source,
workspace, time, exit code and output hashes; [route-stderr.txt](route-stderr.txt)
retains stderr. The CLI operation is `midas task`, a metadata-only preview.
No suggested command, model agent, provider call, GPU stage or evidence import
was executed. A recommended model profile in the preview is not execution proof.

## Export independently measured endpoints

The [exporter](export_endpoint.py) targets the same MIDAS pin and its
`midas.external-training-receipt.v1` validator/importer. After the receiver's
complete source-qualified fixed-E4 comparison, invoke on the receiver:

```bash
export WF_MIDAS=/absolute/path/to/pinned/midas-checkout
test "$(git -C "$WF_MIDAS" rev-parse HEAD)" = dec62b91120bfb6aaf4b0d214f4c4e2ac88bc54d
test -z "$(git -C "$WF_MIDAS" status --porcelain -- midas)"
CELL=angular_s209401
PYTHONPATH="$WF_MIDAS" timeout 600 "$WF_PYTHON" "$PACKET/midas/export_endpoint.py" \
  --run-root "$WF_RUNS" --split-manifest "$WF_SPLIT" --cell "$CELL"
```

The new `$WF_RUNS/$CELL/midas-evidence` directory contains original metadata,
config/split bytes and a validated completion receipt. It binds the cell to all
six independent E4 grades in `comparison.json`. The receipt's source commit is
the experiment packet commit; the effective isolated native runtime has its own
source SHA256. Checkpoint bytes remain external hash references. No dataset rows
or weights are copied. Negative completed endpoints use the identical path.

## Import on the designated MIDAS host

```bash
BUNDLE="$WF_RUNS/$CELL/midas-evidence"
PYTHONPATH="$WF_MIDAS" timeout 600 "$WF_PYTHON" -m midas training-evidence import \
  --receipt "$BUNDLE/receipt.json" --evidence-root "$BUNDLE" \
  --workspace "$WF_REPO" --home "$WF_MIDAS_HOME" --json
```

Use a persistent receiver-owned MIDAS home. `outer_validation` in the importer
schema does not make the repeatedly exposed official valid a new holdout.
Partial/failed cells remain in native stage journals; this adapter cannot present
them as completed E4 outcomes. The commands above are prepared and unexecuted.
