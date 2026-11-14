# MIDAS endpoint handoff

The adapter targets MIDAS `dec62b91120bfb6aaf4b0d214f4c4e2ac88bc54d` and its
`midas.external-training-receipt.v1` validator/importer. It reuses the PR97
metadata-only handoff, adapted to ten perturbation cells. No route preview,
model-agent launch, result import or GPU scheduling was executed for this packet.

After the complete source-qualified fixed-E4 readout, on the receiver:

```bash
export WF_MIDAS=/absolute/path/to/pinned/midas-checkout
test "$(git -C "$WF_MIDAS" rev-parse HEAD)" = dec62b91120bfb6aaf4b0d214f4c4e2ac88bc54d
test -z "$(git -C "$WF_MIDAS" status --porcelain -- midas)"
CELL=stale_burst_s208401
PYTHONPATH="$WF_MIDAS" timeout 600 "$WF_PYTHON" "$PACKET/midas/export_endpoint.py" \
  --run-root "$WF_RUNS" --split-manifest "$WF_SPLIT" --cell "$CELL"
```

The new `$WF_RUNS/$CELL/midas-evidence` bundle includes source/config/split,
terminal and grade metadata. Source commit is the **experiment packet commit**;
the effective patched native source is bound by its separate SHA256. Checkpoint
bytes remain external and hash-referenced. No private rows or weights are copied.
The exporter verifies the complete comparison and calls MIDAS's schema validator
before publishing the directory. Negative completed endpoints follow the same path.

Import each completed cell into the designated MIDAS workspace:

```bash
BUNDLE="$WF_RUNS/$CELL/midas-evidence"
PYTHONPATH="$WF_MIDAS" timeout 600 "$WF_PYTHON" -m midas training-evidence import \
  --receipt "$BUNDLE/receipt.json" --evidence-root "$BUNDLE" \
  --workspace "$WF_REPO" --home "$WF_MIDAS_HOME" --json
```

Use a persistent, receiver-owned MIDAS home. The schema's `outer_validation` name
does not make official valid unexposed. Partial and failed runs stay in the native
journals and stage logs; they cannot be imported as completed E4 endpoints by this
adapter. These are prepared commands, not evidence of a successful remote import.
