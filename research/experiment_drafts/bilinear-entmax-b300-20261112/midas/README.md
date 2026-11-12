# MIDAS preparation and evidence handoff

`route-preview.json` is the exact output of the clean pinned MIDAS `task --kind
research --json` command at source commit
`dec62b91120bfb6aaf4b0d214f4c4e2ac88bc54d`. Its SHA256 is
`56a4556cda728dd6f57189fee45d4b458119e81d198defd7da740a85685d7088`.
MIDAS returned `READY_LOCAL_HANDOFF` and `launchable_remotely: false`. This records
the preparation route; no model agent, provider call, GPU job or evidence import
was started. Publication follows the separately authorized Git CLI workflow.

The packet reuses the PR81 metadata export design, specialized to the ten Signal
Routes cells. The exporter checks source/config/split/owner identities, terminal
state, retained epoch index and full-validation coverage. It includes available
producer logs and epoch metrics. It hashes checkpoint bytes without copying them;
MIDAS therefore receives `external_claim_only` checkpoint evidence, not verified
weights or an independently recomputed score. Failed and incomplete results can
be exported without a score; they cannot masquerade as completed E4 results.

After a real server fit, export its evidence on that server:

```bash
timeout 1800 "$WF_PYTHON" "$WF_PACKET/midas_evidence.py" \
  --fit "$WF_RUNS/$WF_ID/fit" --score "$WF_RUNS/$WF_ID/score.json" \
  --output "$WF_RUNS/$WF_ID/midas-evidence"
```

The metadata exporter does not load a model or run inference. It requires the
current packet source bytes to match those recorded by the fit. Do not regenerate
old producer receipts under a different source revision.

Copy the resulting small evidence directory to the machine running the pinned
MIDAS checkout, then use its existing v1 importer:

```bash
MIDAS_SOURCE=/path/to/provisioned/midas-checkout
BUNDLE=/path/to/copied/midas-evidence
PYTHONPATH="$MIDAS_SOURCE" timeout 600 "$WF_PYTHON" -m midas training-evidence import \
  --receipt "$BUNDLE/receipt.json" --evidence-root "$BUNDLE" \
  --workspace "$WF_REPO" --home "$WF_REPO/.midas/bilinear-entmax" --json
```

No import is claimed in this PR. MIDAS evidence admission checks the producer
metadata; it does not prove which host ran a model or establish generalization.
`campaign.stage=B` is this packet's initial-screen catalog mapping, not R4-B.

For an incomplete or failed run with producer metadata, omit `--score`; preserve
its terminal/log evidence without inventing an E4 score. Failures before metadata
creation remain in the launcher receipts and must be returned as raw failed-stage
evidence. No experiment is repeated automatically by the exporter or MIDAS.

Use MIDAS source `dec62b91120bfb6aaf4b0d214f4c4e2ac88bc54d` for the import contract.
The route preview contains a preparation recommendation, not a GPU command to run.
Do not execute its suggested research-agent command as part of model acceptance.
