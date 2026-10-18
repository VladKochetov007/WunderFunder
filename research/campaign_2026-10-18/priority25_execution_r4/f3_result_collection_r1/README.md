# F3 seed-1001 result collection R1

Status: `PREPARED_NOT_ADMITTED`. This isolated collector validates and captures the already-produced F3 fold-3, seed-1001 wide/deep exports, official paired grade, original v1 attempt records, and independently reviewed stage-1/terminal checkpoint receipts. It does not train, export, grade, open checkpoint tensors, or deserialize prediction payloads.

The validator pins the exact unadmitted F3 plan (`b303146c…f4a2d8`), fold manifest (`21c7fb52…ada1f`), F3 split metadata receipt (`5facd5e5…fc99a9`), and checkpoint-support manifest (`7e4fce0d…fa38dc`). It requires the four successful F3 checkpoint collection/verifier pairs, exact v1 commands/contracts and canonical source-map digests, the F3 ordered train/valid/evaluation identities, 29,760 stage-1 and 89,280 terminal optimizer updates, zero AMP skips/EMA updates, and stage-1-to-terminal training-state ancestry. It then requires a sealed arrival inventory and one `priority25.f3_terminal_result_review.v1` PASS file for each arm, binding exact receipt paths, byte counts, hashes, snapshot paths, and support hashes.

The collector validates both fixed epoch-5 exports as `[505, 20000, 2]` little-endian float32 NPY files, streams SHA-256 and finiteness checks without loading arrays, validates the sidecars and existing official `priority25.endpoint_grade.v1` paired grade, and publishes a create-only snapshot. The snapshot includes the original plan/fold/split metadata, checkpoint-support manifest and code, raw attempts/logs, collection and verification receipts, reviewer receipts, prediction files and sidecars, official grade, and the exact collector source. Archive cap: 256 MiB. Snapshot and receipt paths must be new and project-local.

Before any run, root must independently review this exact packet, stage its allowlist, refresh input ownership/resources, and create a separate root admission record. The result applies to one exposed fold/seed pair only. It does not establish a width winner, a clean holdout, CP30, convergence, serving performance, or transfer.

## Arrival seal

After all four checkpoint collection/verifier receipts exist and both arms have independent PASS reviews, write the seal JSON at the path passed as `--terminal-seal`:

```json
{
  "schema": "priority25.f3_terminal_evidence_seal.v1",
  "status": "sealed",
  "review_status": "passed",
  "plan_sha256": "b303146cf9db05e0a3bddf8f245d3c9b880d055ce7e97ebe5dd13510b5f4a2d8",
  "fold3_manifest_sha256": "21c7fb52e8e69f38126362cbc5dc60dbc46539041d649b9206ec35e0f21ada1f",
  "checkpoint_support_manifest_sha256": "7e4fce0d7232a487a769e12d611050bc608f9cc7fcea300bc2063525abfa38dc",
  "collector_support_sha256": {
    "result_io.py": "<actual sha256>",
    "f3_result_checks.py": "<actual sha256>",
    "collect_f3_results.py": "<actual sha256>"
  },
  "arms": {
    "wide": {
      "receipts": {
        "stage1": {
          "collection": {"path": "<exact path>", "bytes": 0, "sha256": "<actual sha256>"},
          "verification": {"path": "<exact path>", "bytes": 0, "sha256": "<actual sha256>"}
        },
        "terminal": {
          "collection": {"path": "<exact path>", "bytes": 0, "sha256": "<actual sha256>"},
          "verification": {"path": "<exact path>", "bytes": 0, "sha256": "<actual sha256>"}
        }
      },
      "snapshot_paths": {"stage1": "<exact plan path>", "terminal": "<exact plan path>"},
      "review_receipt": {"path": "<actual PASS review>", "bytes": 0, "sha256": "<actual sha256>", "status": "passed"}
    },
    "deep": {
      "receipts": {
        "stage1": {
          "collection": {"path": "<exact path>", "bytes": 0, "sha256": "<actual sha256>"},
          "verification": {"path": "<exact path>", "bytes": 0, "sha256": "<actual sha256>"}
        },
        "terminal": {
          "collection": {"path": "<exact path>", "bytes": 0, "sha256": "<actual sha256>"},
          "verification": {"path": "<exact path>", "bytes": 0, "sha256": "<actual sha256>"}
        }
      },
      "snapshot_paths": {"stage1": "<exact plan path>", "terminal": "<exact plan path>"},
      "review_receipt": {"path": "<actual PASS review>", "bytes": 0, "sha256": "<actual sha256>", "status": "passed"}
    }
  }
}
```

Replace every placeholder with observed values at arrival; this packet contains no predicted receipt hashes. Each reviewer file must use schema `priority25.f3_terminal_result_review.v1`, status `passed`, its arm, the four exact receipt refs for that arm, both plan-declared snapshot paths, and the same plan/fold/checkpoint-manifest/collector-support SHA bindings as the seal. The collector rehashes the seal, reviewer files, and every captured source before create-only publication.

## Run after separate root admission

Run from the campaign checkout using the bounded, reviewed environment:

```sh
python research/campaign_2026-10-18/priority25_execution_r4/f3_result_collection_r1/collect_f3_results.py \
  --project-root /home/grinchenkoeg/computer-vision-homework-20261018 \
  --plan /home/grinchenkoeg/computer-vision-homework-20261018/research/campaign_2026-10-18/priority25_execution_r4/f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json \
  --terminal-seal /home/grinchenkoeg/computer-vision-homework-20261018/receipts/F3_TERMINAL_EVIDENCE_SEAL_V1.json \
  --snapshot-dir /home/grinchenkoeg/computer-vision-homework-20261018/research/campaign_2026-10-18/priority25_execution_r4/f3_result_collection_r1/records/F3_S1001_RESULT_R1 \
  --receipt /home/grinchenkoeg/computer-vision-homework-20261018/receipts/RANK01_F3_S1001_RESULT_COLLECTION_R1.json
```

## Offline checks

```sh
python -m unittest -v test_f3_result_collection.py
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
```

Fixtures cover original-v1 attempt and source-map validation, F3 export sidecar identity, the paired grade, terminal-seal/reviewer binding, and a producer-shaped checkpoint collection/verifier receipt with successful and malformed counter cases. They use tiny local files and do not read model/data payloads or contact Beleriand.
