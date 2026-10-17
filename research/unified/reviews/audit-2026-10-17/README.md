# October 17 audit evidence

Small machine-readable receipts from the local/beleriand/GitHub audit. These describe existing artifacts and measurements; collecting them did not train or score a model. The [manifest](manifest.json) pins each file's SHA256, source and any transformation.

| File | What it establishes | Limits |
|---|---|---|
| [Data metadata](data_metadata.json) | Parquet footer counts, full scoring-mask counts, saved fold/chain/class metadata | No complete scan of feature pages; equal mask counts do not prove rowwise equality |
| [Train row group 528](rg528_audit.json) | Current full decode and finite-value check | Does not establish the cause of the historical failure or alter existing training membership |
| [Submission inventory](submission_inventory.json) | 11 server ZIP manifests/hashes and available local counterparts | Nine local hashes matched; two local dataless placeholders were not read |
| [Official leaderboard snapshot](official_leaderboard_20261017.json) | Browser-transcribed top 22, including kotiki .66832 | Dated public observation, not private submission history or a ZIP-to-score receipt |
| [Preserved public receipts](public_receipts.json) | 81 observations of 70 portal IDs with source references | User-transcribed repository records; no fresh private portal/hash verification |

Paths inside original metadata identify the machine where evidence was collected. They are provenance, not portable download URLs. Data, checkpoints, prediction arrays, ZIP payloads, credentials and large archives are not copied here. Source hashes in public receipts refer to their unchanged repository JSON at Main `0edd432271991250f4718f9b68f33340eff46214`.

Start with the [audit interpretation](../REPOSITORY_AUDIT_20261017.md), [model catalog](../../models/CURRENT_MODEL_CATALOG.md), [validation protocols](../../validation/VALIDATION_PROTOCOLS.md), or [storage map](../../operations/STORAGE_MAP.md).
