# H232 R6 parent-metadata transport repair

R5's Kaggle log ended before export: the unchanged worker's `_verify_payload`
rejected `parent_packet_meta.json`. The asset ZIP contained 17 entries and had
the exact R5 content-addressed SHA. Its `packet_meta.json` still declared the
parent metadata as a required file with SHA-256
`97e01d0c09a4e7dd67c13da6757e6f5bc72e2906c4232ec52415af8fdbc95e20`, but the
asset builder carried `packet_meta.json` in that slot and omitted the required
4,204-byte parent file. The original parent metadata remains available and
unchanged in R5's `basis/` directory.

R6 creates a new content-addressed archive with all 17 R5 ZIP entries preserved
byte-for-byte and appends the exact parent metadata as the eighteenth entry. It
does not edit `packet_meta.json`, `export_worker.py`, the selected checkpoint,
the source archive, or the scientific recipe. A new private dataset slug and
canonical kernel ID identify this transport successor. The builder only stages
local upload metadata and kernel source; it makes no provider calls and grants
no kernel-push authority.

The focused test extracts the complete R6 archive through the rendered bootstrap,
checks that every embedded required path/hash is present, then calls the
unchanged R5 `export_worker._verify_payload` on that extracted tree. It does
not load Torch, initialize CUDA, open Parquet or NPZ files, fit, export
predictions, or score.

Build locally with:

```sh
python research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_v7b_ep3_export_only_20261008/asset_transport_r6/build_r6_packet.py
```

Run the offline checks with:

```sh
PYTHONPATH=src:. python -m pytest -q research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_v7b_ep3_export_only_20261008/asset_transport_r6/test_r6_packet.py
```
