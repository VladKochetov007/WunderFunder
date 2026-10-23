# V3 local checks

The V3 stdin body is 243,576 bytes, SHA-256 `d7e2ca4852da54d8d1554ff3271dd3d4f369575a6c26c7dc19990a2ca54cb509`. Its local source binding is 13,771 bytes, SHA-256 `6d1d599d688590f492a07b129ccea0080d38847130e0df420d3e562dd4a17d30`. The renderer verifies all five payloads by exact length and SHA, verifies the V2 exporter source review against the exporter, binding, and unchanged manifest payloads, and binds the preserved V2 failed-preflight body/transport/stderr metadata.

Checks run from this packet directory with the literal maintenance virtual-environment executables:

- `/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/ruff format --check` on the V3 template, renderer, rendered body, inverse proof, and focused fixture: all five formatted.
- `/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/ruff check` on the renderer, rendered body, inverse proof, and focused fixture: passed.
- `/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python -B prove_admission_v3_inverse.py`: passed; the 3,638-byte proof receipt has SHA-256 `07afd545eb5658db0ea1454ff2632b3096095631a172669cc7d1de1b7d033977`.
- `/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python -B -m unittest test_admission_v3_payload_contract -v`: three new focused source/metadata checks passed. They validate exactly five embedded payloads, the V2 producer-review joins, V2 body inverse equality outside four payload assignments, and the preserved failed-preflight identity. No old fixture or model/data code was imported or replayed.

No B staging, admission, SSH, or execution occurred during this local rendering. The V2 preflight remains the only failed attempt; V3 has not been attempted. Root must repeat fresh resource, unit-absence, evidence/output-absence, GPU/process-ownership, and source/data gates before any later attempt. No prediction export, scoring, or quality result was produced.
