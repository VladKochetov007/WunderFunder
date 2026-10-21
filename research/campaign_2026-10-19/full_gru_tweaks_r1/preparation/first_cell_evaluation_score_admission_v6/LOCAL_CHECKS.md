# V6 local checks

The focused metadata fixture passed: 5 tests, including the exact pinned V3 review, the renderer's review path, duplicate/missing/tampered review rows, failed join claims, still-null export receipt pins, and the AST inverse comparison against V5.

The installed repository Ruff executable passed `check` and `format --check` for the V6 renderer, fixture, and body template. The fixture was run with the literal repository `.venv/bin/python -B` entrypoint. No old score-admission fixtures or suites were replayed.

Frozen-at-check byte pins:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `SCORE_ADMISSION_BINDING_PLAN_V6.json` | 6,077 | `337eb86830e0de924fb55fb4917b8100eda6e712122796580a0dff5f89a6f5d5` |
| `SCORE_ADMISSION_BODY_TEMPLATE_V6.py` | 70,643 | `aecec19975a498174a859228499b2e12a5a24965b0ead94f0b8f78941fdf4e84` |
| `render_score_admission_v6.py` | 18,020 | `312c2ba0190e4abcd7ef8a04c30a0bf2ec212d57c62b9e1701ccb4547f227a61` |
| `test_score_admission_v6_review_metadata.py` | 6,736 | `6744f48ff6f5a5ad0fabedc20fc7b807ef23093fd9d7af09abda307c9302be84` |

These pins precede independent review of the V6 packet. The separate exporter/scorer V3 source-only review is pinned at 7,284 bytes, SHA-256 `ee53ce06a6119152866e8f39801e7575e59fb42bc09515d102db4c8df5d9c37a`, status `PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR`.
