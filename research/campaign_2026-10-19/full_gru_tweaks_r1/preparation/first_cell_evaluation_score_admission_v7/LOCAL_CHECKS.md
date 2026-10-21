# V7 local checks

The focused metadata fixture passed 6 tests using the literal repository `.venv/bin/python -B` entrypoint. Coverage is limited to a tiny exact source inventory, unlisted `.py`/`.so`/`.pyc` files, file and directory symlinks, inventory-before-hash source order, the preserved V6 inventory-gap note, and AST equality outside the two source-inventory functions. The separate inverse proof passed after confirming the frozen V6 body pin.

The installed repository Ruff executable passed `check` and `format --check` for the four new Python files. The conditional renderer was not called; there is no bound export acceptance or exact-stop, no rendered score submission, and no B query or payload access.

Frozen V7 source and local evidence pins:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `SCORE_ADMISSION_BINDING_PLAN_V7.json` | 6,335 | `69a758e000b6fc7fc488ae7da676a63e116b98a3af1a309f9a23e0d80c800a50` |
| `SCORE_ADMISSION_BODY_TEMPLATE_V7.py` | 73,741 | `4ef8124679ae69ea53e4ad8e3ec0251610d5c4e86f16c299710b6f823b98acc1` |
| `render_score_admission_v7.py` | 18,020 | `558957c7b9039c13e6d9a75caa7e5ac41921d9941c5c1c2726423b1fb6239747` |
| `test_score_admission_v7_inventory.py` | 6,392 | `2fe8d5fed460dcd479ea5c7174b3f2f5823219db20a0cd346769dad2a4b30566` |
| `prove_score_admission_v7_inverse.py` | 2,370 | `2ea4d7bfa24124d24361cefd72e77a5ebafe6173b4033e55416a2aa4ac649306` |
| `V7_INVERSE_PROOF.json` | 821 | `c34dd5cde265f7afcb871f00f217bc4341603c0bfc1d9a6b10bb8cb996e21339` |
| `V6_SOURCE_INVENTORY_GAP_NOTE_V1.json` | 1,107 | `f09f07d911c8f313b6c4e00667c1f34bdbb7e0f0a2a73cc92735a42438a5e436` |

The separate V3 exporter/scorer source-only pair review remains pinned in the V7 plan. V7 itself has no independent source review yet and is not admitted or executable.
