# Local checks

Focused verification was run with the literal maintenance `.venv` interpreter
and installed Ruff. No unchanged suites were run.

| Check | Result |
|---|---|
| `python -B -m unittest -v test_score_result_acceptance_v2_registered_parent.py` | 4 focused metadata tests passed |
| `ruff check score_result_acceptance_v2.py test_score_result_acceptance_v2_registered_parent.py prove_score_acceptance_v2_delta.py` | passed |
| `ruff format --check score_result_acceptance_v2.py test_score_result_acceptance_v2_registered_parent.py prove_score_acceptance_v2_delta.py` | passed |
| `python -B prove_score_acceptance_v2_delta.py` | `PASS_NARROW_ALIAS_AND_SCORE_V2_REBASE` |

The fixture reads and pins only these metadata inputs:

| Input | Bytes | SHA-256 |
|---|---:|---|
| `preparation/first_cell_evaluation_export_v4/EXPORT_BINDING_V4.json` | 15,796 | `4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd` |
| `records/first_cell_evaluation_v1/score_v2/ADMISSION.json` | 69,650 | `1252b2f1d39ef255fac83c722cfc1a380705963a0b2f9dede35dcabed45ba631` |
| `records/first_cell_evaluation_v1/score_v2/alias_parent_diagnostic_v1/STDOUT.raw.json` | 3,218 | `e6c08c71185d34c8763ec4fc983d72a91003d3626e6dc2da6a53926c49cb8380` |

The diagnostic shows the accepted cache parent resolves to itself and the
accepted datasets parent resolves to `/home/grinchenkoeg/WunderFunder/datasets`.
All six leaf identity records match the V4 binding and preserved admission.
Payload reads/hashes, scorer/helper execution, B access, and unit queries were
not performed by the source-delta proof.
