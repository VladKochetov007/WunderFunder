# V4 local checks

Focused verification for this packet only:

- `./.venv/bin/ruff format` on the V4 template, renderer, and AST fixture: all
  three files were left unchanged on the final format pass.
- `./.venv/bin/ruff check` on those same three files: passed.
- `./.venv/bin/python test_child_allthree_terminal_verifier_v4_contract.py`:
  five focused static AST fixtures passed. The fixture reads and parses wrapper
  source but does not import or execute it.
- `./.venv/bin/python render_child_allthree_terminal_verifier_v4.py`:
  create-only render passed the checker V2 root preparation, exact source
  review, seven-file import proof, 14-receipt ancestry map, and typed/raw native
  evidence gates. It wrote the final body and its render-binding receipt.

The final wrapper body is 710,826 bytes with SHA-256
`c186e7b4473bc11681f61c55a741f17b2fdab828c7906090ef10574a8aa2df90`. The
render receipt records checker V2 at 40,442 bytes / SHA-256
`6ff4ba93a4020a4b34ade46a2512ac77b871328e95c47f2ca4561b8dcbd435b5`, the
root preparation at 30,221 bytes / `60d864c978699b912a4cab1d09989e9e536e5506960ec89ef985103b6739accc`,
the source review at 8,298 bytes / `ed093f41ee2b8073ed25c9b065a6fd7370dffed962b76a20f60e152e8837863a`,
and the import proof at 12,712 bytes / `d590fccb9c001791cdd3656a423e424a8d0042cf3a33e2415c381b5a1d306a11`.
It also binds the unchanged actual native typed/raw evidence pair (717 / 82,591
bytes; `da91b782...` / `0d9ba24a...`).

The renderer reads only bounded source/metadata and the exact raw-first native
capture. It does not read model, optimizer-state, dataset, split, dictionary,
or cache-array payloads. The rendered body was not imported, executed, staged,
or submitted. No B action, checker CLI, source/model import, fit, or quality
evaluation was run. Independent exact-body review and the separately reviewed
V2 result-capture consumer remain prerequisites for any later support action.
