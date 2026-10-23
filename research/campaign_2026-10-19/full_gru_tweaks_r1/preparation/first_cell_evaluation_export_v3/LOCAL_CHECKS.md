# V3 local checks

The V3 source changes address two pre-mutation failures: canonical API pins are compared against complete source-manifest rows, and the CONTROL acceptance parser matches its actual frozen representation metadata. Binding V3 changes only the four fixed private-evidence-root paths from V2; the 167-file source manifest is unchanged.

Checks run with the literal maintenance virtual-environment executables:

- `./.venv/bin/ruff format` on the V3 exporter and focused metadata fixture.
- `./.venv/bin/ruff check` on those same two files: passed.
- `./.venv/bin/python -B -m unittest discover -s research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v3 -p 'test_export_v3_source_manifest_api_pins.py' -v`: five focused metadata fixtures passed.

The fixture reads only the pinned binding, 167-entry source manifest, and ten exact small local accepted-fit/exact-stop receipts. It mocks source/checkpoint/split digest reads and uses temporary empty paths for the full gate calls. It imports only the stdlib-only V3 exporter module; it does not import canonical source modules, the restricted loader, PyTorch, or model code. No model, cache, target, parquet, prediction, or array payload is read. No B action or execution occurred.

The exporter is 66,080 bytes with SHA-256 `9bfcd5110d200212bff45d2188f04e2eb094fbaf5cd78831ed03db5a130af8cb`; the binding is 15,798 bytes with SHA-256 `372ec16016517baa510cc1899f140cd0978a35b83f96a4a5fa1ff4bda8b39730`; the source manifest is 27,569 bytes with SHA-256 `a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621`; and the focused fixture is 11,832 bytes with SHA-256 `759cd0f0a578c0fe1fa11ac5cf43f1205fd5cc0d21c20bc7d030ff2d164603fc`.

Independent source review is pending. This packet has not been staged, admitted, or executed on B.
