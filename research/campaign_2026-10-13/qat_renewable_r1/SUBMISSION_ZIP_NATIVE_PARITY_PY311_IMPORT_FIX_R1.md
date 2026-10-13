# ZIP parity checker Python 3.11 import correction

The first clean Python 3.11 CLI preflight for `check_submission_zip_native_parity.py --help` failed before argument parsing with `ModuleNotFoundError: No module named 'wnn'` at the checker’s `wnn.splits` import. The project’s regular Python 3.13 editable environment had hidden this test-harness dependency.

The checker now adds the repository `src/` directory explicitly to `sys.path` before importing `wnn.splits`. This is harness-only: the extracted submission callback remains NumPy-only and receives no repository source path. The clean Python 3.11 help command passes with `PYTHONPATH` and `LD_LIBRARY_PATH` unset, and the eight focused checker tests pass. Ruff is clean. No native library or ZIP callback was imported or evaluated; actual archive parity remains root-owned.

The earlier extracted-sibling-helper correction remains documented in [the import-path correction receipt](SUBMISSION_ZIP_NATIVE_PARITY_IMPORT_FIX_R1.md); this note records the distinct repository-split import failure found only in the clean Python 3.11 environment.
