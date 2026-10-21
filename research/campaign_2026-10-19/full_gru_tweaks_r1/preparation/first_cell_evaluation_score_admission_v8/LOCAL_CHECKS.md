# Local checks — score admission V8

The new source-focused fixture passed four tests. It validated the exact 8,712-byte V4 producer/consumer review and all 12 reviewed-file rows; rejected a wrong row pin, a missing review row, and a false join; verified the V4 binding, V5 scorer, export-v3 unit, and evaluation-v3 root; and compared V8 with V7 to confirm only identity assignments and the three binding-filename literals changed. It also checks that the source-inventory gate is byte-for-byte AST-identical and export acceptance/stop pins remain null.

Ruff `check` and `format --check` are run only on the new V8 renderer, body template, and fixture. The test imports the local stdlib-only renderer for its metadata parser. It does not call `render()`, execute either body, import WNN, query B/systemd, or access model, data, cache, prediction, or result payloads. No actual score admission or quality result exists here.
