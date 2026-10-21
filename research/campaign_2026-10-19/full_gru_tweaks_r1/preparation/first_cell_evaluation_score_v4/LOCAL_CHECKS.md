# Local checks — scorer V4

The new metadata-only fixture passed four tests before the dependency manifest was frozen. It checks the exact frozen exporter V4/binding/source-manifest pins, the V3-to-V4 binding delta (four evidence-root paths plus the empty `prop_input` default), the AST-limited score-code adaptation, and the runtime binding/evidence-root literals. The fixture parses code as text and does not import or run the scorer, exporter, WNN, or source tree.

Ruff `check` and `format --check` passed for all five Python files. V3 scorer files and tests remain unchanged; its suites were not run again. No B actions or model, target, cache, dataset, prediction, or checkpoint payload reads occurred. No scorer output or quality claim exists.

The dependency manifest pins the frozen exporter V4 source at 66,790 bytes / SHA-256 `87b6d3626bdafe6353865d95b1d8116bd187cb4da46888815658db6fc192738e`, its binding at 15,796 bytes / SHA-256 `4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd`, and the unchanged source manifest at 27,569 bytes / SHA-256 `a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621`. Independent V4 producer/scorer review is pending.
