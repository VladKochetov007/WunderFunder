# First-cell streaming scorer V4

V4 is a source-only consumer adaptation for the frozen exporter V4 contract. It changes the binding-size and SHA-256 pins, requires `EXPORT_BINDING_V4.json`, and binds the fixed `evaluation_v3` evidence root. The V4 exporter, binding, and unchanged source manifest are recorded in [`dependency_manifest.json`](dependency_manifest.json).

The four scoring modules preserve V3's metric, streamed group processing, paired bootstrap, and bounded prediction I/O. The focused metadata fixture confirmed that the only scorer-code changes are the binding constants, the runtime binding filename, and the evidence-root basename. It also verified the exact V4 exporter and binding pins, their source manifest, and the binding's four evaluation-root path substitutions plus the pinned empty `expected_checkpoint.prop_input` value. No scorer, exporter, WNN source, dataset, cache, checkpoint, prediction, or target payload was imported or read.

The V4 producer/scorer source-only review is pending in the dependency manifest. This packet is not an admission or execution authorization; it does not render a score body, access a remote host, or make a metric or quality claim.
