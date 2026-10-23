# First-cell F0 export admission V4

V4 is a local source repair for the V3 preflight failure. It changes only check_fit_receipts in the rendered stdin body. V3 expected invocation_id and properties at the top level of the original AllThree LAUNCH.json; the frozen receipt has neither. V4 strictly decodes the saved systemctl_after_submission stdout, parses the six unique captured property lines, and joins InvocationID, unit, run name, MainPID, active/running state, and ControlGroup to the pinned native launch.

The exact V3 body, source review, root transport, stderr, and read-only preflight capture are preserved and pinned by the V4 renderer and binding. V3 failed before evidence-directory mutation or systemd-run; its transport records one attempt and no retry. The V4 body is 246,596 bytes, SHA-256 abe3251d9d65200b8e0c745db287e76f77e2dcb2a6524f3de7cd2f31c0cfa9ec. Its source binding and inverse proof bind the fixed five payloads, exporter review, V3 failure evidence, and exact inversion back to V3 when only the check_fit_receipts function is restored.

The new focused fixture runs only the AST-extracted check_fit_receipts function. It uses exact local small receipt bytes (the five accepted/stopped arms plus original AllThree ADMISSION.json and LAUNCH.json), fake POSIX paths and stat-only checkpoint metadata, and no project/model imports or checkpoint/data/prediction payload access. It includes positive coverage for the frozen receipt and negative cases for a mismatched nested invocation ID and misleading top-level launch fields.

V4 is still a local candidate awaiting an independent exact-body source review and a fresh root admission. No V4 staging, B action, export, score, or quality claim occurred.
