# Reviews and audit receipts

This domain stores independent critiques, strategy reviews, and structured audit receipts. Reviews interpret or challenge evidence; they do not replace the source experiment, score receipt, or append-only ledger row they cite.

## Current structure

- [`receipts/`](receipts/) contains machine-readable, source-bound audit records, including the F1 recipe audit.
- [`legacy/`](legacy/) contains historical narrative reviews and critiques moved from the repository root and `writeups/`.
- Operational controller commentary is indexed under [operations](../operations/README.md), and imported external competition writeups are under [sources](../sources/README.md).

## Review files

- [WNN strategy review and agent execution brief](legacy/WNN_2026-10-11_Strategy_Review_and_Agent_Execution_Brief_EN.md)
- [Unified campaign review](legacy/WNN_Unified_Review_EN.md)
- [Independent critic memo](legacy/critic.md)
- [Literature and modeling review](legacy/review_2.txt)
- [F1 recipe audit receipt](receipts/F1_RECIPE_AUDIT_R4.json)

The previous `reviews/` and `writeups/` paths remain as symlinks for existing references. New reviews should be saved here and linked from the relevant domain README and source experiment record.
