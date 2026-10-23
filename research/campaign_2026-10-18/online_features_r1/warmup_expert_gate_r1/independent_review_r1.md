# Independent static review: warmup expert gate

**Verdict: pass for the corrected screen implementation.** Review was static and metadata-only; the reviewer did not run scripts or open numeric target, mask, or prediction payloads.

The parent sidecar declares all 1,873 validation row groups. The other three pinned artifacts declare exactly the 468 F0 evaluation groups in split order. The corrected grader accepts only those two exact orders, and `scripts/grade_prediction_bank.py::aligned` maps either declaration to the frozen requested F0 group order. Prediction and sidecar hashes, checkpoint identities, and split hashes remain bound to `recipe.json`.

Fit and ancestry checks and the preflight receipt precede decoding or use of target/mask numeric values. The grader hashes the target archive and reads its row-group index before preflight; it does not decode or use the target/mask numeric arrays before the receipt is written.

This review corrects the same reviewer's earlier static pass, which had mistakenly described all sidecars as full-validation arrays. The corrected finding was established by comparing each pinned sidecar's exact declared group list to the source-qualified split.
