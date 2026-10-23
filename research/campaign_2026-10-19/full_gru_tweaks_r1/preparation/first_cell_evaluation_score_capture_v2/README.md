# Current root score-status driver V2

The current root-side one-shot score status/result capture driver is
[ROOT_CAPTURE_SCORE_STATUS_V2.py](ROOT_CAPTURE_SCORE_STATUS_V2.py). Its only
source delta from preserved V1 is the four reader/review path and digest
assignments: it pins the hard-bounded
[score-capture V4 template](../first_cell_evaluation_score_capture_v4/SCORE_RESULT_CAPTURE_TEMPLATE_V4.py)
and its exact independent
[V4 source review](../first_cell_evaluation_score_capture_v4/INDEPENDENT_SOURCE_REVIEW_V4.json)
(`PASS_SOURCE_ONLY_READER_V4_HARD_BOUNDED_UNBOUND`). The local inverse-AST
record is [ROOT_SCORE_STATUS_DRIVER_PROOF_V2.json](ROOT_SCORE_STATUS_DRIVER_PROOF_V2.json),
reproduced by [prove_root_score_status_driver_v2.py](prove_root_score_status_driver_v2.py),
and its new source-only metadata fixture is
[test_root_score_status_driver_v2_rebase.py](test_root_score_status_driver_v2_rebase.py).

The five actual score ADMISSION/LAUNCH/invocation literals remain unbound. The
driver and fixture were not run against actual receipts or a unit; the driver
was not imported or executed. This local pair supplies no actual result or
quality evidence. The prior V2 reader and the text below remain preserved
history.

# First-cell bounded score result capture V2

This version preserves the V1 reader and corrects its missing live-unit identity check. The native systemd request includes `Id`, and the reader requires `Id == computer-vision-homework-fg19-first-cell-score-v1.service` immediately after the exact unit query, before it may capture `RESULT.json`. It is bound to the score V3 code pins, exporter V2 unit, and exporter V3 result path under the evaluation_v2 evidence root.

The actual score ADMISSION/LAUNCH byte counts, hashes, and invocation ID remain unbound sentinels. `_require_bound_pins()` fails before path, process, systemd, cgroup, GPU, or receipt access. The exact export receipt schemas and score admission version remain subject to root's forthcoming V6 plan. No actual result exists or is claimed by this local packet.

On a future exact score-unit invocation, the reader captures bounded pinned receipts, code pins, unit and campaign-slice metadata, selected-GPU/process metadata, and the bounded `unit.log` tail. It opens the evidence-root `RESULT.json` only for the exact invocation and a strict `success`, `active/exited`, PID 0, code 1, status 0 tuple, returning bounded bytes without parsing the score. Failure paths retain the log tail and never open the result. The prior V1 independent review rejected the missing `Id` guard; that review is preserved unchanged in the V1 folder. The V2 local source proof is in [SOURCE_PROOF_V2.json](SOURCE_PROOF_V2.json), with an inverse AST proof of the exact narrow delta.

This packet is a narrow local source correction only. It does not query B, execute the reader against a real unit, open any result, prediction, target, model, source, dataset, cache, or checkpoint payload, or make a metric/quality claim. Root owns actual capture, strict result extraction, scalar acceptance, exact stop, and any publication.
