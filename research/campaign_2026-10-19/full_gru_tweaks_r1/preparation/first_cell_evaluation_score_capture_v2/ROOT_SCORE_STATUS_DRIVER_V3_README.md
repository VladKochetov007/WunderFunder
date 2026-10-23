# ROOT score status driver V3

V3 is a conditional source rebase of the V2 ROOT status driver to the
independently reviewed score-capture reader V5. It changes only the four
reader/review references (`TEMPLATE`, `TEMPLATE_SHA`, `REVIEW`, and
`REVIEW_SHA`). The driver still reads the five eventual admission/launch
bindings from the supplied record and remains unbound until those exact
receipts exist.

The V5 reader review is
[`INDEPENDENT_SOURCE_REVIEW_V5.json`](../first_cell_evaluation_score_capture_v5/INDEPENDENT_SOURCE_REVIEW_V5.json),
schema `fg19.first-cell-score-result-capture-source-review.v5`, status
`PASS_SOURCE_ONLY_READER_V5_HARD_BOUNDED_UNBOUND`. The source-delta proof is
[`ROOT_SCORE_STATUS_DRIVER_PROOF_V3.json`](ROOT_SCORE_STATUS_DRIVER_PROOF_V3.json).
This driver has not been imported or run; no score unit was queried and no
receipt, result, or payload was read.

The V3 status driver is a future score-v2 reader handoff. It does not authorize
the capture call; ROOT must bind the exact future admission/launch bytes and
invocation after durable admission.
