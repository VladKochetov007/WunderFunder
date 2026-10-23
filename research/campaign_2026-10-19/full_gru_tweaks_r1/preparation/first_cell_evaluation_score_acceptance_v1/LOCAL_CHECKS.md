# Local checks — score-result acceptance V1

The new standard-library fixture passed seven tests. It checks a valid
synthetic report joined to the exact local approved export acceptance/stop,
binding, and captured export manifest; verifies create-only receipt writing;
and rejects a mismatched native invocation, changed result base64, boolean
score values, missing contrasts, duplicate JSON keys, and non-finite JSON
numbers. It creates no real acceptance receipt and reads no score, prediction,
target, model, checkpoint, cache, or source payload.

Ruff `check` and `format --check` pass for only `score_result_acceptance.py`
and `test_score_result_acceptance.py`. The test imports only this new helper
and Python standard-library modules. No existing or scorer tests were run.

Frozen new-source pins: `score_result_acceptance.py` is 45,795 bytes, SHA-256
`5987d527382f0ce33e72b4c17ae020d03485df3057b398ae285940ead29d528b`;
`test_score_result_acceptance.py` is 13,145 bytes, SHA-256
`de11d08bfb0b986bf11b3d68515910121254f105fffba05fab0e9f903713442f`.

Frozen validation inputs are the score V5 CLI (`score_first_cell.py`,
10,630 bytes, SHA-256
`b2546281deb54f944577d4db19a8bd7d903cc0f5b755bdb21b2b962696f56dd6`), score
core (`first_cell_score_core.py`, 27,226 bytes, SHA-256
`30c456e7db875e2c075a6200c857c375087b18d9c952c354ae9bd3dceb0a0143`), and
bounded score-capture V4 template (26,354 bytes, SHA-256
`88968f6dd73d4b489c4fc57308906c5e390137a49389ab991408408f353b9be6`) with
its accepted source review (5,655 bytes, SHA-256
`b05b06a6e56888348e30ebe0ded0272a5e9f3ca709037feb975b938b4d0bb7d0`). The
helper additionally pins the accepted V4 export binding, ROOT export
acceptance, exact stop, source review, and captured manifest in code. Those
pins are metadata authority only; they do not constitute a score-result
acceptance or authorize any actual score execution.

The captured score response and original score receipts remain inputs for a
future ROOT call. This packet contains no actual score `ADMISSION.json`,
`LAUNCH.json`, or `RESULT.json`, and no acceptance was written. ROOT owns any
actual validation call, acceptance receipt, exact stop, and later publication.
