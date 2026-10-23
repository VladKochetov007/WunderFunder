# Root score-status driver V2 checks

Three focused source/metadata tests passed. They verify that the driver differs
from preserved V1 only in its four reader/review path and digest assignments,
that those assignments bind the exact reviewed V4 reader, and that actual score
receipt literals remain unbound. The inverse-AST proof script wrote
`ROOT_SCORE_STATUS_DRIVER_PROOF_V2.json` with the same source pins.

Ruff format and check passed for the new V2 driver, proof script, and fixture.
The driver was not imported or executed. No actual score receipts, B/SSH,
systemd, result, or payload were accessed. Only the new fixture ran; prior
reader fixtures and broader suites were not rerun.

# Local checks — score result capture V2

Three new synthetic tests passed. They accept the exact score `Id`, reject wrong or missing `Id`, verify the property is requested and the guard precedes any result capture, and bind the result root/export unit to the V3 exporter contract. The inverse proof confirms only the approved changes from reader V1: evaluation_v2 result root, exporter-v2 identity, scorer V3 pins, one `Id` query property, and one exact guard; remaining operational AST matches V1.

Ruff is scoped to the new V2 capture template, fixture, and inverse-proof script. No V1 fixture was rerun. No B/SSH/systemd query, actual receipt binding, result read, score execution, or payload access occurred. V1's independent rejection is preserved and this V2 candidate remains pending independent source-only review.
