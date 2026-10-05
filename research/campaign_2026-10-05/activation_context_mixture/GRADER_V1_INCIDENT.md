# E484 independent grader v1 incident

The first actual `--verify-fit-only` attempt failed closed before scoring or
loading test-role data. The fit manifest records the registered contrast list
as exact labels such as `recurrent_mix - static_mix`; grader v1 incorrectly
expected underscore identifiers. The receipt is preserved in
`research/campaign_2026-10-05/runner/logs/ROOT_FIT_ONLY_GATE_V1.log`.

The manifest's final receipt string is `gelu - fixed_mix`, written in
reference-minus-candidate order. The registered display label is
`fixed_mix - gelu`; grader v2 deliberately maps that exact manifest string to
the preregistered candidate-minus-reference contrast rather than following the
string's reverse sign. This is a label convention in the actor's inventory,
not a change to the preregistered contrast direction.

Grader v1 remains unchanged at SHA-256
`dc312c08b90e0828a2057ca4afba8ae97ab9d49760aee867fb6cb47c67bf62fb`.
Version 2 checks the full exact actor-registered eight-label list and count,
adds an actual attempt/artifact file-path contract test, and independently
recomputes the normalizer from the 24 raw fit-role feature groups. Its gated
actual-fit test exercised probe/fit source, status, runtime, OOM, artifact,
normalization, telemetry, and checkpoint validation; it did not invoke the
test-inclusive loader or compute scores. All seven v2 tests passed under
one CPU and a 2-GiB memory cap. No held-out labels were read by this gate.
