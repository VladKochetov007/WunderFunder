# H382 fold-3 same-science successor R4

The parent R3 sequence stopped at index 2 because its fit attempt remained
`not_started` after aggregate-RAM admission refused 10.5GB against the live
18GB working cap. The original refusal and sequence receipt remain unchanged.
F2 e4 fit and export completed; this packet resumes only fold-3 e3 and e4 with
new attempt/contract IDs. The run names, recipes, frozen source/data, folds,
checkpoint epochs, export commands and per-job limits are byte-bound to their
parent contracts.

R4 is split into two ordinary durable `bandit_sequence.py` invocations so the
existing bounded file-cache handoff can run after e3 export and before e4 fit.
No sequence runner or controller code is changed. A failure before a worker
starts remains infrastructure/admission evidence, not a negative model result.
