# E524: independently implemented C1 submission candidates

The user requested uploadable ZIPs using improvements that have actually landed, and explicitly prohibited copying teammate implementation code. These candidates use our existing Rust streaming engine, our wrapper and our independently trained C1 epoch-1 checkpoints. Read-only teammate snapshots are methodological references and are not included in the packages.

The frozen E511 comparison evaluated 468 exposed fold-0 sequences, 141 connected components and 808,863 scored rows using the official metric. C1 seed3 scored 0.681618 versus C0 0.679952; seed4 scored 0.681647 versus 0.679971. Their fixed equal-weight pair scored 0.682176 versus 0.681153. These are local development improvements, not a forecast that the public score exceeds 0.6660 or Maxim's 0.6674. Conditional bootstrap intervals and exact artifacts are in [E511 report](../local/E511_C1V2_C0_EVALUATION_v1/REPORT.md).

The pair was packaged first. Single seed3 and seed4 variants are being prepared as faster alternatives. Pair inference executes two separate recurrent trunks, while the previous public-best package shares one trunk; therefore pair runtime must be measured before recommending upload. A completed ZIP build does not constitute delivery approval.

These checkpoints retain their fold-0 training membership. They are not newly trained all-data production models. Do not silently relabel their ancestry or add the observed local delta to a public score.

Delivery evidence must bind each final ZIP hash to its static ABI checks, offline Python3.11 CPU callback/reset/determinism tests, numerical parity, fixed membership and representative one-core timing. Only variants passing those checks belong in the final upload recommendation. Public upload remains the human gate.

See [initial frozen pair contract](C1_PAIR_PACKAGE_CONTRACT_V1.json) and [independent implementation decision](../../../../specs/adr/0012-independent-teammate-method-implementation.md). The delivery operator owns the ZIP tests and exact terminal receipts; the independent critic owns provenance and evidence interpretation.
