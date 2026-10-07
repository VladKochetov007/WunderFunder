# E524: independently implemented C1 submission candidates

## Public results reported by the user on 2026-10-07

| Candidate | Public score | Versus incumbent 0.6660 | Portal runtime / memory |
|---|---:|---:|---|
| C1 seed 3, `1n6uxwb6` | 0.6647 | −0.0013 | 33m11s / 624 MB |
| C1 seed 4, `1cymqpkk` | 0.6654 | −0.0006 | 33m07s / 624 MB |

Both uploads completed scoring and neither improved the public best. [Preserved result receipt](E524_PUBLIC_RESULTS_20261007_V1.json) binds the user-reported portal rows to the local filenames and hashes; the portal itself did not report ZIP hashes. These are now scored candidates, not an unscored upload queue. Do not reupload identical files. Their measured runtime and possible value in materially different combinations remain useful. Public single timings do not establish pair runtime or score. Own best stays **0.6660**; Maxim's reported **0.6674** is not reproduced, H230 subsequently fitted and publicly scored0.6649; see [its public receipt](../wide_baseline_heterogeneous_H230/H230_PUBLIC_RESULT_20261008_V1.json).

The original technical delivery and exposed local comparison below are preserved. The E511 gain was against matched C0 continuation, while this public comparison is against the incumbent served endpoint. Neither comparison isolates every recipe, training-membership and serving difference.

The user requested uploadable ZIPs using improvements that have actually landed, and explicitly prohibited copying teammate implementation code. These candidates use our existing Rust streaming engine, our wrapper and our independently trained C1 epoch-1 checkpoints. Read-only teammate snapshots are methodological references and are not included in the packages.

The frozen E511 comparison evaluated 468 exposed fold-0 sequences, 141 connected components and 808,863 scored rows using the official metric. C1 seed3 scored 0.681618 versus C0 0.679952; seed4 scored 0.681647 versus 0.679971. Their fixed equal-weight pair scored 0.682176 versus 0.681153. These are local development improvements, not a forecast that the public score exceeds 0.6660 or Maxim's 0.6674. Conditional bootstrap intervals and exact artifacts are in [E511 report](../local/E511_C1V2_C0_EVALUATION_v1/REPORT.md).

The two single-seed variants are built and passed static ABI, offline Python3.11 callback/reset and cross-build archived-output equivalence checks on v2, AVX2 and v3. Root independently replayed their numerical comparison against the exact own CPU float32 checkpoint reference: maximum differences 0.00007534 (seed3) and 0.00010380 (seed4), below the 0.0003 gate. See `ROOT_NUMERICAL_REPLAY_V1.json` and the six `*_python311_*.json` receipts. The cached GPU FP16 predictions are a separate diagnostic reference, not a float32 weight-quantization oracle.

| Candidate | Named ZIP SHA256 | Size bytes | One-core timing, microseconds/row |
|---|---|---:|---:|
| Seed3 | `95b4a730e61182b37f5bfd238aa8077430666bb9f5acfc5d76635bf23138163c` | 4,674,361 | 71.1 |
| Seed4 | `7d9059c6817b2fefeb98e6bcefa536f9c6488ff25697a7d3e58ee0603186667a` | 4,674,695 | 77.6 |
| Pair, do not upload | `e83909ecccbad8b5af3ab48bf316f7dda71ebfd13cf0da6ab95333ef5b9840ff` | 8,301,244 | 137.4 |

The previous public-best ZIP measured 86.3 microseconds/row on the identical 20-sequence, 400,000-row workload. At the harness's 39.4-million-row reference size, the estimates are 46.7 and 50.9 minutes for the singles, 56.6 for the previous best, and 90.2 for the pair. These are representative local timings, not scorer hardware measurements or runtime guarantees. The pair executes two recurrent trunks and is excluded from upload recommendation because it exceeds the 60-minute envelope in this test. Keep its built artifact as evidence.

These checkpoints retain their fold-0 training membership. They are not newly trained all-data production models. Do not silently relabel their ancestry or add the observed local delta to a public score.

The uniquely named single ZIPs in `submissions/c20261007_c1_s3_i16/` and `submissions/c20261007_c1_s4_i16/` are byte-identical to the tested `submission.zip` files. Serving tests cover two complete streams, reset-prefix replay and archived native output equivalence; numerical parity covers those two streams, and timing covers twenty. They do not establish a fresh full-fold CPU score or public performance. Public upload remains the human gate.

See [initial frozen pair contract](C1_PAIR_PACKAGE_CONTRACT_V1.json) and [independent implementation decision](../../../../specs/adr/0012-independent-teammate-method-implementation.md). The delivery operator owns the ZIP tests and exact terminal receipts; the independent critic owns provenance and evidence interpretation.
