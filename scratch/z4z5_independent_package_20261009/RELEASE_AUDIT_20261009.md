# Z4/Z5 independent package release audit — 2026-10-09

This note binds the two exploratory packages to their exact ZIPs and release-check receipts. Both passed the requested technical delivery gates. Neither has a held-out, public-score, or promotion claim.

| Candidate | ZIP SHA-256 | ZIP bytes | Release-check SHA-256 | Status |
|---|---|---:|---|---|
| `c20261009_z4_combo_i16` | `29e6b89b2673744e9c1ae8abbed4281f126171863bc07e813420cdbea28667dd` | 5,172,422 | `ddaaede420627e7efcbb648147554d6dee320e830408407d4fb5b51ea808bc69` | technical gates passed; exploratory, unpromoted |
| `c20261009_z5_combo_i16` | `60834d3190b76832f038abbdc505c6ac526d44b8f3a191c2e980e2f14f1ee1cc` | 5,172,638 | `85f2a7d6af193818536f3bf2920012b9ebdd70b6e577a0a57b53e9c272c5e0d2` | technical gates passed; exploratory, unpromoted |

The ZIPs and JSON receipts are in the main worktree at `submissions/c20261009_z4_combo_i16/` and `submissions/c20261009_z5_combo_i16/`. The recorded 20,000,000-byte package limit is met. SHA-256 values above were recomputed directly from those files after package generation.

For each package, `CORE_DERIVATION_AUDIT.json` binds the actual 31-row student checkpoint and provenance to the actual derived 23-row core checkpoint. The retained primary-output rows and every non-family tensor are bitwise equal; eight family rows were removed. The core provenance checkpoint map binds the derived checkpoint's own digest. `TENSOR_PROVENANCE_AUDIT.json` then binds the derived core and HFT checkpoint through the fixed 0.5/0.5 tensor merge to the packaged ZIP. The HFT's recorded `initial_checkpoint_sha256` equals the derived core checkpoint SHA-256 in both release receipts.

The remaining release gates also pass: static ABI checks report no Python-linked dependency, Python 3.11.16 callbacks for v2/avx2/v3 each cover 40,000 rows with zero maximum deviation from the v3 reference, and full-stream A/B/A checks report bitwise reset and past-only behavior. The timing comparison used the same 400,000-row native fixture and CPU cap for candidate and incumbent:

- Z4: 96.7 µs/row versus 91.8 µs/row, ratio 1.05338; calibrated estimate 39.10 minutes.
- Z5: 86.7 µs/row versus 91.8 µs/row, ratio 0.94444; calibrated estimate 35.05 minutes.

These runtime estimates use the exact seed-4 incumbent ZIP `c4dc258b529a52267a5e108ef0936f7919221cbdd17913270c8eb3482557f94a`, whose portal runtime was 2,227 seconds. They are matched local replay estimates, not candidate public runs. The prediction fixture is exposed validation data; its score diagnostics are not held-out evidence. Both student/core and HFT fits include validation groups, so no held-out claim is valid. No public score or expected-public improvement is asserted.

The audit used the existing actual-producer receipts in each package directory: `CORE_DERIVATION_AUDIT.json`, `TENSOR_PROVENANCE_AUDIT.json`, `ABI_STATIC.json`, `PY311_V2_CALLBACK.json`, `PY311_AVX2_CALLBACK.json`, `PY311_V3_CALLBACK.json`, `CAUSAL_RESET_A_B_A.json`, `TIMED_REPLAY.json`, and Z4's `INCUMBENT_MATCHED_TIMED_REPLAY.json`. `RELEASE_CHECK.json` records their exact digests and the final status. This review adds no new fit, upload, or score evaluation.
