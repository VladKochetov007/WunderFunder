# E-495 independent source and specification review

Review date: 2026-10-06. This is an independent source-preparation review only.
It does not authorize a resource probe, fit, export, score, or provider launch.

## Verdict

GO for the preregistered, unsealed preparation. No material source/spec blocker
was found in the exact identity set below. Normal controller-owned resource,
quota, deadline, and sealing gates still apply before any run.

## Reviewed identity and evidence

| File | SHA-256 |
|---|---|
| `cadence/DESIGN.md` | `19a11506d0f60e73d4bac250c3747f92c2e40b8da2667de8ec78f56614c2a9f9` |
| `cadence/materialize_specs.py` | `d6497f0253a50f790bad6af9ceedbdf59381020eb444c8861e432f04dccd8c27` |
| `cadence/test_cadence.py` | `3b5780e3967a58d38c167834f6414f0088ea7b212e872a846bde39d2c51157fb` |
| `cadence/dependencies.json` | `85c0873679781ee5a993aa4ce67abaa357048ee8e6839945484e9b80978433ec` |
| `cadence/ledger_packet.json` | `065ae30b72d2332eec439fecc6790886f7869e1a72800a295caace40d5f01d7a` |

The materializer receipt records 8 CPU tests, 0 failures, 0 errors, with a
2,147,483,648-byte host cap and GPU disabled. Its receipt SHA-256 is
`b77e8b5e0caa3167f7ab4861ae9302a056b49816b9bb30f21e5e634d47c930e0`.
No review-owned rerun was needed; the note is bound to that exact receipt and
source set.

## Findings

The three arms use the existing strong recipe and differ only in the
registered row clocks: `(update, detach) = (1000,1000), (2000,1000),
(2000,2000)`, with 1,000-row microchunks. The actual controlled trainer pools
each update window's weighted numerator and denominator before applying one
optimizer/scheduler/clip/EMA step. Detaching at 1,000 rows preserves the
carried hidden-state values while cutting the second microchunk's gradient to
the first; the 2,000-row connected arm retains that credit path. The unit
fixtures exercise unequal mass pooling and the gradient distinction.

The row-clock EMA factors are `0.999` and `0.999**2 = 0.998001`. This matches
nominal retention per processed row, while the design correctly disclaims
identical EMA trajectories because student update times differ. The 20,000-row
sequence divides exactly into 20 or 10 update windows, with no tail. The fixed
epoch-3 parent and equal parent/head endpoint, three seed/fold blocks, fold
input lineage, and paired component-bootstrap screens are consistent with the
materialized drafts. Probes are full-shape, one-worker profiles and remain
diagnostic only.

The bounded screen is one seed/fold and the later blocks are confirmations;
these prepared experiments do not establish a public-score effect. The drafts
remain unsealed and record zero probes, fits, and score delta. Any trainer or
row-clock source drift invalidates this review because the materializer fails
closed against the reviewed implementation.
