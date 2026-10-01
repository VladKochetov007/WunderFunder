# First independent supplied-data audit: negative and unknown results

The fixed audit used 48 TRAIN groups and 24 distinct fold-0 validation components, with every evaluation component excluded from the TRAIN fit. Selection used component metadata hashes before label access. `sample_manifest.json` freezes membership; `results_primary.json` records source identities, mapping confidence gates, every evaluated rule and resources.

The strict gates left less than 0.1% of scored target mass eligible for the four-code-event test in classes 0–2 and zero eligible mass in class 3. All horizon/barrier comparisons are therefore unknown. This is not a falsification of the reported E-079 rule. The negative/unknown statement is preserved verbatim in `claim_ledger.jsonl`.

## Identification results

In all 24 held-out sequences, treating the price-like values as unordered literal quote levels produced zero positive-spread rows. Reversing a common monotone transform also produced zero. Thus neither direct common coordinate interpretation is viable. Per-column normalization remains possible; no physical price quantum or quote origin was recovered without labels. The four target quanta used for retrospective regime grading are supplied-label lattices, not recovered physical tick sizes.

The mean within-sequence fraction of rows whose repeated price code had multiple target-implied spreads was approximately 85.1%, 96.2%, 56.5% and 0% for classes 0, 1, 2 and 3, respectively. This supports treating label inversion as a diagnostic, not a current physical spread definition outside class 3. Reconstructing targets from their own inverted spread/reference is an algebraic identity and supplies no independent evidence for one physical future quote.

## Why coordinate validity blocked the horizon test

The separately labeled supervised decoder used only TRAIN target jumps and TRAIN code-spread modes, requiring at least three modal observations and 95% confidence. Unknown edges, ambiguous code spreads and disagreement between independently integrated ask/bid moves and the trained spread invalidated downstream intervals. Modal transition mass was 90.5%, 83.1%, 82.5% and 95.4% across the four TRAIN regimes. No mapping was learned or selected using evaluation labels/masks.

Existing `src/wnn/ticks.py` learns moves from target jumps and spread from targets, then overwrites bid as reconstructed ask minus a trained spread. Consequently its causal feature-only *application* is not label-free quote identification. The v1 consistency test was stricter than that existing reconstruction and cannot simply assume the joint jump pair supplies both physical quote moves.

There was also a specification mismatch: v1 advanced its event clock at every 22-column representation-code change, whereas E-079's `rebuilt_labels` advances at nonzero decoded quote changes. V1 remains preserved. V2 is a separately preregistered TRAIN-supervised constrained-transport diagnostic using a fresh fold-1 component panel, the corrected quote clock, and a feature-only coverage gate before evaluation label access.

## Readout novelty

A hard projection of two predictions onto one deterministic future-price law is not generally aligned with the metric. For spread one, an equal mixture of target pairs `(-3, +1)` and `(+1, -3)` has target-absolute-value-weighted means `(-2, -2)`, which is outside the deterministic spread-one payoff support. A shared latent distribution can produce the separate weighted target means; that is the rational readout mechanism already attempted in E-066/E-073/H-044. Its oracle-spread gain did not survive causal spread decoding. Another version requires newly verified causal coordinates, a better exit law, or a distinct supervision mechanism before it qualifies as a new research branch.

Resources: 11.68 CPU seconds, 12.10 wall seconds, peak RSS 1,230,852 KiB, exit code zero; one CPU thread, 4 GB cap, 600 CPU-second hard limit. Four independent payoff/event/uncertainty tests and Ruff checks passed. No external data or numerical state was read by this audit, and no competition model, trainer or feature pipeline was changed.

Primary inspected evidence: `research/experiments.jsonl` E-063/064/074/079, `notebooks/eda.py` sections 4–6, `src/wnn/ticks.py`, and `scripts/online_c3.py`. E-079 names a session scratchpad script; that original script is not provided by its ledger record. The independent implementation therefore tests declared mechanisms rather than treating the historical number as reproduced evidence.
