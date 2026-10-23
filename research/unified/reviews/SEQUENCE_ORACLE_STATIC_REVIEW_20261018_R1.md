---
kind: review
status: complete
created_utc: 2026-10-18T05:33:59Z
updated_utc: 2026-10-18T05:33:59Z
hypothesis_ids: [H-SEQUENCE-EXPERT-PREFERENCE-PERSISTENCE-20261018-R1]
experiment_ids: [E-SEQUENCE-ORACLE-PERSISTENCE-20261018-R1]
evidence_paths:
  - research/campaign_2026-10-18/online_features_r1/sequence_oracle_r1/recipe.json
  - research/campaign_2026-10-18/online_features_r1/sequence_oracle_r1/grade_sequence_oracle.py
  - research/campaign_2026-10-18/online_features_r1/sequence_oracle_r1/sequence_oracle_sources.py
  - src/wnn/splits.py
  - src/wnn/gmetric.py
  - src/wnn/evaluation.py
---

# Early-to-late expert preference diagnostic static review R1

## Question or claim

Does the frozen sequence-level selector diagnostic have a source, timing,
scoring, or uncertainty blocker before its bounded profile and exposed-F0
grade?

## Mechanism and method

The independent critic reviewed the allocation packet, recipe, grader, source
verifier, official metric, component bootstrap, and split loader without
opening target or prediction payloads, running the grader, or editing code.
The packet selects TL3 or R3 per sequence and target from rows 99–4,999, then
freezes that choice for rows 5,000–19,999. It is a label-informed persistence
diagnostic, not a deployable router or online-learning method.

## Evidence and interpretation

**PASS; no blocker to the planned bounded profile or exposed-F0 grade.** The
review confirmed the declared hashes for the grader, verifier, metric,
bootstrap, split loader, contract, and split; prediction cache and sidecar
identities and shapes are checked before target arrays are opened. The split
loader validates the target cache's row-group/sequence-index digest before
those arrays are accessed. Window boundaries and tie handling match the
recipe. Scoring uses the canonical official mask and `global_wp_clip2_v1`.
Component-bootstrap comparisons use the frozen seed, replicate count, and
component unit.

The review caught one documentation error: the README said the target archive
was streamed. The grader materializes the full target and mask arrays from the
compressed NPZ after identity checks. The README was corrected before any
grade. The subsequent eight-group profile confirmed that the archive can be
loaded under its 2 GB worker reservation; see the profile and resource receipt
in the experiment packet.

## Limits and validity

F0 and both endpoints are repeatedly exposed; TL3 ancestry is unknown, and R3
was selected on F0. The selector itself uses validation labels and is not
available to the test callback. The late-label selector is hindsight on its
scoring segment, not a formal upper bound under the pooled metric. Bootstrap
intervals condition on fixed caches and early choices, so they do not measure
training-seed, fitting, target-feedback, or public-transfer uncertainty.

The recipe pins target archive size, mtime, and its index digest, while the
grader records a whole-archive SHA during profile/grading rather than pinning
that SHA in the frozen recipe. The split loader verifies the index before
opening target values, so this was not considered a blocker for this exposed
diagnostic; whole-archive pre-pinning would strengthen byte-level provenance.

## Decision and next step

Proceed with the bounded profile and the predeclared diagnostic if resource
admission succeeds. Interpret any result only as evidence about early-to-late
expert-preference persistence on this exposed endpoint. Do not use it alone to
create a submission or claim a useful label-blind router. A B300 follow-up
requires a separate, label-blind input signal and a reason to expect meaningful
student or teacher value.

## Source records

- [Experiment packet](../../campaign_2026-10-18/online_features_r1/sequence_oracle_r1/README.md)
- [Recipe](../../campaign_2026-10-18/online_features_r1/sequence_oracle_r1/recipe.json)
- [Grader](../../campaign_2026-10-18/online_features_r1/sequence_oracle_r1/grade_sequence_oracle.py)
- [Source verifier](../../campaign_2026-10-18/online_features_r1/sequence_oracle_r1/sequence_oracle_sources.py)
