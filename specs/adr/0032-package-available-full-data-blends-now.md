# ADR0032: Package available full-data blends now

Status: Accepted. Date: 2026-10-11.

## Context

The latest user asks to package submissions now, maximize validation and
cross-validation quality of the complete blend, compare Q4 and Q8, measure
diversity from seeds and model/training families, and use full-data final runs.
This supersedes the instruction in ADR0031 to wait for the exploration wave
to finish before delivering submissions. Healthy admitted research continues.

## Decision

Prepare a small set of the strongest available independently trained full-data
endpoints now. Existing uploaded and frozen ZIPs are preserved. Rank supporting
recipes using official held-out evidence where it exists, including clean
cross-fold comparisons and conditional contribution to fixed mixtures. Report
missing matched folds or changed parent/teacher recipes explicitly. Supplied
validation predictions from a final model fitted on those groups are in-sample
and cannot establish cross-validation performance.

Use a bounded candidate set and simple fixed weights. Record every tested
mixture, its selection exposure and exact component identity. Additional seeds
measure training variation; family diversity must earn inclusion through its
contribution to the endpoint. Lower prediction correlation alone is insufficient.
Do not add a local improvement to a public score or promise 0.67 from a proxy.

Build Q8 candidates first using the existing independently written native
serving implementation. A bounded Q4 precision screen is now authorized even
before a Q8 capacity failure. Describe the actual representation and backend:
four-bit-rounded weights run through an int8 kernel are not native INT4
inference or evidence of a twofold speed improvement. Recommend a lower-bit
candidate only with measured quality/capacity value and deterministic causal
reset, ABI, ZIP-size and CPU callback checks on its delivered bytes.

Human uploads remain the external gate. This changes no resource, provider,
spending, weight-independence or data authority. Own Vast B is stopped with no
credit; no restart, rental, top-up or billing change is authorized. Local and
Kaggle research continue under their existing measured admissions.

## Consequences

Only the timing requirement of [ADR0031](0031-cross-fold-top-tier-submission-release.md)
is superseded. Its clean ancestry, distinct-seed accounting, cross-fold ranking,
exposure and serving requirements remain binding. [ADR0030](0030-local-evidence-selects-submissions.md)
continues to govern local selection and public confirmation.

The root selects and integrates; a Luna/max actor packages, a separate
Luna/max grader evaluates, and a Luna/max critic independently reviews the
frozen deliverable. Full-data counterparts with incomplete matched recipe
evidence may be released as explicitly provisional competition choices.
