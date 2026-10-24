# F0-only online grade review

This folder records an independent protocol review of the proposed exploratory
grade for the saved F0/seed1001 online relationship endpoints. The review
checks the frozen seven-endpoint question, artifact and ancestry gates,
official-metric handling, component uncertainty, exposure limits and the new
shared-memory budget. It is a review of readiness requirements, not a grade:
this reviewer did not open prediction or target payloads and did not compute
scores.

- [REVIEW.md](REVIEW.md) gives the pass/fail criteria, confirmed blocker in the
  frozen all-fold grader, and the claims a successful F0-only grade could
  support. Status: protocol-level conditional pass; execution/admission is not
  approved until the separate adapter and recipe pass the listed checks. The
  first adapter snapshot had a terminal source-pin schema mismatch, incomplete
  recursive ancestry, and did not enforce the 400 MB worker cap; see the
  code-review findings in the report.
- [REVIEW_R2.md](REVIEW_R2.md) independently rechecks the revised F0 adapter,
  recipe R2 and label-free preflight R3. Status: pass for root-controlled
  admission of the fixed exploratory grade, conditional on a live resource
  receipt; the reviewer did not open prediction/target payloads or score
  labels. It records the resolved source-pin, ancestry, memory, bin-delta and
  target-identity checks, while retaining the limits on exposed F0 evidence
  and inference precision.
- [RESULT_REVIEW_R1.md](RESULT_REVIEW_R1.md) independently checks the completed
  scalar grade, admission/source pins, recursive ancestry and cgroup receipt.
  Status: internally consistent as a fixed-endpoint, exposed-F0 comparison;
  all candidate WP scores are below the parent and all primary conditional
  component intervals cross zero. It records the near-limit memory peak and
  explicitly avoids a broad mechanism-kill claim.

Sources reviewed include the October 24 resume and marimo resource amendment;
the feasibility scout's report; the original online-relationship recipe,
grade specification, terminal inventory and metadata review; and the frozen
grader contract, scoring, target, cache, provenance, parent and output modules.
No remote/provider calls or experiment execution were performed.
