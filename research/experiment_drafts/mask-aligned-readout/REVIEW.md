# Independent source review — 2026-10-22

A separate gpt-6-luna/max reviewer checked fit counts, masks, lineage, sources and
the held-out comparison. No publication-blocking document errors were found.
The four head fits and zero new parent fits, actual-mask exclusion from inputs,
complete evaluation coverage and exploratory uncertainty agree across files.
Existing H-023/H-030/H-035 and pinned parent metadata support the stated lineage.
The root review traced strict FP32, prefix freezing and mask construction directly
in canonical source. Because propensity overrides the mask branch, both arms
explicitly clear propensity. Only the new computations claim FP32.

Parent artifacts, full source-clean ancestry/manifests and executable settings are
unbound. The review accepts this proposal for draft publication, not execution or
scientific/submission readiness. Public source links were pinned and “no parent
fits” clarified to “no new parent fits” after independent review. No tests, model
imports, fits or inference ran in this task.
