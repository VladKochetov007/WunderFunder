# E482 versioned grader v2 review

Root inspected the v1→v2 diff and independently reran all5 updated synthetic tests under1CPU/2GiB/no swap; all pass. The exact source remains frozen at5154c1c4b1473e90f92d430aeb18604fc0931c55f66bcae400c75618b9469694 and tests0718cc37686d5760678eaeaf948240671827878e80593d463799b88fc961758a. The correction accepts only the observed five-field Kaggle metadata normalization via exact full-object comparison, retaining private input/hardware/internet semantics, the local submission hash and provider-inventory hash, and requiring byte-identical posted executable source.

The original actual gate failure remains in GRADER_PROVIDER_METADATA_INCIDENT_V1.md/log and frozen v1source. Other provenance/checkpoint/metric/contrast logic is unchanged from the original independent review. GO for actual fit-only proof; explicit separate CPU scoring admission still requires that gate's success. No test rows were read during review.
