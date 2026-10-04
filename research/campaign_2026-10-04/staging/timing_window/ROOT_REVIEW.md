# Root admission review

The original never-run drafts were preserved under `pre_root_review_v1/`. Review found output-directory arguments, missing process affinity, and an aggregate timeout that could expire during predecessor fits. These were fixed before sealing; frozen scientific timing loops were unchanged.

The root validated actual prerequisite artifact hashes, sealed both timing workers, and approved a 6-hour oneshot window with the documented CPU/RAM bounds. No timing result is claimed at admission.
