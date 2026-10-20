# Preserved local terminal-checker draft

The preserved [checker](teacher_terminal_check.py), [tests](test_teacher_terminal_check.py) and [preparation receipt](TEACHER_TERMINAL_PREPARATION.json) are the exact bytes before the final independent criticism was resolved. This draft was never staged or executed on B and never opened an actual checkpoint.

The final [checker and preparation proof](../TEACHER_TERMINAL_PREPARATION.json) supersede this draft. They derive the canonical seeded train/evaluation order from the pinned split and bind normalization identities to the actual saved mean/std tensors. The original draft checked membership sets and cross-compared declared normalization identities. These were local readiness defects, not failures of the running teacher or scientific results.

The current preparation receipt records all three preserved SHA256 digests. Do not execute this historical copy; its sibling runtime dependency is intentionally not duplicated here.
