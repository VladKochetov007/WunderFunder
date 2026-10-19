# R12 preparation

R12 is a fixed CPU comparison of canonical pooled WMSE against a fixed, initial-gradient-norm-scaled canonical pooled-correlation objective. Both arms use the same R7-shaped MLP, seed, fit-only normalization, clipped predictions, 31,808-row fit panel, AdamW settings, and 128 full-fit updates. The frozen protocol and its limits are in [DESIGN.md](DESIGN.md); the design-only review is [DESIGN_REVIEW.json](DESIGN_REVIEW.json).

The implementation is split across [training_pair.py](training_pair.py), [probe_execution.py](probe_execution.py), [result_payload.py](result_payload.py), and the hash-gated `python -m idea_screens_r12.run_probe` entrypoint. It reuses the pinned R1 sample reader, R2 view, R3 panel helpers, R7 model/panel/runtime utilities, and canonical `wnn.nn` losses. The complete closure is pinned in [PREPARATION_MANIFEST.json](PREPARATION_MANIFEST.json).

Source-only CPU checks are recorded in [TEST_RECEIPT.json](TEST_RECEIPT.json). They use synthetic arrays and do not load the accepted panel, train a model, or access remote resources. Root owns staging, admission, execution, capture, interpretation, and publication; no R12 fit has run yet.
