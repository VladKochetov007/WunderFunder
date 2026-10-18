# Exact propensity asset recovery

This separate bounded helper restores only the archived F0 propensity model
and its provenance (187,249 bytes total). It does not load tensors, run inference
or fit. [Source](restore_propensity.py) reuses the exact hash-pinned redirect
and size/hash-checked streaming functions from the earlier reviewed restoration
helper; that source remains unchanged. Existing outputs are never overwritten.
The token remains in process memory, is stripped on redirects, and is not logged.

Pins come from [the archive inventory](../../../unified/models/huggingface/MODEL_INVENTORY_20261012.json)
and [the frozen revision receipt](../../../unified/models/huggingface/FINAL_PRIVACY_AND_INTEGRITY_20261012.json).
Root owns independent review, fresh headroom and execution: 25% CPU, 192MB,
120 seconds, no swap, CUDA hidden and physical scratch. This restores a missing
source dependency; it does not establish clean ancestry or resolve the old
fold-cache compatibility requirement.
