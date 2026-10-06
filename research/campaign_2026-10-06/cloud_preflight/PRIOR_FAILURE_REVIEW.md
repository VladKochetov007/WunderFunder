# Prior cloud failures reviewed before new packet admission

This preflight uses new immutable Kaggle IDs and new source packets. It does
not resubmit either prior failed packet.

## W384 compatibility failure

The W384 v2 CUDA preflight reported 24 failures in legacy fixtures. The
fixtures deliberately shorten Parquet row groups to eight rows and monkeypatch
`wnn.data.SEQUENCE_LENGTH`; the transported loader had imported that constant
by value, so it retained the production length. The reviewed W384 v3 transport
fix resolves the value from `wnn.data` when opening the file and still checks
the production 20,000-row contract, decoded offsets, and single-sequence row
groups. The recorded width-transport suite passed 16/16 checks and the staged
legacy integration suite passed 36/36. Evidence: the main worktree's
`research/campaign_2026-10-05/continuation_night/kaggle/W384_V3_FIX.md`.

This root-GRU hardware packet does not use the W384 loader or launch a W384
fit. It loads the separate, hash-verified 256-hidden epoch-3 combo checkpoint
from the private E463 source dataset and uses a 2,000-row model chunk.

## E463 TPU errors

The real-data E463 sequence has three distinct failures. E463 v1 failed before
the worker started because a file-hash helper received an unsupported
`deadline` keyword. E463 v2 ran on CPU rather than an assigned TPU, then hit a
2 GiB host RSS cap at about 2.31 GiB before a training step. The later v3
`wnn-night-e463-tpu-20261005-v3` reports provider `ERROR`; its downloaded Kaggle
log is exactly `[]`, with no worker output. The raw download and status records
are preserved under `diagnostics/prior_errors/e463_v3/`.

The new TPU packet will therefore be a separate, smaller device gate. It will
write a Python startup record before importing JAX, run JAX in a bounded child
process whose stdout/stderr and exit status are saved, require eight actual TPU
devices before model work, collect per-device HBM at every stage, and write
terminal JSON in `finally`. It uses the already-published private E463 source
dataset as the hash-verified code/checkpoint source and the private
`wunder-funder` dataset as an attached data source. It cannot establish a
scientific score.
