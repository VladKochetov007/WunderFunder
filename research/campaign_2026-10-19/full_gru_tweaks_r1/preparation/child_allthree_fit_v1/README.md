# Conditional AllThree child preparation

This folder holds a local, fail-closed preparation packet for the single F0, seed-1001 AllThree arm. It does not admit or launch a run. The native NLL child is still an unaccepted prerequisite: its terminal acceptance and exact-stop receipts remain explicitly unbound, and `REMOTE_SUBMISSION.py` is intentionally absent.

The recipe uses the canonical `ema_sparse` representation: stable signed top-four-of-sixteen dictionary codes with the retained 112-wide normalized-raw residual, followed by causal signed innovation EMA with half-life 32 over the combined 128-wide channel. The dictionary stays frozen and shared. The model starts from parent epoch 3; the accepted NLL child is only a sequential prerequisite and never supplies initializer weights. The head and NLL objective, common 467-wide input, target, cache, inherited normalization, seeded batches, six-epoch fixed endpoint, and optimizer remain fixed.

`ALLTHREE_LAUNCH_TEMPLATE_V1.py` is derived byte-for-byte from the pinned actual NLL body, changing only the run name and representation-mode argv values and adding a conditional accepted-and-exact-stopped NLL predecessor gate. Both the local renderer and the eventual remote gate bind the NLL terminal acceptance to the original NLL launch acceptance SHA `44a708fc5a32b101c97a8a28c22e3b15e815059582de564e17f528f5369dc7a5`. A missing, malformed, mismatched, or still-active predecessor fails closed.

The binder's `--check` verifies the pinned template derivation and its unbound slots. `--render` exists only for a later turn after actual NLL acceptance and exact stop; it has not been run. No B-side staging, query, training, evaluation, profile replay, SSH, or model/cache payload reads are part of this packet.

See [ALLTHREE_RECIPE_BINDING_V1.json](./ALLTHREE_RECIPE_BINDING_V1.json) for frozen identities, source pins, argv delta, unchanged ancestry, resource gates, and pending receipts. [LOCAL_CHECKS_V1.json](./LOCAL_CHECKS_V1.json) records only this packet's focused local checks.
