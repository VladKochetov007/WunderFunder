# Full-fit admission V1 — pre-mutation guard rejection

[Raw stderr](STDERR.raw) records the rejection in `_data` at the registered `data/cache/chain_folds.npz` symlink; [transport](TRANSPORT.json) and [original submitted source](REMOTE_SUBMISSION.py) preserve the sole attempt. This occurred before `OUTPUT.mkdir`, systemd submission or any fitting. Do not blindly repeat it.

A bounded root readonly inspection found the alias resolves to `B/receipts/chain_folds_derived.npz`,462719B, within the same B root. The exact unit was not-found/inactive/dead and both planned output/run paths were absent. These are root diagnostic scalars from the tool response, not a byte-identical raw receipt. V2 must allow only source-registered, profile-identity-matching in-B regular-file aliases. It must retain source and data identity gates. No training failure or quality result is implied.
