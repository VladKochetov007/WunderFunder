# Teacher seed-3 startup health capture

No durable health observation is available. One read-only SSH call was invoked, but the local wrapper raised `AttributeError: datetime.UTC` after subprocess return and before saving stdout/stderr. The raw bytes and return code were not preserved, so this receipt makes no claim about unit health, optimizer progress, or training failure. No retry was made. See [local failure receipt](LOCAL_CAPTURE_FAILURE.json) and [remote read-only source](REMOTE.py).


## Correction — recovered raw snapshot

A later local inspection found existing `STDOUT.raw` and `STDERR.raw`. The 16,740-byte stdout parses as the pinned remote script’s compact JSON serialization and matches the expected teacher S3 unit/invocation; stderr is empty. The original failure receipt is preserved: its wrapper failed before its own write calls, and the process that placed the subsequently found bytes is unknown. The SSH return code is also unknown. See [validated observation](OBSERVATION.json) and [correction receipt](CORRECTION.json). No remote command was repeated.

The first log line serializes CLI caps/defaults (`train_seqs=10607`, `valid_seqs=600`); it is not realized membership. The following membership line reports10535train+1405fit-valid/468evalfold groups, matching373drop-lastB32batches and the accepted profile. Source `data_membership.py` applies the current split before `setup_progress.py` emits actual counts. No evaluation predictions were made during fitting.
