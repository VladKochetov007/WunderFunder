# H217 A3 transport repair plan

The A2 attempt failed in staging, before `run.sh` launched. The owned instance was verified destroyed at `2026-10-07T22:40:30Z`; its partial transfer is gone. Do not reconnect to or revive A2. Any successor needs a new root admission and creation receipt. This is transport evidence, not a result about H217.

## Measured staging evidence

The A2 upload began at `22:10:28Z`; the last read-only remote-size receipt was taken at `22:39:22.774498Z`. At that snapshot, 2,249,841,882 bytes were fully hash-verified (the packet and four cache files), and `datasets/train.parquet.part` held 20,825,505,792 bytes. Of the 37,105,570,021-byte total upload, about 14,030,222,347 bytes remained. The partial train transfer averaged about 12.01 decimal MB/s over the full 1,734-second interval; counting all completed and partial remote bytes gives about 13.31 MB/s. The 1,800-second deadline therefore expired before the large source files could finish.

At the observed 12–15 MB/s range, the complete 37.1 GB transfer would take about 41–52 minutes before remote SHA-256 verification and launch. A 5,400-second stage ceiling gives at least 38 minutes of margin over the slow end of that observed transfer estimate while remaining bounded by the attempt’s existing hard stop. It is a maximum for a freshly frozen intent/admission, not a mutation of A2’s 1,800-second receipt or a guarantee that the next rental will sustain that rate.

## Narrow controller change

The adapter now allows a new intent to bind an upload deadline up to 5,400 seconds. It applies the earlier of that upload deadline and the immutable `hard_stop_utc` to every blocking SSH/rsync operation. Uploads use rsync `--partial --append-verify` to reuse the same receipt-bound `.part` path after a transport interruption. The existing remote SHA-256 check and atomic move still gate each completed file; launch occurs only after every file matches the frozen manifest. Host-key checking remains strict and attempt-local.

At worker launch, `WNN_MAX_RUNTIME_SECONDS` is capped to the lesser of the frozen runtime and the actual remaining hard-stop runway minus two bound reserves: the packet's 450-second dependency-install timeout and the supervisor's 300-second artifact-collection timeout. The 450-second setup cap matches `run.sh`'s `timeout 450s pip install` command and is frozen in the successor intent and root admission; the 300-second collection timeout is recorded in the job manifest and launch command. The effective runtime is persisted in attempt status and passed to the packet's profile gate. The H217 worker reserves a further 90 seconds inside that cap for its finalization path; if the cap leaves no positive worker window, launch fails closed and the supervisor tears down the owned instance. A long upload therefore reduces the worker window instead of allowing a fresh full-runtime estimate past the admitted paid hard stop.

The 5,400-second upload deadline and 450-second setup timeout must be repeated exactly in the new root admission; the job hash binds the 300-second collection setting. A3B preserves the H217 packet/source, six input identities and byte ceiling, worker disk request, image, and six-hour worker ceiling. No source data are rehashed or altered by this code change. A2’s ledger and the first-cohort reservation remain untouched. The $2.00 reservation is a ceiling, not its actual bill.

Focused validation after the adapter change:

```text
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=500000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python -m pytest -q tests/test_vast_campaign.py \
  -k 'upload_resume_window or upload_calls_never_outlive or rsync_push_resumes_partial'
5 passed, 48 deselected in 0.49s
```

The current full adapter suite is 55 passed in 0.88s; Ruff and `git diff --check` pass for `scripts/vast_campaign.py` and `tests/test_vast_campaign.py`. Focused tests cover the upload deadline, exact root-admission binding, hard-stop clipping, resumable rsync with attempt-local strict host pinning, setup-timeout admission binding, effective worker-runtime calculation, and refusing launch without enough worker/collection runway. No worker or provider instance was created by this repair.

## Optional faster data path

The installed Kaggle CLI is 2.2.4. Its dataset-download code asks the authenticated Kaggle SDK for an `ApiDownloadDatasetRequest`; the SDK models the response as `HttpRedirect` with a URL and expiry, and the CLI then downloads through the returned response. This supports a controller-mediated, direct HTTPS pull as a candidate, but does not establish the redirect lifetime, access requirements, or worker egress rate for this dataset.

If root chooses that path for a fresh owned worker, keep the Kaggle credential only on the controller. Request a redirect for the exact dataset file/version, pass the short-lived URL to the worker only over the pinned SSH channel’s stdin (never command-line arguments, environment, logs, packet, or receipt), fetch to a temporary file, and require the frozen byte count and SHA-256 before atomic rename. The worker receives no Kaggle or Vast API key. Before transferring tens of gigabytes, use the admitted worker for a bounded HEAD/range probe to verify outbound access, redirect validity, and useful throughput; any download failure remains infrastructure evidence. Continue to upload the small packet/cache inputs over the current verified path. Do not switch to this path until the exact-file identity and secret-handling checks are independently reviewed.

## A3 admission gate

There is no A3B offer admission or rental in this plan. Root must freshly revalidate the exact offer and price, bind the complete transfer quote, the selected transport, the 5,400-second upload bound, the 450-second setup cap, 300-second collection setting, unchanged H217 source/data hashes and disk ceiling, and the existing cohort hard stop in a new admission. The prepared offer snapshot is a candidate only and may have expired. Keep the authorized cumulative cap at $20, do not top up, and reconcile A2's charge from an itemized bill when it becomes available. The invoice query at `22:44:59Z` returned one instance-charge row but no exact match for contract `54725432`; actual A2 spend is still unsettled. The provisional account-credit change is recorded separately and must not be inserted as an exact invoice amount.
