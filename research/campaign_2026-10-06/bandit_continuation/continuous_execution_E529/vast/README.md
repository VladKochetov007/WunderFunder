# E529 Vast lifecycle adapter

[`scripts/vast_campaign.py`](../../../../../../scripts/vast_campaign.py) supervises one frozen, root-admitted Vast on-demand attempt from create through worker staging, exact artifact collection, teardown, and later billing settlement. It never selects an offer by FLOPS alone. The root controller chooses one offer from a fresh read-only search using the frozen workload, whole-attempt quote, measured throughput and capacity.

## Admission and ownership

Read-only `inspect` and `search` do not change account state. Search fixes verified, rentable, non-external, unrented, on-demand filters and caps results at 20. Live rows expose `verification="verified"` and `is_vm_deverified=false` rather than a reliable verified boolean or offer type; the adapter normalizes verification and binds on-demand to its fixed query. Revalidation searches by machine ID, GPU name and count, then requires exactly one row with the selected offer ID and matching frozen snapshot (the live exact-ID filter returned no rows). The query binds `allocated_storage` in GB to the requested worker disk. Vast offer `gpu_ram` and `cpu_ram` are normalized from the live API's MiB values to GiB before checking the 24 GiB minimum. The hashed offer snapshot retains raw values and normalized values, effective CPU cores, CUDA maximum, network rates, and storage rate. The worker host-memory cap is expressed in decimal GB and checked in bytes against raw MiB while preserving 20% headroom. Marketplace values remain prefilters; measure actual instance memory, GPU availability and I/O before admitting a fit.

For available account credit, the adapter uses `credit` when the live current-user response includes it, fails closed if that field is malformed, and falls back to the documented `balance` field only when `credit` is absent. Vast's [show-user API schema](https://docs.vast.ai/api-reference/accounts/show-user) documents `balance`; the [official CLI](https://github.com/vast-ai/vast-cli/blob/master/vast.py#L3636) displays `balance` and `credit` as separate `Balance` and `Credit` fields. The adapter's `inspect` output names which field it used and returns only an allowlist; it never emits the raw user response or `api_key`.

Root admission binds the exact intent, packet, request, account scope, source hashes, offer, dispatcher receipt, transfer caps, payment-policy receipt and deadlines. Admission expires within 120 seconds and is rechecked after provider preflight. The durable ledger reserves the full all-in ceiling before one create request. One attempt owns one instance; multiple first-cohort attempts may run under distinct attempt IDs while their combined reserved and settled spend stays at or below $4. Cumulative reservations and settled spend stay at or below $20, and later attempts require unique root evidence-release receipts within the remaining $16. A legacy `status: approved` field has no authority. The adapter does not modify account billing or top-up settings.

The root cost quote has exactly `compute_storage_usd`, `upload_usd`, `download_usd`, and `cleanup_usd`; their cent-exact sum must equal the attempt ceiling. Admission checks that the compute/storage reserve covers `dph_total` plus disk storage through the bounded cleanup deadline, upload reserve covers two worst-case `max_upload_bytes` staging attempts at the offer's instance-download fee, and download reserve covers `max_artifact_bytes` at the instance-upload fee. Root must add a quote for cold image pulls to the upload category because the worker image is fetched by Vast at instance startup. The hashed offer snapshot includes `inet_down_cost`, `inet_up_cost`, and `storage_cost`, documented by Vast's [official CLI field definitions](https://github.com/vast-ai/vast-cli/blob/master/vast.py#L3636). The cleanup deadline may be at most 30 minutes after hard stop. No top-up or alternate account funding is attempted.

An instance is owned only when its provider ID is persisted from the create response, or reconciled from a persisted pre-submit baseline plus one unique new ID, exact campaign label, selected machine ID and post-submit timestamp. Existing labels block create and are never adopted. Ambiguous create holds the reservation and cannot POST again. Every instance status, upload, collection and destroy checks account scope and the persisted receipt identity. Destroy refuses unrelated IDs before any provider call and verifies absence from the current account listing afterward. No bulk stop/delete operation exists.

## Frozen packet and input contract

The code packet is a small uncompressed tar archive. It has exactly one root-level `run.sh`; members must be regular files or directories with safe relative paths. Links, path traversal and credential-like file names are rejected. The current H217 packet stages six separately pinned source inputs totaling about 37.1 GB; keep those files outside the code archive.

The frozen intent declares each separately staged input as:

```json
{
  "source_path": "data/train.parquet",
  "source_sha256": "<hash from the existing frozen data-identity receipt>",
  "size_bytes": 123,
  "remote_path": "train.parquet"
}
```

Each source path must be project-relative and regular, its current size must match the identity receipt, and its SHA-256 must be lowercase hex. The adapter does not rehash large local inputs. The worker receives each file under `.worker/<campaign-label>/inputs/<remote_path>` only after the remote SHA-256 matches the receipt and the temporary upload is atomically renamed. A source hash mismatch prevents launch. Packet and input sizes share the frozen `max_upload_bytes` ceiling and one `upload_timeout_seconds` wall-clock deadline that persists across service restarts. Staging has at most two attempts; repeated or integrity failures are recorded and the exact owned instance is torn down without pretending a scientific run completed. The requested worker disk must fit the packet, inputs, unpacked code and maximum artifact allowance.

The adapter invokes `bash ./run.sh` from the unpacked packet with no positional argument and supplies these environment variables (no Vast API credential is sent to the worker):

| Variable | Value |
|---|---|
| `WNN_ATTEMPT_ID` | Frozen attempt ID |
| `WNN_PACKET_SHA256` | Frozen packet hash |
| `WNN_OUTPUT_DIR` | Frozen remote output directory |
| `WNN_INPUT_DIR` | Input staging directory |
| `WNN_INPUT_MANIFEST` | JSON list of each input as `{path, sha256, size_bytes}` |
| `WNN_EXPECTED_ARTIFACTS` | JSON array of exact collector paths |
| `WNN_MAX_RUNTIME_SECONDS` | Frozen worker runtime cap |

For the H217 worker, freeze the exact expected artifact set as `artifact_bundle.tar`, `result.json`, and `worker.log`. `run.sh` must write `result.json` and a tar archive of available outputs on both success and failure, with the worker exit code and success flag in the result. The launcher captures logs to `worker.log`; the script should emit no additional output after writing the final manifest. Set `worker_terminal: true` only after the process exits and all three files are closed and hashed. This marker means the worker finished; a nonzero exit or `worker_success: false` remains a failed experiment whose archive, result and log should still be collected. The final manifest is written as `artifact_manifest.json` in `WNN_OUTPUT_DIR` with exactly this structure:

```json
{
  "attempt_id": "<frozen attempt ID>",
  "packet_sha256": "<frozen packet hash>",
  "worker_terminal": true,
  "files": [
    {"path": "result.json", "size_bytes": 17, "sha256": "<sha256>"},
    {"path": "artifact_bundle.tar", "size_bytes": 123, "sha256": "<sha256>"},
    {"path": "worker.log", "size_bytes": 32, "sha256": "<sha256>"}
  ]
}
```

The collector fetches the small manifest, requires its attempt and packet bindings, exact expected path set, terminal marker, sizes and hashes, then writes artifacts into an empty attempt-specific local destination. A partial collection is quarantined and retried once into a fresh directory; a second failure is terminal and triggers teardown of only the receipt-owned instance. A missing manifest is not success. Provider `running` status alone is not evidence that fitting started or completed.

## Running and resuming

The public direct `create` command remains disabled. The combined `run` command requires exact `--confirm-create <attempt-id>` and a systemd service invocation (`INVOCATION_ID`) so process restarts resume the same persisted ledger entry instead of submitting a second create. Root must prepare the packet, separate input manifest, frozen intent, fresh admission, dispatcher receipt, durable ledger and empty destination first. The controller launches the command inside a restart-on-failure systemd unit with a bounded runtime:

```sh
systemd-run --user --wait \
  --unit="wnn-vast-${ATTEMPT_ID}" \
  --property=Restart=on-failure \
  --property=RestartSec=10s \
  --property=RuntimeMaxSec="${SERVICE_MAX_SECONDS}s" \
  .venv/bin/python scripts/vast_campaign.py \
    --ledger "$LEDGER" run \
    --intent "$INTENT" \
    --admission "$ADMISSION" \
    --dispatch-receipt "$DISPATCH_RECEIPT" \
    --destination "$DESTINATION" \
    --confirm-create "$ATTEMPT_ID"
```

Set `SERVICE_MAX_SECONDS` to the frozen cleanup deadline duration and retain the durable systemd unit/ledger until verified teardown. If the unit or host is interrupted, resume only the same attempt with `supervise`; never recreate it:

```sh
systemd-run --user --wait \
  --unit="wnn-vast-${ATTEMPT_ID}-resume" \
  --property=Restart=on-failure \
  --property=RestartSec=10s \
  --property=RuntimeMaxSec="${SERVICE_MAX_SECONDS}s" \
  .venv/bin/python scripts/vast_campaign.py \
    --ledger "$LEDGER" supervise \
    --attempt "$ATTEMPT_ID" \
    --destination "$DESTINATION"
```

Useful read-only and receipt-bound commands:

```sh
python scripts/vast_campaign.py inspect
python scripts/vast_campaign.py search --criteria path/to/offer-criteria.json
python scripts/vast_campaign.py status --attempt attempt-id
python scripts/vast_campaign.py reconcile --attempt attempt-id
python scripts/vast_campaign.py collect --attempt attempt-id --destination path/to/results/attempt-id
python scripts/vast_campaign.py destroy --attempt attempt-id --instance-id 123456 --confirm-destroy 123456
python scripts/vast_campaign.py settle --attempt attempt-id --receipt path/to/root-billing-receipt.json
```

Before SSH staging, the supervisor waits for provider status `running` without consuming staging retries. Startup has a bounded 30-minute deadline, capped by the admitted hard stop. An offline, exited or failed worker and an expired startup deadline enter receipt-bound teardown. The controller's existing public SSH key is attached only to the newly created, receipt-owned instance; it is never installed account-wide or sent with the worker packet. The first host key is pinned in a per-attempt `known_hosts` file only after ownership, account scope, running status and key attachment are verified. All SSH and rsync use that pin with strict checking; a later mismatch fails closed.

Read-only API commands parse only the lowercase `vastai` value from project `.env` through `python-dotenv`; the file is never sourced and the key is never placed in argv. SSH and rsync use strict host-key checking and bounded commands/transfers. Keep the durable ledger and runtime receipts. Do not clear or rewrite an ambiguous attempt to make the lane appear free.

## Validation boundary

`tests/test_vast_campaign.py` uses an in-memory provider and mocked SSH/rsync. It covers ambiguous create and no resubmission, account credit allowlisting, MiB/GB conversion, all-in fee caps, frozen input identity and transfer limits, worker tar safety, remote hash verification, one-shot launch, exact artifact collection, hard-stop teardown and refusal to touch unrelated IDs. These checks make no live API calls, create no instance and do not authorize a rental. Root must complete the independent review and apply its existing budget/admission gate before any paid creation.
