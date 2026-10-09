# Independent review handoff: E529 Vast lifecycle V1

Review scope is limited to `scripts/vast_campaign.py`, `tests/test_vast_campaign.py`, and this directory's Vast lifecycle README. No paid attempt was created during implementation or tests.

## Claims to verify independently

1. The `run` command can make one provider create request only after matching root admission, dispatch receipt, account scope, offer snapshot, current credit and the durable ledger reservation. A duplicate attempt ID cannot submit again; an ambiguous create is reconciled from its persisted pre-submit ID baseline and exact provider identity.
2. All mutating/status/SSH actions bind to the persisted creation receipt and current account scope. No operation adopts a pre-existing instance based on label alone, and teardown is impossible for an unrelated ID. Review `destroy_owned`, `reconcile`, the exact-ID CLI arguments and supervisor hard-stop path.
3. Live rows map `verification="verified"` and `is_vm_deverified=false` to admission; deverified or unverified offers are rejected. On-demand is bound to the fixed query because live rows omit type. Exact-ID revalidation searches by machine ID/GPU name/count, then filters for the selected ID. `allocated_storage` is bound to the requested disk. Offer `gpu_ram` and `cpu_ram` are treated as MiB; host worker limits are decimal GB checked in bytes; actual offer-shaped inputs such as 24576/42885 admit 24 GiB GPU/41.880 GiB host, while a 6144 MiB GPU is rejected. Ensure fees, effective CPU and CUDA fields are included in the bound offer snapshot.
4. The root cost quote is cent-exact and covers the bounded compute/storage lifetime, two worst-case staged input transfers, collector artifact cap, and image cold-pull allowance. Check any uncertainty in mapping instance download/upload rates to local-to-worker and worker-to-local transfer direction; the root quote should remain conservative.
5. Input paths are project-relative, regular, non-symlink credential-free files; sizes match the pre-existing identity receipt without local rehashing. Upload uses the fixed byte ceiling and wall-clock deadline across restarts; every staged file is remote-hash checked before atomic rename. Launch is durable-before-SSH and cannot be duplicated after an uncertain SSH result.
6. The runner calls `bash ./run.sh` without arguments and supplies `WNN_INPUT_MANIFEST` as JSON records `{path, sha256, size_bytes}`. The worker terminal manifest is attempt- and packet-bound, has `worker_terminal:true`, and the exact frozen artifact set. The supervisor collects and hashes `artifact_bundle.tar`, `result.json`, and `worker.log` even from a worker that recorded failure; terminal worker failure is not treated as scientific success. A partial collection should quarantine its destination, retry once and then trigger exact-owned teardown.
7. Startup waits for provider `running` before SSH, attaches only the controller public key to the receipt-owned instance, and pins its first host key in a per-attempt `known_hosts` file before strict SSH/rsync. `run` and `supervise` require a systemd `INVOCATION_ID`, resume an existing ledger row, stop at hard stop, and retry exact-owned teardown only through cleanup deadline. Review whether the deployment's systemd restart policy and total unit runtime match the persisted deadline.

## Focused verification already run

Under `scripts/limited` with 1 CPU, 1 thread, 1.5 GB memory and a 120-second task cap:

- `ruff check scripts/vast_campaign.py tests/test_vast_campaign.py`
- `ruff format --check scripts/vast_campaign.py tests/test_vast_campaign.py`
- `python -m py_compile scripts/vast_campaign.py tests/test_vast_campaign.py`
- `pytest -q tests/test_vast_campaign.py` (42 passed)

The tests use mock provider and mocked SSH/rsync only. They do not load credentials, call Vast, rent, upload worker inputs or touch any provider instance.

## Staging needed before root admission

Root must freeze the H217 code tar with one root `run.sh`; write six input records from the existing pinned data-identity receipt (about 37.1 GB total); choose a worker disk that fits inputs, packet, extracted code and maximum result/archive bytes; set upload timeout and all-in transfer/compute/storage estimates; and freeze the exact expected three outputs. `run.sh` should capture exit code and `worker_success` in `result.json`, always package available partial outputs into `artifact_bundle.tar`, and provide the launcher-captured `worker.log` in a manifest even after a fit failure. Hash the packet and construct the fresh root admission/dispatch receipts, then have the independent reviewer check these against the code before root launches.
