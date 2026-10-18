# DataSphere rank-1 profile packet

This directory prepares a sealed, fit-only profile job for rank 1 of
`E-RR18-01-LABELS`. The [latest user authority](../receipts/DATASPHERE_AUTHORITY_UPDATE_20261018.json)
permits discretionary DataSphere spending within the existing balance; the
numeric-budget wait is superseded. This packet does not submit a job by itself.
The controller's live host, credential, quota, dataset, and durable-cancellation
preflight must pass before the paid `execute` call.

**Current status:** the [v2 terminal incident](../receipts/DATASPHERE_PROFILE_V2_TERMINAL_INCIDENT.json) records providerERROR during preparation, no reported start/allocated machine and no produced files. The short-lived token has been removed. V3 is staged only; unchanged retries are held pending diagnosis or recovery. F1 preparation remains unallocated. A separate [runtime canary](canary_v1/README.md) also [failed before start](../receipts/DATASPHERE_CANARY_V1_TERMINAL.json) with INTERNAL13 and no files; both temporary credentials are removed and no paid job is active. Earlier admission and version notes below are historical.

The [v2 launch receipt](../receipts/DATASPHERE_PROFILE_V2_LAUNCH.json) records actual provider job
`bt13i2k83lo8n4m8kplk`, submitted at23:58UTC on17October. The durable service is
`computer-vision-homework-datasphere-r01-profile-v2.service` on Beleriand; its
state and outputs live under the campaign `datasphere/r01-profile-v2/` directory.
The original v1 attempt failed before provider creation: SDK0.10.0 turns a
fully manual empty `local-paths` list into `None`. The preserved v1 config
remains historical. Use [config.r2.template.yaml](config.r2.template.yaml) and
the inert [runtime_marker.py](runtime_marker.py) for this corrected envelope;
actual SDK environment/path/module/limit preparation now passes. Trainer,
adapter, recipe, data hashes and dependency versions are unchanged.

The one g2.1 job profiles three training batches for each of two cold-start
GRUs, H512×1 and H384×2. It makes no validation score, checkpoint selection,
or quality claim. The canonical trainer is imported from source commit
`a266b036b02e8e012c6eb31072291dad618a169d`; this packet contains only the
provider adapter and does not copy or edit the trainer.

The sealed [profile-spec.json](profile-spec.json) binds the source archive,
supplemental archive, graph/fold/split inputs, the complete train and valid
parquet hashes, and the target-index hashes and sizes from the fresh
portable-source registry. Beleriand reproduced the train, valid, and target
content hashes; they match prior expected values. The spec deliberately does
not invent a mapping from filenames to DataSphere dataset IDs: the adapter
searches the four existing mounts for exact basenames using filesystem
metadata only, with a 250,000-entry and 16-level depth bound. It rejects
missing, duplicate, or matching symlink names before opening file contents. It
then assembles the ordered parquet pieces, checks the whole-file identity,
copies the target caches under sealed size/SHA-256 checks, and records which
mount supplied each file. The train and valid target caches are 1,909,335,790
and 337,186,198 bytes respectively (2,246,521,988 total), with a 2.1 GiB cap
for each file. Total staged data is 37,102,250,127 bytes; the 100 GB disk check
preserves at least 10 GiB free. No local 34 GB parquet upload is needed.

Four existing dataset IDs are retained from the prior successful job. Current
provider admission of four mounts is unverified; Yandex's three-dataset limit
describes Jupyter project-disk attachments, while job admission must be
confirmed by the provider. Dataset paths and content are not live-validated
until a budget-authorized job reaches the adapter. The runtime also fails
closed if a mount is missing, ambiguous, or contains different data.

The job pins Python 3.12, Torch 2.14.0+cu130, NumPy 2.2.6, PyArrow 19.0.1, and
attrs 26.1.0. These versions are declared but have not yet been validated in a
DataSphere container. The 3,300-second in-process timer starts when the adapter
starts and covers archive hashing, all staging, and training. The controller
must cancel no later than 3,600 seconds from submission; this outer bound also
covers queueing and dependency setup before adapter entry. The config has no
provider-side execution timeout. At the observed public g2.1 rate, a full hour
is estimated at about $4.45 before VAT; working-storage charges and
account-specific billing are separate. The latest user authority permits
discretionary use within the existing balance, with no top-up or billing change;
the balance itself is not a new spending ceiling.

## Durable submission controller

[controller.py](controller.py) owns one run directory from submission through
terminal output collection. It writes an atomic, mode-0600 `SUBMITTING` receipt
before calling the DataSphere 0.10.0 CLI with the fixed project ID and private
`config.yaml`. It captures the CLI's JSON `job_id` and `operation_id`, then
monitors only that ID. A restart resumes the saved ID; a missing or malformed
submission response becomes an `AMBIGUOUS_SUBMISSION` and exits fail-closed
without retrying `execute` or cancelling by job name. A filesystem lock prevents
two controller instances from sharing the same run directory.

The controller polls every 30 seconds and retries transient status errors with
bounded backoff. At 3,480 seconds from the recorded submission start, it sends
the CLI's default forced `cancel --id JOB_ID` (no `--graceful`) and continues
polling until `SUCCESS`, `ERROR`, or `CANCELLED`. Starting the timer before the
network submission request gives the 3,600-second external stop target a
120-second margin. This is a client-issued cancel request, not a provider-side
execution deadline: host clock, connectivity, and provider responsiveness are
required to stop the job. The controller retries exact-ID cancellation every
60 seconds while the job remains nonterminal and keeps polling through outages.
Calls are individually bounded: 300 seconds for `execute`, 60 seconds for
status/cancel, and 600 seconds for output download. It downloads files, logs,
and diagnostics only after observing a terminal provider status. It does not
unpack or evaluate results.

Use one private durable run directory per attempt. Place the private config,
sealed spec, adapters, requirements file, and referenced source/data archives
where the config expects them. The service host must have a tested DataSphere
0.10.0 CLI, the exact job inputs, and durable user-systemd availability. A
short-lived `YC_IAM_TOKEN` is accepted through the service environment; the
controller never writes or prints it. The token must remain valid through the
cancel deadline, terminal polling, and output download because the CLI does not
refresh IAM tokens supplied through that variable. Alternatively, a verified
`yc` profile can be passed with `--profile`; in that mode the controller removes
ambient token environment variables and asks the CLI to use that profile.
Credential issuance, host verification, token lifetime, and cleanup remain
preflight duties. Never place a token in `config.yaml`, command arguments,
provider inputs, or the run directory.

Example user service; replace every path/profile placeholder with values
verified on the selected host. For token-based authentication, add one
`EnvironmentFile=` entry pointing to a mode-0600 file outside the job input
directory. Omit that entry when using `--profile`. Do not enable the unit until
the live preflight is recorded.

```ini
[Unit]
Description=One bounded DataSphere rank-1 profile
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=0

[Service]
Type=oneshot
TimeoutStartSec=infinity
WorkingDirectory=/private/datasphere/profile-run
EnvironmentFile=/private/path/outside-job-inputs/datasphere-auth.env
ExecStart=/usr/bin/python3 /absolute/path/to/controller.py --cli /absolute/path/to/datasphere --job-dir /private/datasphere/profile-run --config /private/datasphere/profile-run/config.yaml
Restart=on-failure
RestartSec=30
RestartPreventExitStatus=78
RemainAfterExit=yes

[Install]
WantedBy=default.target
```

`TimeoutStartSec=infinity` keeps systemd from killing the controller at its
default service-start timeout while it monitors the one-hour provider job.

Exit status 78 means local identity or submission state is ambiguous; systemd
must not restart it automatically. Inspect the saved private receipt and
reconcile the exact project/job identity before any operator action. The
controller never guesses an ID or uses a name-based cancel. Exit status 75
marks an operational failure such as a failed download; with a terminal job
receipt, a service restart retries collection without re-submitting.

Files in this packet:

- [config.template.yaml](config.template.yaml) defines the fixed job name,
  g2.1, 100 GB SSD, four historical dataset IDs, job inputs, Python runtime,
  dependencies, and local adapter files.
- [profile-spec.json](profile-spec.json) seals the current source/data hashes
  and the fixed profile recipe. It contains no credentials.
- [profile-spec.template.json](profile-spec.template.json) is the unsealed
  schema template with required hash placeholders and no assumed mount map.
- [profile_entry.py](profile_entry.py) validates the seal, starts the runtime
  deadline, stages inputs, validates the canonical split, runs both profiles,
  and writes a result receipt without interpreting quality.
- [profile_staging.py](profile_staging.py) safely extracts archives, discovers
  exact mounted filenames, assembles and hash-checks inputs, and bundles the
  output. Its synthetic safety fixtures are in
  [test_profile_staging.py](test_profile_staging.py).
- [profile_recipe.py](profile_recipe.py) freezes the two-arm command arguments,
  checks the declared runtime/GPU versions, and records valid profile/update
  evidence.
- [controller.py](controller.py) is the durable one-job submit, monitor,
  forced-cancel, and terminal-download controller. It does not contain tokens
  or change the frozen trainer recipe.
- [test_controller.py](test_controller.py) exercises interrupted submission,
  duplicate prevention, deadline cancellation, exact-ID validation, transient
  polling, private credential handling, locking, and terminal-only collection
  entirely offline (`python3 -m unittest -v test_controller` from this
  directory).
- [requirements.txt](requirements.txt) pins the Python dependencies and the
  PyTorch CUDA 13.0 wheel index.
- [launch-commands.md](launch-commands.md) gives the exact CLI shape and
  one-hour monitoring/cancel procedure; it has not been executed.
- [launch.env.example](launch.env.example) documents a token variable using a
  placeholder only. Never replace it with a real token in a file or transcript.
- [VALIDATION.json](VALIDATION.json) records static, CLI-config, archive, and
  synthetic fixture checks; it contains no live job or training result.

The CLI and project access findings are preserved in the sanitized
[DataSphere discovery receipt](../receipts/DATASPHERE_DISCOVERY.json).

## Prepared F1 pair

The [F1 adapter](F1_FIT_ADAPTER_README.md) reuses the frozen canonical trainer/exporter for F1/seed1001 wide/deep six-epoch endpoints. [Preparation and checks](../receipts/DATASPHERE_F1_ADAPTER_PREPARATION.json) record independent review and15 passing CPU fixtures, including signal deadlines and partial artifacts. Separate fit/collection deadlines remain unset. Provider file size/SHA verification and a name-bound follow-on controller are required before dispatch; no F1 job or quality result exists. The [full v2 error read](../receipts/DATASPHERE_PROFILE_V2_FULL_ERROR_RECONCILIATION.json) adds no internal diagnostic detail; the distinct no-op canary also failed before start, leaving the exact provider cause unresolved.
