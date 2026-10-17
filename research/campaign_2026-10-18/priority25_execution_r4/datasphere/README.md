# DataSphere rank-1 profile packet

This directory prepares a sealed, fit-only profile job for rank 1 of
`E-RR18-01-LABELS`. It does not submit a job or authorize spending. The pending
DataSphere budget answer and the controller's live preflight are still required
before the paid `execute` call.

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
account-specific billing are separate. The observed project balance is not
spending authorization.

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
