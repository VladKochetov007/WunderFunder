# DataSphere rank-1 profile packet

This directory prepares one bounded, fit-only profile job for rank 1 of
`E-RR18-01-LABELS`. It does not authorize or submit a paid job. Submission stays
blocked until the user answers the pending campaign budget question and the
controller confirms the provider's current project balance and its one-hour
watchdog.

The profile runs three batches for each of two cold-start GRUs: hidden 512 with
one layer, then hidden 384 with two layers. The canonical trainer is imported
from source commit `a266b036`; this packet contains only the thin staging and
invocation adapter. It performs no validation scoring, checkpoint selection,
or quality fit. A successful profile is throughput and finite-update evidence
only.

`config.template.yaml` preserves the four dataset mounts used by the prior
successful job. The public three-dataset limit concerns the Jupyter disk; job
admission for four mounts is still uncertain and must be validated by the
provider on a budget-authorized run. Historical path-to-hash metadata does not
map each file to a mount slot. `profile-spec.template.json` is intentionally
unsealed: replace its required placeholders from the sealed source and
supplemental archives and the fresh Beleriand mount map before building a real
`profile-spec.json`. The entry point rejects unresolved hashes and mappings.

The adapter reassembles train and valid parquet from the existing mounts,
copies only explicitly declared small target-index files, stages the supplied
graph/fold/split inputs, validates identities through the canonical portable
source registry, checks free disk space, pins CPU thread pools to one, and
enforces a 3,300-second in-process deadline. The controller must cancel the job
no later than 3,600 seconds after submission, including provider setup time.
Both limits are safety bounds, not provider-enforced job timeouts.

See `launch-commands.md` for the exact CLI shape. Commands there are examples
with placeholders and were not executed. Never put a token in a config,
archive, log, or command transcript; use `YC_IAM_TOKEN` in the submitting shell.
