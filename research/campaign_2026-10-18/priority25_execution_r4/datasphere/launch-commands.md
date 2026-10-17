# Operator command template

This submission has not been run. Do not submit until the user has answered
the DataSphere budget question, the root controller records that ceiling, and
the preflight confirms current project access, balance, quota, g2.1 admission,
and dataset mounts.

From the maintained repository, copy the template to a private working config
in this directory. The template already points to the sealed profile spec and
the two exact local archives. Keep the config out of Git if you edit it, and
never put a token in any config, archive, log, or command transcript.

```sh
cd /Users/nb/Documents/wunder/WunderFunder-maintenance/research/campaign_2026-10-18/priority25_execution_r4/datasphere
cp config.template.yaml config.yaml
export YC_IAM_TOKEN='<fresh_iam_token>'
/Users/nb/ds-venv/bin/datasphere project job execute \
  -p bt1dqf9jg0tuhggcrj4k -c config.yaml --async -o submit.json
```

The token above is a placeholder; use the refreshed token only in the
submitting shell. The DataSphere CLI uses it for authentication. Preserve the
returned job ID and UTC submission timestamp in the campaign receipt outside
this packet. Poll and cancel through the same token-scoped CLI:

```sh
/Users/nb/ds-venv/bin/datasphere project job get --id '<job_id>' --format json
/Users/nb/ds-venv/bin/datasphere project job cancel --id '<job_id>'
```

The Beleriand controller watchdog must cancel by 3,600 seconds from submission
if the job has not ended. The adapter independently interrupts at 3,300 seconds
from script entry, including archive hashing, dependency/runtime checks, mount
metadata discovery, parquet assembly, and training. DataSphere's job config
does not provide a provider-enforced execution deadline.

The prior successful job used g2.1 and the same four dataset IDs, but current
admission is not established. The latest project-only query showed no visible
active jobs; it did not inspect community-wide activity. `Project.GetRestrictions`
returned HTTP 403, so g2.1 quota/whitelist remains unverified. The observed
project balance was 41,652,791 billing units (about 99.74 g2.1 hours or $443.84
at the public net-of-VAT rate); it is not a spending limit or user
authorization. The one-hour GPU estimate is about $4.45 before VAT, with
working-storage charges and contract billing separate.
