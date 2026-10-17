# Operator command template

These commands describe the submission path only. They were not run. Replace
the config input paths and create a sealed `profile-spec.json` from the fresh
asset/mount receipt first. Do not submit until the user’s DataSphere budget
answer is recorded and the Beleriand watchdog is armed.

From the repository root, prepare a private config copy in this directory using
`config.template.yaml`. Replace the three `/ABSOLUTE/PATH/TO/...` values with
the source archive, supplemental archive, and sealed profile spec paths. Keep
the token out of the file. The requirements file and adapter are included from
this directory.

```sh
cd /Users/nb/Documents/wunder/WunderFunder-r4-datasphere-packet/research/campaign_2026-10-18/priority25_execution_r4/datasphere
export YC_IAM_TOKEN='<fresh_iam_token>'
/Users/nb/ds-venv/bin/datasphere project job execute \
  -p bt1dqf9jg0tuhggcrj4k -c config.yaml --async -o submit.json
```

The caller must persist the returned job ID and submission timestamp outside
this packet, then poll with the configured DataSphere CLI and cancel at 3,600
seconds from submission if the job has not ended. Once a job ID is returned,
the same token-scoped CLI commands are:

```sh
/Users/nb/ds-venv/bin/datasphere project job get --id '<job_id>' --format json
/Users/nb/ds-venv/bin/datasphere project job cancel --id '<job_id>'
```

The Beleriand watchdog is mandatory because the job config has no provider-side
execution timeout field. The 3,300-second in-process timer starts only after
DataSphere has installed dependencies, started the container, and entered the
adapter; the external 3,600-second clock bounds that setup interval too.

Before the paid call, verify the provider's live unit balance, current project
limits, job admission, current dataset activation, and resource eligibility.
The prior successful job used g2.1 and four dataset IDs, but those historical
facts do not guarantee current admission. One g2.1 profile job is estimated at
about $4.45/hour before VAT at the observed list rate; actual billing depends on
the account's contract. A currently observed project balance is not a user
spending authorization.
