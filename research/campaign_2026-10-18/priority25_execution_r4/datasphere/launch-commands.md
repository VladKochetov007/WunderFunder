# Durable profile launch procedure

The [18 October authority](../receipts/DATASPHERE_AUTHORITY_UPDATE_20261018.json)
permits discretionary bounded use of the existing DataSphere balance. The
numeric-budget wait is superseded; billing changes and top-ups remain prohibited.
This procedure is for the one sealed engineering profile, not a scientific cell.

Stage the sealed adapter, spec, requirements, source archive and supplemental
archive in a private directory on the linger-enabled Beleriand host. Make a
private config with relative archive paths and the exact job name
`computer-vision-homework-rank01-profile`. Preserve the pinned trainer source.
The config lists explicit upload inputs; controller state and credentials are
not upload inputs.

Fresh preflight must verify project access, no conflicting active job, balance,
`ALLOW_JOBS`, and `ALLOW_SPEC_G_2_1`. The mounted bytes and cloud runtime are
verified by the adapter inside the bounded job. Historical dataset IDs alone
do not prove their current contents.

The existing Mac profile supports IAM issuance with server expiry metadata:
`yc --profile default --format json iam create-token`. Capture this response
privately in process memory, check that the expiry covers the run and collection,
and send only the short-lived IAM token over encrypted SSH stdin into a mode-0600
EnvironmentFile outside the upload tree. Never print credentials or put them in
shell arguments, Git, archives, or provider configs. Beleriand's old OAuth
profile cannot currently refresh IAM; do not rely on it or change account auth.

Run the reviewed [controller](controller.py) from a persistent user systemd
service within `wnn-campaign.slice`, with CPUQuota=50%, MemoryMax=1500M,
MemorySwapMax=0, physical-disk TMPDIR, Restart=on-failure and
RestartPreventExitStatus=78. The campaign parent retains its aggregate two-core
and 20GB limits; this controller does not allocate a local GPU.

```sh
/usr/bin/python3 controller.py \
  --cli /home/grinchenkoeg/ds-venv/bin/datasphere \
  --job-dir /home/grinchenkoeg/computer-vision-homework-20261018/datasphere/r01-profile-v1 \
  --config /home/grinchenkoeg/computer-vision-homework-20261018/datasphere/r01-profile-v1/config.yaml
```

The service inherits `YC_IAM_TOKEN` through its private EnvironmentFile. It
writes SUBMITTING before executing once, persists the exact provider job ID,
and updates its status receipt after every successful poll. An ambiguous
submission stops for reconciliation; never retry blindly or cancel by name.

The controller requests forced cancellation at 3480 seconds from submission,
leaving nominal margin inside the one-hour allocation. The adapter separately
interrupts at 3300 seconds from script entry. Actual cancellation depends on
host clock, network and provider availability; there is no provider-enforced
execution deadline in this config. Continue exact-ID monitoring through terminal.

After terminal status the controller invokes `project job download-files` with
`--with-logs --with-diagnostics`, then marks COMPLETE. Inspect the actual bundle,
verify hashes and finite-update reports, reconcile provider timing and balance,
archive a sanitized receipt and remove the temporary credential file. COMPLETE
means collection finished; provider ERROR or a failed adapter is not a successful
profile. Future paired fits need separate measured allocations and cell claims.
