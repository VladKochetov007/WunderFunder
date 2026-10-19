# R15 records index

This directory preserves the R15 V1 launch and terminal evidence. The bounded CPU unit used invocation `5da8269acf294ac0bd02bd94ec4b59e0`; its result was captured and validated before stopping that exact unit. The resource summary is documentary and does not audit the scientific result.

- [Resource reconciliation](RESOURCE_RECONCILIATION.json) and [metadata-only review](RESOURCE_METADATA_REVIEW.json), also exposed as byte-identical aliases in the R15 packet root.
- [Launch transport receipt](LAUNCH_TRANSPORT.json), [root capture validation](ROOT_CAPTURE_VALIDATION.json), and [exact stop receipt](STOP_RECEIPT.json).
- [Terminal capture index](terminal_capture/README.md), with the raw capture envelope, exact extracted JSON/log bytes, status and resource snapshot.
- [Startup V3 transport record](startup_v2/README.md). Its [root startup review](startup_v2/ROOT_STARTUP_REVIEW.json) documents the prelaunch harness correction and confirms the corrected packet; [ROOT_CFG_SUMMARY.json](startup_v2/ROOT_CFG_SUMMARY.json) is a keys-only summary retained from the root packet.
- `startup_v1/` is preserved and was not executed. `source_v1/` and `source_v2/` preserve superseded source attempts; the executed launch used the corrected V3 source pins recorded in the startup packet and receipts.

The model archive is represented only by producer-reported metadata and a stat size. No archive or array payload was opened or copied locally. Scientific interpretation and result acceptance are recorded separately in [RESULT_AUDIT.json](../RESULT_AUDIT.json) and [RESULTS.md](../RESULTS.md).

[Source/preparation history](SOURCE_HISTORY.md) indexes V1/V2 preserved candidates and all local check logs. The compact [audit script](../audit_scalar.py) produced [V2 scalar audit](RESULT_AUDIT_V2.json); [V1](RESULT_AUDIT_V1.json) is preserved and [canonical records alias](RESULT_AUDIT.json) matches V2. [Historical implementation review V2](../IMPLEMENTATION_REVIEW_V2.json) precedes the executed [V3 review](../IMPLEMENTATION_REVIEW.json). These documentary source/review stages did not rerun scientific compute.
