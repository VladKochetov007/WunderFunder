# Read-only DataSphere and incident observation

Observed at **2026-10-18 16:12:31 UTC** with the pinned v2 observer (SHA-256 `86373399fe4ce1c29bb4335ec2a23c190bb85cedb4e04dd62f578a8e3bb23bd6`). Its exact CLI was `/Users/nb/ds-venv/bin/python /Users/nb/Documents/wunder/runtime-20261018/datasphere/v2-terminal/ds_readonly_observer_20261018_v2.py --live-read-only`; all three calls succeeded in 5.16 seconds.

- Balance: **41,652,791 units**. The provider restriction record reports `UNIT_BALANCE_LIMIT=41,652,791`, `MAX_VM_COUNT_PER_PROJECT=18`, and `ALLOW_JOBS=true`.
- Recent job list: 100 returned, sorted newest-first; 46 `SUCCESS`, 33 `ERROR`, 21 `CANCELLED`; zero nonterminal jobs in that returned page. The two newest listed attempts are terminal `ERROR`.
- Official incident [2136](https://status.yandex.cloud/en/incidents/2136): **state unobserved**. The page fetch rendered a JavaScript-disabled notice, not incident details; this is not evidence of recovery.
- This was a read-only observer refresh. It reports `provider_mutations=false` and `credential_persisted=false`; no paid tests or Zen actions were made.

Evidence files: [observer API response](DATASPHERE_READONLY_OBSERVATION.json), [invocation and output binding](DATASPHERE_OBSERVER_EXECUTION.json), and [official status fetch](OFFICIAL_STATUS_INCIDENT_2136.json).
