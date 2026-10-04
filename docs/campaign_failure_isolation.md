# Contained scientific failures

A positive child exit below 128, a clearly missing/content-changed declared input,
a missing required output, or the worker's own execution deadline can quarantine
that specification's job name. The worker must first confirm unchanged resource
limits, no new worker or aggregate-parent OOM events, readable final monitoring,
the GPU operational threshold when applicable, and a cgroup containing only its
monitor after child teardown. Other job names remain admissible and active jobs
continue. Each terminal attempt retains its exit, signals, elapsed/billed time,
initial/final resource evidence and failure scope/category.

Negative exits and shell signal-like exits >=128 remain global incidents. So do
source/readiness drift, unknown failures, uncertain launch or teardown, monitor
failure, resource-limit mismatch and any OOM event. A per-job deadline whose
cleanup cannot be verified also pauses globally. Lost monitors are still charged
the full reservation and reconciled as global uncertainty. The first global
incident remains the PAUSE reason; later cascading interruptions are separate
incident records that reference it.

There are no automatic retries. A reviewed job-specific acknowledgement names
the current incident and preserves its evidence:

```
scripts/run_campaign.py acknowledge-job JOB INCIDENT_ID --review 'diagnosis and correction'
```

This acknowledgement can occur while independent jobs are active, but not while
the same job still has an active attempt. It does not change READY's code hashes,
clear a global pause, reuse an attempt ID, erase previous outputs or reset time
accounting. Corrected code still requires the normal between-jobs readiness
review. The quarantine identity is the explicit spec job name; the controller
must not invent a new name merely to bypass its unresolved incident.

Integration requires a new reviewed READY record between jobs. No legacy pause
marker or live unit is changed by this code patch.
