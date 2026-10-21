# Conditional first-cell score admission V2

This V2 packet preserves the V1 conditional score-admission packet and changes one selected-GPU ownership boundary. The worker reads `/proc/<pid>/cgroup` and `/proc/<pid>/cmdline` only when the process directory owner UID matches the worker UID. Foreign GPU owners remain in the observation with PID, UID, GPU memory, and start-time metadata; their command and cgroup files are not opened. The before/after PID-directory identity check remains in place.

The five-arm F0, seed-1001 CPU scoring contract, unit, resource limits, receipt gates, staged files, read-only export inputs, result destination, and still-unbound export acceptance/stop pins are copied from V1. No B query, staging, admission, score execution, or quality result is authorized by this local packet. The root-owned source review must inspect the final V2 bytes before any rendering or admission.

Files:

- `SCORE_ADMISSION_BODY_TEMPLATE_V2.py` is the conditional remote body with the same-UID-only proc detail guard.
- `render_score_admission_v2.py` is the V2 renderer. It remains fail-closed while the root-owned export acceptance and exact-stop pins are unset.
- `SCORE_ADMISSION_BINDING_PLAN_V2.json` carries the unchanged score identity/resource contract and the V2 owner-inspection scope.
- `test_score_admission_v2_owner_scope.py` checks the foreign/same-UID read boundary and the body’s call-site binding without accessing `/proc` or invoking a GPU query.

The scorer/exporter source-pair review is only a dependency pin. This V2 packet has no independent source review yet and is not admitted or rendered.
