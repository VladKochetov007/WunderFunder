# Conditional first-cell score admission V3

This separately versioned packet corrects V2's overly broad privacy gate. It reads `/proc/<pid>/cgroup` for every selected-GPU owner so campaign cgroups are rejected regardless of UID. It reads `/proc/<pid>/cmdline` only for a process whose observed proc-directory UID matches the worker UID. Foreign owners retain their PID, UID, memory, start-time, and cgroup observation; their command line is not opened. The process-directory identity is still checked before and after inspection.

The five-arm F0, seed-1001 CPU scoring contract, unit, resource limits, receipt gates, staged files, read-only export inputs, result destination, and unbound export acceptance/stop pins remain as in V1/V2. No B query, staging, admission, score execution, or quality result is authorized by this local packet. The root-owned independent source review must inspect these V3 bytes before any rendering or admission.

`SCORE_ADMISSION_BODY_TEMPLATE_V3.py` holds the conditional body, `render_score_admission_v3.py` is fail-closed while root receipt pins are unset, and `SCORE_ADMISSION_BINDING_PLAN_V3.json` records the V3 inspection scope. `test_score_admission_v3_owner_scope.py` exercises synthetic process metadata only; it does not inspect `/proc`, query a GPU, import WNN/scorer code, or access any model/data/cache/prediction payload.
