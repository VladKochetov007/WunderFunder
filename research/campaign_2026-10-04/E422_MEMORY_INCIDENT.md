# E422 gain cache memory incident

The first4GiBCPU helper was killed by its child memory limit. Systemd reported `Result=oom-kill`, `ExecMainStatus=9`, and exactly4,294,967,296bytes peak; the monitor also died, so its last sample undercounts the terminal incident. The parent aggregate remained within its38.65GB cap. See `checks/e422_oom_systemd_evidence.txt`.

After selecting anchor rows, the extractor retained NumPy views into each full20k×512 hidden-state array (~40MB per sequence). It accumulated these arrays across sequences. The corrected selection copies the small anchor subset. Four CPU checks passed under2GiB, including a weak-reference lifetime regression that proves previous full-state arrays are released during later sequence extraction. Source closure from before the fix is archived; the failed attempt is immutable.

The resource guard stopped an uncheckpointed local continuation during stale-job reconciliation. Neither interrupted continuation produced an epoch checkpoint; both have separate identities and charged runtime. Fresh identical FP16 continuation is running. The gain-cache production retry remains held until its production memory can be verified without risking another uncheckpointed fit. Increasing its memory cap without fixing retention is not the remedy.
