# Memory reporting reconciliation

The final getrusage ru_maxrss receipt was347,136KiB. The maximum across every preserved /proc VmRSS/VmHWM snapshot and getrusage observation is347,560KiB. Use347,560KiB as the overall observed peak. The original outcome-summary.json and outcome.md remain preserved; outcome-summary-amended.json changes only this reporting scope and retains the original value explicitly. All runtime qualification gates remain unchanged, including the16GiB bound. This correction reads existing receipt metadata only; no rerun or empirical-array read occurred.
