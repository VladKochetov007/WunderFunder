# R2 comparison postprocessor: finite WMSE roundoff

ROOT found a latent failure in the frozen R1 final comparison: it requires
`valid/loss == valid/z/wmse` bitwise. Actual C0 E1 values are
1.0054704748780694 and1.0054704889970745, differing by about1.41e-8 because
the two monitoring reductions use different numerical accumulation. The live
training and prediction source remains5d73bc2b1; no fit is restarted or changed.

The original receiver, plan, comparison script and eventual failure receipts
remain unchanged. The separate `receiving/compare_r2.py` rechecks all six
complete E4/3280 endpoints, full-valid coverage, source/config/ancestry,
model/terminal/producer byte bindings, official metric, exact raw-z WP alias,
and score-file hashes. Only the two finite WMSE representations use
`math.isclose(rel_tol=1e-6, abs_tol=1e-7)`. Real discrepancies and nonfinite
values still fail. R2 pins the exact failed R1 grader hash29314bdac0cd… and
records its own source hash, tolerances, reason and ROOT exit-proof hash.

`receiving/wait_compare_r2.py` is a one-shot CPU observer for this exact cohort.
It waits for receiver PID1818502 /birth158011868 to stop running (including a
zombie exit or PID reuse), then requires all six complete terminal/score files.
It emits the new create-only cohort observation `R/receiver-exit-r1.json`,
with source, plan and authority identities. It calls the grader once, bounded
by900 seconds and the remaining original collection deadline1794197564.8507051.
It has no GPU, fit/resume, signal, SSH or network action.

R2 exclusively creates canonical `R/comparison.json`; it never replaces an
existing file. A pre-existing canonical R1 comparison can be verified as a no-op
when all six scientific contents and checkpoint/score hashes match. Existing
R2 repair metadata, when present, must also match. New worker status/log/source
files live in `R/postprocess-r2`; the new exit proof and canonical comparison
are deliberately cohort-level files at R. No original R1 file is overwritten.

The already running read-only collector/publisher consumes that same canonical
comparison schema and hashes. It continues posting producer-verified grades
first and independently byte-bound final grades only after the comparison
appears. No publisher restart or new credential is needed.

## Checks and staging

Seven focused grader tests pass, including the actual E1 rounding difference,
over-tolerance/nonfinite rejection, wrong checkpoint and no overwrite. Three
waiter tests pass for live/reused/zombie identities, incomplete-fit refusal and
exit-proof preservation. Ruff checks and formatting passed. The independent
source review is retained separately; native admission follows that gate.

Stage the exact frozen R1 `compare.py` beside R2 in the new postprocess folder,
and use the frozen receiver's `admission.py` through PYTHONPATH. The live source
capsule and scientific plan stay unchanged. ROOT records the actual new source
commit, native preflight, owner/birth and status in the execution artifact root.
These scripts are specific to this dated cohort and original clock; they are
not a reusable admission for another experiment.
