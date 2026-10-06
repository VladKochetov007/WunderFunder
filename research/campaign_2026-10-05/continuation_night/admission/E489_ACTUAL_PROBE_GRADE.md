# Independent actual E489 v2 probe grading

Verdict: no full-fit admission. The actual frozen v2 probe failed its original
requirement that every native/custom parity update exercise clipping. This is
a failed prerequisite, not a predictive comparison or OOM result.

Evidence: runner/attempts/night-e489-frozen-v2-probe.json and corresponding
runner/logs/night-e489-frozen-v2-probe.log. The monitor reports exit1,
job-scoped failure, 19.665757676004432 charged seconds, 2,429,419,520-byte
peak charged RAM and 421,527,552-byte observed VRAM. Worker and parent OOM
counters did not increase; teardown is verified. Prelaunch MemAvailable was
35,862,728,704 bytes for the 3,000,000,000-byte incremental reservation.

The traceback ends at gpu_probe_fp32_duration_v2.py:202:
`E489 native alpha-zero parity update must activate clipping`.
An earlier parity call reached the return statement's scalar-conversion warning,
but no durable per-seed report exists because the probe only writes its report
after all checks. Do not infer which seed or norm failed from that warning.
No six-cell timing report or full-fit eligibility result was produced.

The original tolerances, seeds and clipping-exercise gate remain unchanged.
A separately preregistered diagnostic could report per-seed native/custom
preclip norms and raw/clipped update deltas to distinguish fixture coverage
from a numerical discrepancy. That would not retroactively pass this E489
attempt, and requires a new identity and controller admission. No automatic
retry, fit, scoring or CUDA action was performed by this independent grader.
