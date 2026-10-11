# CPU submission runtime review

## Result

The exact two-member H390/H388 Q8VNNI package is the fastest currently measured
candidate in this comparison. The Rust LLVM PGO build preserved every measured
prediction, passed the Python 3.11 import/callback panel, stayed within the
Debian Bookworm GLIBC symbol ceiling, and changed only the v3 native library in
the ZIP. It did not produce a repeatable callback speedup, so I do not recommend
the PGO ZIP over the unchanged package.

This is an inference-capacity result, not a quality grade. Both members of the
H390/H388 package are full-data trained. The fold-1 prototype results are
exposed evaluation evidence and are not a score for this exact full-data ZIP.
The local host is an Intel i9-11900H, not the Kaggle AMD EPYC 9224, so its timing
does not establish portal completion time.

## Fixed Python 3.11 callback panels

Every panel used one CPU, ten complete 20,000-row sequences from the same
validated-parquet fixture, two state-reset passes, and the exact `predict`
callback loaded from the measured ZIP. Each model had to report Q8VNNI and
native level v3; outputs had to be finite and bitwise identical after reset.
The panels time callbacks, including warm-up rows, while excluding parquet
decode and reset. The panel includes 400,000 calls per package; each sequence
requested predictions on 19,901 of its 20,000 rows.

| Package | Members | ZIP bytes | Measured callback median |
|---|---:|---:|---:|
| Historical Q8 pair, portal score 0.6668 | 2 | 9,544,031 | 52.74 µs/row |
| Historical Q8 single, portal score 0.6649 | 1 | 5,419,442 | 21.77 µs/row |
| H390 student + H388 teacher, unchanged Q8 build | 2 | 12,914,019 | 74.84 µs/row initially; 77.09 and 77.78 µs/row on paired runs |
| Same two members, local PGO build | 2 | 12,905,551 | 77.20 and 77.95 µs/row on paired runs |
| H388 s4/s5 teacher pair | 2 | 17,272,496 | 104.48 µs/row |
| H388 s3/s4/s5 trio (the uploaded ZIP) | 3 | 13,735,320 | 158.62 µs/row |

The uploaded trio's portal result is right-censored at a reported 61m56s timeout
with 647 MB. That is a provider timeout, not a completed runtime or evidence
against model quality. The two-member H390/H388 package has not been uploaded;
its local callback panel is faster than the trio, but a local Intel timing does
not guarantee it will finish within the portal's 60-minute cap. Prior portal
times for the unrelated historical pair and single were 27m01s and 15m41s.

## PGO and package checks

Two interleaved, same-fixture baseline/PGO comparisons measured 77.7796 versus
77.9534 µs/row (+0.223%) and 77.0885 versus 77.1964 µs/row (+0.140%). A separate
unpaired run appeared faster under different host load; it is not evidence of a
PGO gain. The instrumented build used 400,000 exact candidate callbacks, and
per-sequence output hashes matched the baseline. The final PGO ZIP differs from
the baseline in exactly `v3/wnn_native.abi3.so`; both weights, the solution,
precision declaration and blend are byte-identical. The candidate is 12.91 MB,
below the 20 MB package limit, but its small size reduction does not justify
choosing it.

The native predictor reports 1,071,552 streamed-weight bytes for the 256-wide
student and 2,294,976 bytes for the H384 teacher, plus 2,048/3,072 recurrent
state bytes and 11,864/17,368 scratch bytes respectively. These are the shipped
runtime's per-member storage counters, not a Kaggle RSS estimate. On the local
Python 3.11 panels the cgroup peak was about 231 MB for the two-member package.

`readelf` on both exact ZIP libraries reports the same dynamic dependencies
(`libgcc_s.so.1`, `libm.so.6`, `libc.so.6`, `ld-linux-x86-64.so.2`) and maximum
GLIBC symbol version 2.34, below the Debian Bookworm 2.36 ceiling. The PGO
library hash was matched to its ZIP member, and Python 3.11.16 loaded the
candidate in the exact callback panel.

Skipping the stateless output head on `need_prediction=false` rows could affect
at most 0.5% of these panel's callback rows. That opportunity is too small to
address the observed timeout, so no such source change was made. No Zen4-specific
kernel optimization was implemented or claimed; the timing host is Intel and
does not validate an EPYC-tuned binary.

## Evidence

`CPU_SUBMISSION_FIX_EVIDENCE_R3.json` binds the package/build receipts, callback
panels, per-sequence output hashes, source and data hashes, member-storage
counters, exact library hashes, ELF requirements, and the user-provided portal
timeout receipt. `audit_cpu_submission_r1.py` regenerates it with create-only
semantics. The ZIPs remain untracked runtime artifacts; the committed record
contains no model binaries.
