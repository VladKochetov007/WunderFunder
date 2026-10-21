# Research admission resume review — 2026-10-18

## Decision and authority

The durable October 15 `STOP_ADMISSION.json` records a user-requested pause and
allows only the already-authorized export and grade to drain. The user's later
instruction explicitly resumes autonomous local research until an explicit
pause, including experiments on causal online learning, custom features,
previous-competition techniques, and a B300 draft when a substantial direction
is found. This later instruction supersedes the old pause; the old receipt is
preserved as history rather than edited.

## Drain and resource review

Before reopening, a bounded read of the durable attempt ledger found 535
attempts and no active or unresolved jobs. The recorded outcomes comprise 435
finished, 56 not-started, 43 failed, and one lost attempt. The campaign slice
was observed at two CPU cores, 18 GB `MemoryHigh`, 20 GB `MemoryMax`, and zero
swap. It held 6.25 GB at the latest snapshot. The single local GPU was an RTX
3060 Laptop with 75 MiB used of 6 GiB; no campaign fit or replay process was
running. Unrelated host work was left untouched.

The online-feature contract remains binding: two aggregate CPU cores, a 5 GB
task reservation, one GPU worker, 5 GB observed VRAM ceiling, 17 GB conservative
parent commitment, physical scratch under
`/home/vlad/development/wunder/research_tmp`, no swap, and no new paid compute.
The next planned action is CPU-only, inference-only, uses exposed F2 for
cross-fold robustness diagnostics, and does not tune blend weights or claim
clean validation. No B300 execution, Kaggle mutation, or external submission is
authorized by this resume.

## Admission action

The resume command archives the original stop barrier with its reason and
reopen timestamp. The original `STOP_ADMISSION.json` is not overwritten. The
campaign's active status and later receipts remain the source of truth for
subsequent admissions.
