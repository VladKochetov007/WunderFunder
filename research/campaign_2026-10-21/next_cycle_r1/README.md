# October 21 next-cycle execution packets

This folder contains the frozen local small-screen plans attached to the
accepted October 21 cycle allocation. The root [cycle plan](../../../specs/plans/2026-10-21-next-local-kaggle-cycle.md)
owns resource admission and run ordering; this folder records runnable source
packets and execution evidence only.

- [i1 causal-history screen](i1/README.md): sibling branch
  `experiment/i1-screen-20261021`, commit `b1b680d4a700d353f3e6db4f761f6670bd02fc7f`.
  Its [F2 fit subset](i1/fit_f2_10pct_s1002.json) is included byte-for-byte for
  the dynamics plan and has SHA-256
  `841576ddd469befa98b37d1a567e47462979f298314b2a4cf4a399beb5ddc908`.
- [Future-input dynamics auxiliary screen](dynamics/README.md): four exact
  cold fits, full-fold prediction and score steps, a sustained F0 treatment
  profile plan, and a fixed paired full-fold grader. No profile or fit has run
  from this packet.
