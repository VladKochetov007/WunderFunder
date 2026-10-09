# Independent E-515/E-516 pre-score review

Review completed 2026-10-07, before the E-515/E-516 source statistics were
computed. Reviewer inspected only the registrations, implementation and
synthetic tests; no E-514 source data or result outputs were opened.

- **E-515: PASS.** The fixed top-quartile count is `ceil(25%)`. Exact
  disagreement ties at the cutoff are ordered by canonical chain ID, and the
  last fixed count is selected. The helper implements this rule and the test
  confirms its selected chains are unchanged when input order is shuffled.
- **E-516: PASS.** H-206 is a distinct target-wise stacking hypothesis rather
  than a mislabelled H-203 residual-corrector. The preregistration clearly
  records the one-time user-directed reuse of E-514's already exposed pool and
  fold-0 chains, makes no freshness or confirmation claim, and prohibits
  same-pool tuning. The result binds E-516/H-206/arm identity to the verified
  source-manifest and chain-partition digests.

No blocking pre-score findings remain. This review authorizes only the two
frozen analyses as registered; it is not a scientific result or allocation of
follow-up training.
