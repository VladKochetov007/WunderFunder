# E492 source review and raw fixture result

Verdict: **GO for the registered fit-only descriptive EDA**. This review is
bound to `analyze.py` SHA-256
`36c82e18d4a9b15b613f629ce78ab29d7b96c1e2b4f0fd2dcd84a063ffd57388` and
`DESIGN.md` SHA-256
`9686b61209d61e53c1d3ecbebf27a3be5a28376f9ebadb7539d73bcc05d8092c`.
Neither file was modified by this independent fixture work.

The source now rejects a nonconsecutive `step_in_seq` against exactly
`arange(20,000)` and requires both row masks to be finite binary values. It
derives bid, ask, a6 and target transitions over the complete raw group before
selecting current rows, then applies `need_prediction & step >= 99`; the scored
population is an additional mask. It streams only the 24 fit-role row groups,
selected raw fields and no evaluation groups, with Arrow threads disabled.
The `bid` and `ask` clocks include simultaneous changes in their respective
coordinate; the separately named OR and AND clocks include both and only both,
respectively. This matches the clarified design.

`raw_fixtures.py` is an independent count implementation. Its full-size hand
fixture includes the first eligible target transition, bid-only, ask-only and
simultaneous changes, an a6 transition, and a row removed only from the scored
population. It rejects a changed sequence step, a mask value of 2 and NaN. The
fixture test compares every output row to `analyze.summarize_group`; all 6 E492
tests passed (combined with 5 E489 math tests: 11 total, 1.18 seconds, one CPU,
CUDA disabled).

The sealed EDA attempt finished successfully in 5.664 charged seconds with
507,981,824 peak host bytes, no GPU use and no worker OOM. The independent
direct-parquet count reproduction is prepared in `reproduce_counts.py` with an
unsealed one-CPU/two-GiB spec at `reproduction.draft.json`; it has not yet been
admitted or run. The EDA remains descriptive evidence on an exposed fit panel.
It establishes neither conditional predictability nor model gain.
