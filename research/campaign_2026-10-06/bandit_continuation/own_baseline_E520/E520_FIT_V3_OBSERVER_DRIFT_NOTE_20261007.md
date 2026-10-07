# FitV3 observer compatibility discovered from V7 terminal evidence

Root inspected frozen `throughput_optimization_E522/e520_fit_digest_handoff_candidate_v1/fit_worker.py` after the actual V7 profile observer failure. Lines111–114 require `matched_pid` and non-null process allocation outside idle phase; lines1156–1157 install torch and synchronously check before fit. This is the same unavailable-query risk repaired in the independently reviewed V8 profile worker.

The V3 source/digest/recipe review remains preserved and valid for its reviewed scope; it does not establish observer compatibility on the actual Kaggle device. No V3 fit has run, no science result and no new negative reward. Root directs cloud operator to prepare an identified observation-only V4 fit successor with exact independent source/transport review before fit. Preserve recipe/cache targets/seed order/checkpoint policy/grader; retain all RAM/VRAM/headroom caps. V8 profile admission is independent and continues.
