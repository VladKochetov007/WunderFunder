# R5 conditional-moment decoder screen

[Measured results](RESULTS.md) compare two fixed decoders of the same six fitted moment heads on the unchanged R3 panel: ratio WP0.524107 versus constant-denominator WP0.549662. The primary ratio-minus-constant delta is −0.025555; all32 drop-one contrasts are negative. The constant endpoint's +0.013658 against the stored R3 WMSE recipe is a descriptive secondary lead.

- Scientific record: [pre-code contract](CONTRACT.md), [raw RESULT](records/probe_v1/terminal_capture/RESULT.json), [independent audit](RESULT_AUDIT.json), [source manifest](records/probe_v1/terminal_capture/SOURCE_MANIFEST.json).
- Readiness: [23-pin manifest](PREPARATION_MANIFEST.json), [design review](DESIGN_REVIEW.json), [implementation review](IMPLEMENTATION_REVIEW.json), [tiny test receipt](TEST_RECEIPT.json).
- Execution: [records](records/README.md), [resource reconciliation](RESOURCE_RECONCILIATION.json), [continuation](CONTINUATION.md).
- Frozen implementation: [fit](analysis_fit.py), [score](analysis_score.py), [diagnostics](analysis_diagnostics.py), [report](analysis_report.py), [entry point](run_probe.py), [synthetic tests](test_probe.py).

One analytic CPU fit, alpha100, no CV/grid/resampling. Actual unit exited0/captured/stopped; models/arrays remain on B. The package's staged README was the earlier preparation snapshot; this navigation/result annotation does not change the frozen23 numerical/source pins. No new job is created by these links.

The root RESULT_AUDIT.json is a byte-identical navigation alias of the independent receipt in records/probe_v1/terminal_capture. Its SHA256 is32978779ace161e28b88f4c0f2a579a5275e2ed14973b090d095b8558e5a764d; maximum score reconstruction error2.32e-14. No arrays or model payloads were replayed.
