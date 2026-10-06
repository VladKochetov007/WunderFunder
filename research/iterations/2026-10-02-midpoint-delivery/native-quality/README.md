# Native quality of the fixed seed3 midpoint

Source-only preparation. Root may invoke allocate.py, then start.py, only after the fixed midpoint gate is verified: seed3 pooled improvement >=.0001 versus incumbent, seed4 >0 versus its own original, and all four seed/fold deltas >0. Production seed3 is fixed, with no best-seed choice. This native check itself does not authorize final fitting or deployment.

Reuse the qualified VNNI CP311 binary without source or numerical changes. Export constructed oct02_wm_s3h2 fold0/1 ep1 plus unchanged on_ctl128s3, on_ctl128s4, on_ctlc500 ep1, equal quarters. Bind the completed midpoint CPU preparation, arithmetic/normalization proof and construction receipts before export and both prediction passes. Primary reference remains CURRENT student_native4; secondary is exact GPU midpoint_seed3_4 from frozen M sufficient statistics, not warm-only W.

Two independent CPU lanes30/35, one core and4GiB each,1800seconds per OOF, no GPU or retries. Input is full raw causal sequence history; no target/mask decode in prediction. Outputs are new cache/pred_midpoint_quant_fF_valid.npy and .json receipts, full1873-row indexing with unselected NaNs. The unchanged evaluator convention scores the same860 pure heldout sequences. Export/rowmap data bindings and native kernel SHA remain mandatory. Existing D and M outputs are immutable.

Root CLI: `.venv/bin/python research/iterations/2026-10-02-midpoint-delivery/native-quality/allocate.py`, then `.../start.py`. Allocation reads already completed reports and hashes only; it creates allocation.json without starting any work. All runtime preparation, inference and the one fixed metric readout are separate authorized execution.
