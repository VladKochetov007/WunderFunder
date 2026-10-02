# Fixed midpoint final delivery — source preparation

The final member is `oct02_wm_final_s3h2/model_ep1.pt`, constructed once with CPU FP32 `(old + warm) * 0.5` for all 10 learned ordinary-GRU tensors. `mean` and `std` must be byte-identical and are copied unchanged. This is parameter interpolation, not prediction averaging or another fit. Final seed 3 and coefficient 0.5 were fixed by M's contract; no seed or coefficient search is added.

`commands.json` contains exactly two final warm fits, copied from the previously reviewed D recipe: `oct02_ow_final_s3` uses one epoch, LR .0005, original full-data `cont_sh256_final_s3@ep3`, same teachers/EMA/sham constructor RNG and preserved normalization; `oct02_ow_final_s3h2` uses two ordinary H2 epochs, own parent ep0, frozen input and GRU. All 103 arguments were independently reconstructed. Existing legacy fin256_s3/s4 teacher cache bytes are retained; their original producer epochs remain unknown. No stronger teacher provenance is claimed.

Root allocates and starts training through `run_queue.py --allocation <training-allocation.json>` (or its root-owned launcher). No command here authorizes execution. D, W and M files are untouched. After completed fits and the separately allocated native OOF gate, root may invoke:

```text
.venv/bin/python research/iterations/2026-10-02-midpoint-delivery/finish_delivery.py --config research/iterations/2026-10-02-midpoint-delivery/finish-config.json --config-sha256 <SHA>
```

The config must bind the actual training/native allocations, deadline and exact sources with `execution_authorized=true`. The once-only finisher applies the prospective native gate, then runs `final_midpoint.py`, `package_student.py`, clean CP311 `runtime/check_package.py`, and `runtime/prepare_runtime.py` on CPU8/quota1CPU/4GiB. It stops at `READY_RUNTIME_PLAN`; it never allocates or starts the runtime audit. Existing paths, failure or deadline exhaustion stop the pipeline without retry. A 3GiB disk floor is retained.

Only `weights_0.npz` changes in the qualified student fixed4 ZIP. The wrapper, three native binaries, three partner weight files and other entries are copied byte-for-byte; ensemble quarters remain unchanged. Constructor and packager independently reconstruct the final midpoint and validate exact warm completion/checkpoint lineage. Package checks cover deterministic native callbacks, reset/prefix and ordered ensemble arithmetic, not GPU/native score equivalence; the latter belongs to `native-quality`.

The runtime plan is initially unallocated. Root supplies a new authorization to `runtime/start_runtime.py --allocation <JSON> --allocation-sha256 <SHA> --execute`. Exactly 39.4M complete real callbacks, 1,970 resets and 39,204,970 emitted outputs run through the actual wrapper, including constructor, I/O and pre/post input hashing. Qualification requires actual total time <=3,600s; hard cap3,900s, CPU8/SMT48, one compute CPU,16GiB/swap0, no retries. Other ISA/hardware timing remains outside this run.

`collect_delivery.py --phase ready|runtime --deadline-utc <ISO> --remote-cpu <idle CPU>` is an optional explicitly invoked local collector after closed scopes. It preserves all three final run directories plus the campaign; no blanket deletion or model decoding. It can losslessly gzip large prediction caches.

M's observed seed3 gain and positive seed4 direction are conditional on repeatedly exposed folds; seed4's interval crosses zero. Native qualification and successful runtime are still required before promotion. This preparation does not claim a hidden-test gain.
