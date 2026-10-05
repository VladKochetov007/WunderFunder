# Independent E482 v2 CLI and packet review

Disposition: **GO for the controller's private v2 submission and bounded probe gate.** This is a local source/packet review only. I made no Kaggle calls and did not launch a GPU or fit.

The v1 failure is reproduced by inspection of its dispatcher: `parse_known_args()` consumed `--output` in the outer parser, then the `--worker` branch replaced `sys.argv` with only the unknown arguments before calling `worker_main()`. The worker parser consequently received no required output path and exited with argparse status 2. V2 fixes this at dispatch: `orchestrator_main()` checks the original argument list for `--worker` before constructing the outer parser, removes only that sentinel, and passes the remaining arguments intact to `worker_main()`. Ordinary kernel invocation without `--worker` still parses the required output path and calls `run_remote()`.

The child-process regression exercises the production script entrypoint and argument vector, not only a parser function: it launches `e482_remote.py --worker --phase {probe,fit} --optimizer {adamw,muon_adamw} --output <path> --parse-only`. All four cases returned the requested phase, optimizer, and exact output path. The parse-only branch exits before `run_probe()` or `run_fits()`, so the regression does not read the dataset or initialize CUDA. The generated kernel separately invokes the orchestrator with `--output /kaggle/working/e482-results`; inspection confirms this is the correct outer CLI path.

I independently ran the complete packet tests under a 1-CPU, 2-GiB, no-swap systemd scope:

```bash
systemd-run --user --scope --collect -p CPUQuota=100% -p MemoryMax=2147483648 -p MemorySwapMax=0 \
  .venv/bin/python research/campaign_2026-10-05/activation_t4_replication_v2/test_e482_packet.py
```

All five tests passed, including four subprocess CLI cases and regenerated bundle/extraction/mount checks. `ruff format --check`, `ruff check`, and `py_compile` also passed for the frozen worker, builder, tests, and generated kernel.

I confirmed these frozen source hashes:

| File | SHA256 |
|---|---|
| Worker | `ad8bc19fd46a1394b7f9346eeb7451e14f7e4a6fc6bb643fdb6584d9566e4982` |
| Builder | `9acb6bd62b43f6cae66a09847ed9d2ff8dc37e6d3f12e009a23ede4ccd6fed54` |
| Tests | `94d1049c3bdf3d2c9bd23095a1e13c5352c1b6fcf441b01334267d4f8f38bc2c` |

The generated packet matches its build receipt: archive SHA256 `6cf83debabd43ef46365d224492c31f3c33aa579f07190ae387dc5384ccd4245`, manifest `64796bd729527c70a0be5402979be1bd5e5d0c525207cc57fb2fb0a3c38aa473`, kernel bootstrap `2d630c8c8d6d43d23c0053bc4c3f4d6c1a22ce293510754ac14312529fa2cf0c`, and kernel metadata `8935ac6dfe870891c8c6d2fc4308e0dac2bdf17c697528d5ed3ffe7e39894eb0`. The archive has 52 unique members and excludes `valid.parquet`; the manifest binds that dataset as an external input and marks the bundle private. Metadata has the requested kernel ID `vladkochetov07/wnn-e482-t4-sixfit-20261005-v2-runner`, title `wnn-e482-t4-sixfit-20261005-v2-runner`, GPU enabled, internet disabled, and both intended dataset sources.

This clears the dispatch defect and packet-integrity gate for the new private v2 kernel. The first provider run still needs to pass its registered two-T4 probe/parity and resource gates before its six fits; this review makes no claim about provider runtime or scientific results.
