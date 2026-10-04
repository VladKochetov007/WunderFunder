# Independent source critique: E430 dual-T4 bundle

Reviewed source-only on 2026-10-04. No provider calls, uploads, imports, tests, or scientific jobs were run.

## Findings

No blocking path or artifact-mapping issue found in the reviewed E430 package. The archived plan report string equals each spec's declared artifact path, and the phase runner reads those same reports. The phase runner's spec paths are correctly rooted under `research/campaign_2026-10-04/kaggle/E430_cadence_dualT4_v1/kaggle_specs/` after extraction. The kernel allowlist includes external-mount paths without treating them as archive members. Fold-0 assets map to the E419 fold-0 asset dataset and common `wunder-funder` data dependency; no fold-1 W384 input is declared.

The staged order implements the required barrier: baseline and connected probes run as a pair, then the accumulation probe runs alone; all three must pass before the paired baseline/connected fits, and accumulation fitting waits for that pair to finish. The fit UUID checks are consistent with the runner assigning devices in spec order and using the first device for solo phases. The phase runner checks fixed epoch 3, full four-epoch histories, 373 complete batches per epoch, per-topology attempt counts, reconciled optimizer successes plus AMP skips, EMA count, and required checkpoint/provenance/history/training-state/report artifacts. Partial artifacts can exist after a failure, but the phase report is `failed_closed` and the source does not score or promote them.

The resource interpretation matches the stated remote envelope: 11 GiB allocator target, 12 GiB observed operating limit, 13 GiB sampled stop, and 10 GiB RSS per worker. Its upper model counts both allocated T4s during solo phases, then adds one hour of startup/input-verification margin and the six-GPU-hour reserve. Current W384 completion and fresh provider quota are deliberately not self-asserted by the package. They remain external admission gates: the durable controller should verify W384 terminal state and current quota of at least the recorded 23.34 GPU-hour planning requirement before provider submission, then record the quota snapshot and six-hour reserve. The remote `kaggle_gpu_runner` enforces device/host resource checks but is not a quota or predecessor-status controller.

One operational limitation is acceptable but should remain explicit: the per-worker 3.5-hour timeout covers each worker after launch; repeated source/input verification and phase startup are outside that worker timeout. The one-hour whole-session planning margin covers this in the current estimate, but the provider-level kernel timeout must separately exceed the full roughly 8 h 40 min phase-plus-margin envelope.

## Reviewed identities

- `phase_runner.py`: `3cb7216268d4319503e928e9292c4b46617a1553b8b777e189467766c2a58307`
- `build_e430_bundle.py`: `fd19a3a88bb194dfbe9389bd0ea49512674c11aa4d45f6c4ef0b7682dcc22cf2`
- `bundle.wnn`: `140654e0dd4d9abfe1b966db28574320799da07852deaa743053aae389f80f82`
- `bundle-manifest.json`: `4941ac09baf64b753ac7863a9bba5d902c3db2e1e3aeef19ad36667fe16b4319`
- `source_inventory.json`: `7cd8deb968c429b8b76450b3e5c7341643a2eaef7ed58922e35b246999cd38c9`

This review does not authorize or claim an upload, admission, or run.
