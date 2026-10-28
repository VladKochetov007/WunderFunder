# First TL3 receiving evidence

The raw records below are copied byte-for-byte from the already-owned L4 runtime
`/data2/lc-epanov_s/tabdata/runtime/launch-20261028/tl3-trainonly-20261028-r1`.
They contain no model/data payloads or credentials.

- `profile-r1.owner.json` and `logs/fin256_s3-profile-r1.log`: first attempt
  stopped before model updates at contradictory historical/chronological CLI
  checkpoint gates. It is excluded from throughput and scientific evidence.
- `source-r3.identity.json`: copied R9 base with Git-LF exports of changed
  files, source commit ec3f1a666 and per-Python-file SHA256. Earlier source-r2
  patch application failed because the base had CRLF bytes; no model updates
  or fit were admitted under that failed source.
- `logs/receiving-r3-tests.log`:9 receiving CPU passes on Torch2.8/Linux for
  tiedlead causality/schema, chronological admission and input allocation.
- `profile-r3.owner.json` and `logs/fin256_s3-profile-r3.log`: real-data strict
  FP32 GRU256x2 batch32/chunk2000/EMA1/aux profile;40 finite, completed updates,
  zero skips;17.21seq/s,1.85897s per full sequence batch. No quality claim.
- `fin256_s3.launch.json`: exact command/environment/source and split hashes,
  process/start-tick ownership and six-hour bound for the fresh four-epoch fit.
  Timeout parent4125363, trainer child4125364 at launch; it survives SSH logout.

Actual GPU0 allocation was about4.2GiB with the fit active. A co-resident profile
was finite but yielded only9.11seq/s per profiled fit; no second scientific fit
was started on that card. All seven original GALAM trainers remain unchanged.
The first teacher completed4 epochs with12880/12880 optimizer updates and no
skips/nonfinite gradients. [First-teacher result](FIRST_TEACHER_PROGRESS_R1.json)
preserves the actual VALID records; fixed ep3 WP0.6882596674 is development
evidence, not the complete TL3 endpoint.

All4 first-teacher train/valid WMSE and WP points were backfilled from SHA-bound
owned checkpoints and verified through the existing dashboard API, run
`tl3c_fin256_s3`. The comparable objective is WMSE; the fitting probe uses32 train
groups/all-after-warmup, while valid uses all1873 groups/official scored mask.
The remaining finite TL3 chain and final blend score are pending.
