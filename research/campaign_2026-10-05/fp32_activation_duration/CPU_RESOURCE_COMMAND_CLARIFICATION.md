# E489 CPU resource command clarification

`cpu_preflight_v1/fit_manifest.json` is a valid fit-only source/data receipt,
but the command block in the initial `SOURCE_NOTE.md` did not invoke the
repository's `scripts/limited` systemd wrapper. The `WNN_MEM_MAX` environment
variable by itself is only a wrapper setting; when the wrapper is not used it
does not impose a 2 GiB cgroup memory ceiling. Therefore that displayed
command is not evidence of a hard 1 CPU / 2 GiB run. The v1 output and source
hashes are preserved, and the v1 run must be described as uncapped by that
wrapper.

The root-limited rerun completed successfully as
`cpu_preflight_root_limited_v2/fit_manifest.json`, SHA-256
`3bcd0909c800f9c6f133ad55a23b9490cf85ea64ed8b4deec8151e73702bed6a`. It is
byte-identical to v1 because the receipt is deterministic. The bounded rerun
passed all six fit-only one-update cells and records `test_rows_loaded: false`.
The command below ran through `scripts/limited` with one CPU, `MemoryMax=2G`,
and swap disabled. No actor, probe, or synthetic-test source was changed for
this correction.

The canonical form of the command is:

```bash
WNN_CPU_PCT=4 WNN_CPU_QUOTA=100% WNN_MEM_MAX=2G WNN_THREADS=1 \
  WNN_RUNTIME_MAX=300 scripts/limited .venv/bin/python \
  research/campaign_2026-10-05/fp32_activation_duration/fit_fp32_duration.py \
  --preflight-real --device cpu \
  --output research/campaign_2026-10-05/fp32_activation_duration/cpu_preflight_root_limited_v2
```
