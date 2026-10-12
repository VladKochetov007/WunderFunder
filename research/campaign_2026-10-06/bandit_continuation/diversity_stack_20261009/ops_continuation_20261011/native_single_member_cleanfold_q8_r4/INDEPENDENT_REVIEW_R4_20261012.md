# Independent static review: native Q8 singleton controls R4

Reviewed packet commit: `f6fe3a725bb9f845c446dc75aa852aaa3512a9d3`  
Reviewer scope: static packet/source review only; no model inference, prediction, scoring, training, package installation, provider action, or MAIN edits.

## Findings

- **PASS — MAIN and producer/source bindings.** R4 pins MAIN `INPUT_PREFLIGHT_MAIN_R1.json` SHA-256 `e2b7bab342ee35dc1122f88909f06d9f980188d80a6471341a9ec9104352264c` and R3 contract SHA-256 `312b57d7cff1386be2cb727a30ce66133588db1804fb50180ef96768374eea12`. The declared validator independently rehashed all 20 R4 source bindings and 10 R3 producer receipts in MAIN. R4 keeps the R3 admission path/schema, contract, output root, result filename and result schema; no R2 output path is substituted.
- **PASS — actual data footer fingerprint, metadata-only.** A bounded stdlib read of MAIN `datasets/valid.parquet` confirmed 5,615,509,128 bytes, 10,686,926 footer bytes, `PAR1` magic, footer-plus-trailer SHA-256 `4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b`, inode `50919096`, and mtime_ns `1577826000000000000`, matching the R3 preflight. Only the footer/trailer was read; this is not a full-file or row-data hash.
- **PASS — point-of-use guard by code inspection.** R4 opens `valid.parquet` once, checks the footer on that descriptor, passes that same open stream to `pyarrow.ParquetFile`, rechecks the footer on that descriptor, checks metadata, and then reads row groups through the same `ParquetFile`. The focused mutation test covers a synthetic footer only; the actual runner path was not executed.
- **PASS — NPY size accounting.** The plan includes a 128-byte header for each `.npy`: 74,880,128 bytes per F0 file and 73,120,128 per F1 file (296,000,512 bytes total). `open_memmap` writes the planned shapes; inventory checks exact sizes. The dedicated accounting test passed.
- **PASS — CPU/memory/swap/cgroup gate, with a wall-cap caveat.** The runner checks campaign-slice membership, actual cgroup CPU quota at or below one core, memory max at or below 1.5 GB, zero swap, current memory, one-thread environment, hidden GPU and fixed wrapper variables. Six standard-library tests passed on CPython 3.11.16.
- **BLOCKED — runtime gates remain unsatisfied.** The validator reports no `ROOT_ADMISSION_R3.json`, no `CALLBACK_FIXTURE_SINGLE_R3_PY311.json`, and missing `/usr/bin/python3.11` packages `numpy`, `pyarrow`, and `torch`. The fixture is required to establish actual single-member Q8 callback/reset parity. The recorded status is `PASS_STATIC_PACKET_BINDINGS_BLOCKED_RUNTIME_GATES`; it is not execution validation.
- **CAVEAT — wall limit begins outside the runner’s own timer.** `run_full_controls` starts its elapsed-time timer after `build_predictors`, so its 1,800-second check excludes model/weight/native predictor setup. The README’s `scripts/limited` launch sets systemd `RuntimeMaxSec=1800s`, but R4 does not hash-pin that launcher and the runner verifies the environment value rather than the active systemd timeout. Before any runtime admission, root should bind/verify the launcher or otherwise establish a hard 1,800-second process limit that includes startup.

## Verification boundary

Ran the six declared stdlib gate tests (`/usr/bin/python3.11 -m unittest ...test_single_controls_r4 -v`): all passed. Ran `validate_preparation_r4.py` against the specified MAIN and R4 worktree: `PASS_STATIC_PACKET_BINDINGS_BLOCKED_RUNTIME_GATES`, with 20 source bindings and 10 producer receipts rehashed. Independently checked the real Parquet footer/trailer as stated above. No row groups, model callbacks, prediction arrays, or scores were produced.

**Verdict:** safe to integrate only as a frozen `PREPARATION` packet with these limits recorded. This review does not create or imply root admission, authorize execution, or support a score/public claim.
