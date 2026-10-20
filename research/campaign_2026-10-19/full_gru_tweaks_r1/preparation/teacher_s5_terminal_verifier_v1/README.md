# S5 terminal verifier packet

This packet binds the completed seed-5 strict-FP32 teacher run `fg19_t3_h384x2_e3_s5_v1`, native unit `computer-vision-homework-fg19-teacher-s5-v1.service`, invocation `4317408cdc69442c8438c74aaf5809eb`, and captured terminal evidence `e284787effbb9ad1a64be763c6bc5eb9c805bc1715c37b08aa032b2673b433b4` (raw health capture SHA-256 `9bdc3861961d19ed6454007376bb69f94f4429eb2eba5dce99497a5335d98421`). The evidence records the explicit native successful exit: active/exited, MainPID 0, ExecMainCode 1, and ExecMainStatus 0.

[The S5 checker](../teacher_s5_terminal_check_v1.py) carries over the S4 scientific integrity checks and restricted CPU `weights_only=True` loading with its narrow `TorchVersion` allowlist. Its identity pins seed 5, the S5 run/unit/invocation and frozen recipe/admission/launch/acceptance bytes, and the actual S5 launch-acceptance field layout. Canonical seed-5 train order, manifest-order fit-valid membership, evaluation order, source pins, tensor/layout checks, optimizer counters, and artifact caps remain bound.

[The local packet generator](../admit_teacher_s5_terminal_verifier_v1.py) wrote the [transport body](REMOTE_SUBMISSION.py) and [preparation receipt](PREPARATION.json). Before one bounded checker run, the body rechecks the exact native exit and the source, split, registered data identities, output absence, parent cgroup, persistent disk/free space, and memory headroom. It stages only the pinned checker/helper and JSON metadata. The verifier unit is `computer-vision-homework-fg19-teacher-s5-verify-v1.service`, capped at 50% CPU, 1.4 GB `MemoryHigh`, 1.5 GB `MemoryMax`, no swap, and 600 seconds. It uses the existing literal `/home/grinchenkoeg/WunderFunder/.venv/bin/python` and installs nothing.

The packet contains checker/helper code and frozen JSON metadata only; it contains no checkpoint, optimizer, dataset, or array payload. It remains prepared locally and has not been transmitted or launched. The [checks receipt](CHECKS.json) records the three focused identity tests, Ruff, and a cold checker/helper import with data-reader and CUDA sentinels clear.

After reviewing the packet, the root operator can run this exact command from the maintenance workspace to send the single prepared body to B:

```sh
ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 beleriand python3 - < /Users/nb/Documents/wunder/WunderFunder-maintenance/research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/teacher_s5_terminal_verifier_v1/REMOTE_SUBMISSION.py
```

Frozen hashes:

- Checker: `cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5`
- Local packet generator: `ef67f98ce81a370fd76ea520d673baddb4c46d4efb5ff455c9134c5656c0ed7d`
- Transport body: `bd08f04328fb95eb81429e5b64ce7428fa078b5639d35e70ac16b4311a6a4abc`
- Preparation receipt: `33ecf08a49c79dbef27e00fe046c3ada4e9bd7c2b6b38e0b12093e0a5ad5b0f2`
- Focused checks receipt: `185554d7df000db9660092d8deead5906beeec705ab3fea6c34fc84d75d6aade`
