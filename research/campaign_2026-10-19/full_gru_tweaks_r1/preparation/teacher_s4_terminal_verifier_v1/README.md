# S4 terminal verifier packet

This packet prepares one bounded integrity check for the completed seed-4 teacher run. It binds run `fg19_t3_h384x2_e3_s4_v1`, unit `computer-vision-homework-fg19-teacher-s4-v1.service`, invocation `b6d71599e0044970b8641e3e241c40a4`, recipe `c460b098…`, admission `0f68060b…`, launch `f303534d…`, launch acceptance `53291735…`, and captured native terminal evidence `e11c8c01…` whose raw health capture digest is `0ced2733…`.

[The checker source](../teacher_s4_terminal_check_v1.py) carries over the S3 scientific integrity checks and restricted CPU `weights_only=True` loading with its narrow `TorchVersion` allowlist. Its identity changes pin seed 4 and the S4 run, recipe, unit, invocation, and launch acceptance. Canonical seed-4 train ordering, manifest-order fit-valid membership, evaluation ordering, source pins, tensor/layout checks, optimizer counters, and all artifact caps remain bound.

[The local packet generator](../admit_teacher_s4_terminal_verifier_v1.py) wrote [the transport body](REMOTE_SUBMISSION.py) and [preparation receipt](PREPARATION.json). The body rechecks the exact current native exit and the source, split, registered data paths, output absence, campaign ownership, empty parent cgroup, disk type/free space, and memory headroom before staging the checker and metadata. Its verifier unit is `computer-vision-homework-fg19-teacher-s4-verify-v1.service`, capped at 50% CPU, 1.4 GB `MemoryHigh`, 1.5 GB `MemoryMax`, no swap, and 600 seconds. It references the existing literal `/home/grinchenkoeg/WunderFunder/.venv/bin/python` and installs nothing.

The packet contains checker/helper code and frozen JSON metadata only. It contains no checkpoint, optimizer, dataset, or array payload. The body has not been transmitted or launched, so it reports no B-side verification result. Focused identity tests, Ruff, and cold transitive import/read-sentinel results are recorded in [the checks receipt](CHECKS.json).

After reviewing the receipt and packet, the root operator can run this exact command from the maintenance workspace to send the single prepared body to B:

```sh
ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 beleriand python3 - < /Users/nb/Documents/wunder/WunderFunder-maintenance/research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/teacher_s4_terminal_verifier_v1/REMOTE_SUBMISSION.py
```

The frozen hashes are:

- Checker: `e6891fb6273b0f8b7f37af4f713e0ab2e6e16a13fbaf8ce8275401c2c6d4a0cf`
- Local packet generator: `7b3ab65a49acddc82179dce80f0b53c1e40beb099e7b9c39ab6e85e7dd185b92`
- Transport body: `18332ba6e073fb2b0dad3fe5a0725198984a4e772a5cc3adda203ee2d115ddf3`
- Preparation receipt: `58c7f61ed8be64428f5a87020978bf9d73acceeb79186de1f5a710b5484240bb`
- Focused checks receipt: `f8ed7a9f7aeb46a3e60f5a2b01f7d1d450d210fd44720c364a251a36c8b8e837`
