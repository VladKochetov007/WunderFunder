# Remote source metadata

This directory contains small text files read from public GitHub branches through the contents API, plus the exact API JSON responses. It is evidence metadata for the parent PR audit; it does not contain the linked tarball payloads.

- `PR32_ENGINEERING_MANIFEST.json` and `PR32_ENGINEERING_MANIFEST_API.json` describe the newest PR #32 engineering archive on `experiment/pr32-b300-evidence-20261024`.
- `PR40_*.API.json` and their decoded text files preserve #40's throughput report, evidence manifest, execution README, and resource estimate at merged head `d0c6a4a1b90f1f7cebf06c24c815811ed78ae59d`.

The newest archive is explicitly engineering-only and records zero scientific fits. Root later recovered its tarball from the fetched PR32 branch and verified all 546 members and hashes; see [REMOTE_ARTIFACT_RECOVERY_R1.json](../../REMOTE_ARTIFACT_RECOVERY_R1.json). That archive does not contain trained weights, checkpoints or prediction payloads and was not executed. The exact GitHub directory listing and archive size/hash are linked from [the PR audit report](../REPORT.md).
