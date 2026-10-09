# E520 profile v4 portable-manifest repair candidate

This directory is a code-only, profile-only candidate for the v3 split-loader failure. It does not modify packets v1–v3 or the frozen E520 fit recipe. No packet was built and no provider, fit, prediction export, or scoring action was run.

The v3 execution had already matched the mounted parquet, asset, fold, registry, and target-cache hashes before `load_campaign_split` rejected the relocated `cache/chain_folds.npz` mtime. The pinned loader supports relocated inputs through `WNN_PORTABLE_DATA_MANIFEST` and `src/wnn/portability.py`, which verifies registry-bound content hashes. The candidate now removes any inherited manifest environment value at preflight start; after every mounted input identity check and staged symlink is complete, it rechecks the exact staged portable registry hash, atomically records the expected and observed path/hash, and only then binds `WNN_PORTABLE_DATA_MANIFEST` to `source_root / PORTABLE_RELATIVE`. The unchanged pinned loader then applies its registry/content checks to every source and performs the same split-membership validation.

The page-cache-safe full-file hashing, expected identities, fold counts, corrupt-group exclusion, target-index checks, profile-only trainer command, runtime/resource caps, and abort behavior are inherited unchanged. Any missing or mismatched manifest is persisted before failure. The worker uses fresh `e520_source_v4` and `e520_profile_v4` directories and aborts if either already exists. The exact Kaggle metadata constants are title `E520 Profile V4` and id `vladkochetov07/e520-profile-v4`.

The review-gated builder emits a self-contained `profile_kernel_bundle.py` as the Kaggle `code_file`. Its deterministic ZIP embeds the reviewed packet members, binds the archive and every member by SHA-256, rejects unsafe paths, and extracts into fresh `/kaggle/working/e520_profile_packet_v4` before invoking the worker. This preserves the v3 transport pattern where Kaggle starts only the declared script. The bundle is generated only after independent review; the actual built bundle still needs a separate transport hash check before an operator push.

## Focused local checks

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/pytest -q \
  research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/portable_manifest_repair_candidate_v1/tests
```

The tests cover fail-closed environment binding, durable mismatch receipts, registry content verification after mtime relocation, deterministic safe bundle construction, the exact v4 Kaggle slug/title, and the existing profile-only resource and export guards. They use synthetic small files and do not load parquet or model arrays.

## Independent review targets

Generate the exact source map without creating a packet or calling Kaggle:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/portable_manifest_repair_candidate_v1/prepare_profile_packet.py \
  --review-targets-only --repo-root . \
  --plan research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/portable_manifest_repair_candidate_v1/E520_PACKET_PLAN.json
```

The builder's `--build` mode remains gated on an exact independent review receipt and is intentionally unused here. A later v4 packet build must use a new empty packet directory; any provider push/retry remains the cloud operator's separate, admitted action.
