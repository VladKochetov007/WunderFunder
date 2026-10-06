# E510 raw teacher pair packet

E510 remains unmaterialized. Packet creation requires the actual E508 profile
receipt and terminal output, plus a separate controller review receipt created
outside this worktree. The receipt must approve the E508 profile, bind its two
file hashes, bind the exact E510 kernel/worker/builder/trainer/source hashes, and
approve the fresh provider quota and production budget. The builder rejects
missing, stale, or mismatched evidence before it creates the packet directory.

Once the actual E508 files are available, obtain the exact E510 code identities
with `--review-targets-only`:

```sh
./.venv/bin/python research/campaign_2026-10-06/bandit_continuation/e510_raw_teacher_pair/prepare_packet.py \
  --e508-profile-receipt <actual-E508-profile-result.json> \
  --e508-profile-receipt-sha256 <actual-profile-receipt-sha256> \
  --review-targets-only
```

The preview reports the kernel, worker, builder, trainer, and source-archive
hashes that the separate receipt must bind. It does not create a packet or assert
that the profile or code review has happened. After the controller creates the
review receipt outside this worktree, build with:

```sh
./.venv/bin/python research/campaign_2026-10-06/bandit_continuation/e510_raw_teacher_pair/prepare_packet.py \
  --e508-profile-receipt <actual-E508-profile-result.json> \
  --e508-profile-receipt-sha256 <actual-profile-receipt-sha256> \
  --e508-terminal-output <downloaded-E508-terminal-output.txt> \
  --controller-review-receipt <external-controller-review.json> \
  --controller-review-receipt-sha256 <controller-review-sha256>
```

The review receipt format is `wnn.controller.e510-review.v1`. It records
`profile_review` with the approved E508 receipt and terminal-output hashes,
`e510_code_review` with the exact code identity fields, and an approved
`production_budget`. The builder checks the hashes and the 6-hour freshness of
the quota check. It does not create this receipt or treat a hash supplied by the
packet author as approval. No packet is built by local tests.

The production worker first inventories actual mounted inputs and devices. It
then requires at least 4 GiB and 20% of effective host/cgroup RAM to remain
available, repeats that check every 10 seconds during work, and terminates if the
headroom bound fails. The external review also binds the measured E508 profile
to the frozen dual-T4 four-GPU-hour production budget. No production launch is
authorized by this code change.

The sealed teacher recipe is scratch raw-only GRU(128, two layers), seeds 3 and
4, four epochs with fixed zero-based epoch 3, wMSE, AdamW/OneCycle at lr 0.001,
batch 16, chunk 2,000, and no EMA, distillation, auxiliary, or propensity losses.
It uses the exact fold-0 fitting groups (10,535 train plus 1,405 valid), excluding
468 evaluation groups and corrupt group 528. The legacy `--train-seqs 10607` is
only an upper bound. Feature-only float32 predictions preserve the complete
existing target-cache row order.
