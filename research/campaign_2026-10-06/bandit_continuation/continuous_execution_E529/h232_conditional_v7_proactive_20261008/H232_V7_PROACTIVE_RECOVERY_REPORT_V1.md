# H232 V7 proactive recovery packet

V7 is a bounded execution repair of the V6 seed-3 fit attempt. It changes the
resource trigger, failed-proactive-recovery handling, and explicit worker wall
budget. The training source archive remains byte-identical to V6.

## V6 diagnosis

The terminal receipt is
[`basis/V6_TERMINAL_POLL_RECEIPT.json`](basis/V6_TERMINAL_POLL_RECEIPT.json),
SHA256 `95e5ddbb8027d1f8bbc56326dbde64c7943373bfd4edcb8b17720005f906b503`.
V6 passed its actual-device profile: 200 timed optimizer updates were finite,
with zero AMP skips. The fit reached epoch 0 batch 1060/1492 at 1452.987 seconds.
It produced no checkpoint, prediction, or score.

The 14th and final relief event was eligible and advised 280 identity-bound
rowgroup ranges totaling the 768,000,000-byte per-event cap. Cgroup availability
rose from 8,036,401,152 to 8,467,537,920 bytes. The unchanged 25% plus 512 MB
recovery target was 8,565,063,680 bytes, leaving a 97,525,760-byte shortfall.
The post-advice sample still passed the 25% hard fraction gate at 26.2867% and
the 6.5 GB hard byte gate; host headroom also passed. There were no worker OOM,
swap, scan, or refault events in the terminal receipt.

At the same 32,212,254,720-byte cgroup limit, a 28% trigger is 9,019,431,322
bytes, about 966 MB above the 25% hard fraction threshold. V7 would have
attempted bounded relief at the recorded post-advice sample while all hard
gates still passed.

## V7 execution contract

- Trigger the existing bounded recovery when cgroup available fraction falls
  below 28%. Keep the original 6.5 GB and 25% hard gates.
- Preserve the existing memory-stat, same-cgroup history, anon/active-file/
  refault, input identity, rowgroup range, rate, event, and cumulative-byte
  checks. Advice remains `POSIX_FADV_DONTNEED` against recent traced ranges from
  the already-open identity-bound train and valid files.
- Keep the 768 MB event cap, 320-event cap, 245.76 GB cumulative cap, and
  20-second interval. Continue only after advice when the full target is met,
  except that a proactive-only attempt may miss that margin and continue if the
  consistent post-advice host and cgroup samples still pass every hard gate.
  If hard pressure existed at trigger time, missing the full target remains a
  fail-closed abort. A veto before advice leaves enforcement to the unchanged
  hard gates.
- Cap the whole worker at 16,000 seconds. The profile gate projects a complete
  four-epoch fit and reserves the full 1,800-second fixed-epoch-3 FP32 valid
  export. It does not shorten training. The fit remains GRU384x2, width 211,
  B8, seed-3 scratch, AMP scale 4096, raw WMSE, four epochs, fixed epoch 3,
  with no propensity, teacher, auxiliary loss, or evaluation-group fitting.
- Export valid predictions only. The worker computes no metric or score.

V7 retains V6 source archive SHA256
`f1333c63eca38e69149e14dbc94c6a701ba97e113d6679b8939d7c860f1a95e4` and
manifest SHA256
`6ee80717ab4cec6da8b9ef3f124439a7598c2dfde2fa25de9fb35bf47a0adc42`.
[`V7_BUILD_RECEIPT.json`](V7_BUILD_RECEIPT.json) binds those hashes to the
packet, builders, tests, and generated Kaggle transport bundle. The new
canonical kernel ID is `vladkochetov07/h232-gru384-b8-amp4096-v7-proactive`.
[`V7_V6_EXECUTION_DIFF.patch`](V7_V6_EXECUTION_DIFF.patch) contains the exact
text diff for the changed execution, packet, and test files; its hash is in
the build receipt. The generated transport wrapper is reproducible with the
bundle builder and is omitted from that diff.
The packet review is pending; provider status was not queried and no fit or
push was made.

## Checks

Under the assigned 1 CPU / 1.5 GB boundary, the focused recovery and fit-gate
suite passed: 22 tests. Ruff checks passed for the modified Python files. The
source archive builder reproduced the V6 archive byte-for-byte.
