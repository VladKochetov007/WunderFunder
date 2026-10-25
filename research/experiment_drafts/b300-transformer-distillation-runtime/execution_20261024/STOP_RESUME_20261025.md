# Stopped with resumable states — 2026-10-25

The user explicitly requested stopping training and continuing later. The
operator verified that both training/observer controllers exited, no CUDA
compute processes remained, and all eight GPUs were free. The campaign's three
training/validation reporting and watch cron entries were disabled; unrelated
server-monitor cron entries were preserved. No new fit or replay was launched.

| Cell | Last complete fit epoch | State |
|---|---:|---|
| 630M F0 | 30 | Fit and final evaluation completed |
| 630M F2 | 30 | Fit and final evaluation completed |
| 1.54B F0 | 21 | Stopped during epoch22; resumable safe state retained |
| 1.54B F2 | 21 | Stopped during epoch22; resumable safe state retained |

The large models' safe states contain 9,870 successful updates, model weights,
optimizer, scheduler and per-rank RNG state. Partial epoch22 updates/logs are
not a complete epoch22 checkpoint. All retained epoch15–21 snapshots and all20
complete validation points remain available. Epoch15 is the best retained and
evaluated large-model epoch, not a demonstrated optimum across earlier epochs.

**Metric clarification:** lower WMSE is better; higher official weighted Pearson
WP is better. Large-model F0 WP fell from0.371964 at epoch15 to0.267118 at epoch21
while val WMSE rose from1.516398 to1.536049. F2 WP fell from0.414020 to0.283140
while val WMSE rose from1.510489 to1.541116. Online train loss declined. This
supports stopping this continuation for worsening observed generalization;
neither the fit nor the experiment is relabeled as a completed30-epoch result.

The [PR32 stop record](https://github.com/VladKochetov007/WunderFunder/pull/32#issuecomment-6392305520)
and [PR40 record](https://github.com/VladKochetov007/WunderFunder/pull/40#issuecomment-6392305667)
preserve the result and stopping reason.

Runtime handoff root:
`/data/epanov_s/tabdata/runs/pr32-user-stop-20261025/`.
Read `RESUME_HANDOFF.json`, `STOP_REQUEST.json`, `STOP_VERIFIED.json` and
`GITHUB_STOP_PUBLISHED.json` there. The per-cell `safe-epoch-capture.pt` files
are immutable hardlinks captured before termination; completed small-model
`final-model.pt` files are also retained.

Resume only after an explicit user instruction and fresh resource verification.
Use the original canonical resume path and preserve source
`f1beeb4c210a9da01735e03b1ff6e27be878b873`, config/split/profile identities,
precision, batch, world size, optimizer and RNG/sampler semantics. The original
cohort controller is a one-shot fresh-fit launcher, not a resume launcher.
Canonical resume must truncate partial epoch22 log suffixes to the saved epoch21
offsets and retain truncation evidence. Restore monitoring explicitly when
resuming; its cron entries are now disabled. Keep all weights and data off Git.
