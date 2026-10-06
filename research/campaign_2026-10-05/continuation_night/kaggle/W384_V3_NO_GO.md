# Independent review: W384 v3 packet no-go

`cloud_critic_luna` reviewed the frozen packet
`wnn-night-w384-serial-20261006-v3` and returned NO-GO before any provider
action. The reviewer found that `phase.py` checked the epoch-3 checkpoint but
used a stale `c20261005night_w{width}_*/*` glob for the artifact manifest. The
manifest therefore omitted the fresh v3 checkpoint and its provenance/history
files; it appended only the prediction files after the stale scan.

The rejected packet is preserved unchanged at
`research/campaign_2026-10-05/continuation_night/kaggle/packets/wnn-night-w384-serial-20261006-v3/`.
Its kernel SHA-256 is
`10f832310f5242d0b7f2e999631594418cdabf1d4cb153e3b2d460a304163814` and its
review packet SHA-256 is
`af0831693d2664bd602c9c666d49e641216c72a4ac478928af4e0b165a56370a`.

The v3r1 revision corrects the exact run-directory binding and requires all
fixed-epoch output files before writing an artifact manifest. It retains
provider recovery attempt 2 of 2; this packet revision does not consume another
recovery attempt. No packet was pushed and no provider job was launched.
