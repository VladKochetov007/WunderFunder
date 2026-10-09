# Inactive prediction archive terminal receipt

Status: PASS. The admitted user-systemd unit finished successfully with exit status 0. The frozen plan covered 82 regular prediction files totaling 27,368,330,496 bytes. The journal contains 82 full-copy verification events and 82 completed source handoffs, with matching SHA-256 receipts for every pair.

Independent terminal checks found all 82 original paths are symlinks to their exact planned HDD targets; all 82 destinations are regular files with the planned sizes on the separate HDD filesystem. The archive contains only the 82 planned targets, README.md, plan.json, and moves.jsonl; no partial or handoff files remain. HDD free space is 169,477,492,736 bytes.

I did not rehash the 27.37GB payload. The reviewed transfer helper streamed and checked each full SHA-256 before source replacement; this terminal pass cross-checked the verified/moved checksum pairs and metadata. The last live cgroup sample showed 0 OOM events and a 512,004,096-byte peak against a 512,000,000-byte cap; the cgroup was removed after the successful unit exit.
