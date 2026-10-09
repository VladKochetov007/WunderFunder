# Independent review of fold1 teacher export preparation

Reviewed 2026-10-09T03:40:03+00:00. The preparation is **passed pending seed4 terminal audit**. The review binds preparation SHA `72037a8b884dd1db8dcd361c37fa65fe656c8f858396cfada71a53e61fe465b1`, wrapper SHA `703e85fd31afd9db462e3ee475d8c2982a91190404cfbd903c14073180301210`, and the exact 51-entry pin map (`b9ba89c5e9bc7b7553f5a4b92dd75027f0508be6c9ae890e5bd0b587b5f5f093`).

All 51 current pins were present and matched; all eight prediction/partial/sidecar/shared-seqsums output paths were absent. The eight declared FP32 NPY file sizes match the sequence shapes including the 128-byte NPY header. At review time, free disk was 139,682,811,904 bytes, above the 10,000,000,000-byte reserve plus 10,781,121,024 expected output bytes.

The wrapper checks split membership and ordered target rows, runs ancestry closure once during startup preflight per distinct run/epoch, rechecks pinned source/checkpoint/provenance artifacts before each export, refuses overwriting existing or dangling-symlink outputs, and validates each prediction with the repository's actual teacher consumer. It also checks `STOP_ADMISSION.json` before starting each export.

No blocking preparation defect found. The seed4 epoch-3 checkpoint and provenance are intentionally not included in the current pins; bind them only after independent terminal review. The declared claim is exposed-fold1 export readiness, not a score or public-performance result. Root reports the focused actual-producer tests passing 3/3; I reviewed their source but did not rerun them.
