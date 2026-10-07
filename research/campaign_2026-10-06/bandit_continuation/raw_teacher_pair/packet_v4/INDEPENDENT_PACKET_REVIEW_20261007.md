# Independent E-508 v4 packet review

Reviewer: native Luna subagent, read-only. Result: **PASS; no blocker before
cloud push.** The reviewer did not run Kaggle commands or launch compute.

- The v4 guard rejects train row-group 528 only in the train namespace and
  permits validation row-group 528. Exact fit counts remain checked, and the
  pinned split loader validates train/valid membership separately against its
  component graph.
- The packet embeds the current worker byte-for-byte and the pinned trainer
  source hashes match the receipt.
- `PROFILE_ONLY=True` exits before the fit loop. Packet metadata remains
  private, offline, GPU-enabled and TPU-disabled; the review receipt explicitly
  says no fit is launched.
- Raw file SHA-256 and footer identities, fold/assets/source manifests, split
  membership and target-cache row-index checks remain in the pre-profile path.
- The existing v3 packet and metadata hashes remain unchanged.

The reviewer references `packet_worker.py` lines 184, 973, 1047 and the pinned
`src/wnn/splits.py` loader. The precise terminal evidence for v3 is preserved in
`cloud_preflight/e508_profile_results_v3/` and its terminal receipt.
