# Live execution, 4 November

PR: https://github.com/VladKochetov007/WunderFunder/pull/61, draft targeting research-team, published as SemyonEpanov.

At13:41UTC the six GRU scientific fits are running (controller PID1541719; phase fit_running). Each has reached its first real-data batch. GPU0-5 utilization99-100%, owned model allocations about3.7-4.0GiB; GPU4 also retains the unrelated1148MiB service allocations. Two Transformer engineering profiles (controller1541485) are preparing fitting-only all-train normalization; their worker CPU usage is100% each and they have not yet produced a resource receipt or scientific fit.

All six native GRU resource profiles completed1360updates each, with zero AMP skips/nonfinite gradients and exit0. Four-epoch training-only projections from measured throughput: GRU256x2 about27min, GRU384x2 about40min, GRU512x1 about36min. Full validation, diagnostics and exports add time; Transformer duration remains unmeasured.

Frozen GRU code is27e605aa488651a3ab6cd38d3a1965075c24760f; frozen Transformer code isffebdbdc573e4494e0b53bd661c544ecc490b3df. New local orchestration/monitoring changes must not replace either live source. Exact source/admission/profiles and independent actual-profile review are retained under the receiving control directory. The first dispatcher attempt failed before training because its admission omitted fit_members_sha256; R2 verified the unchanged canonical10325-train/0-valid membership and admitted the same panel. No failed scientific run or recipe retuning is hidden.

The four student fits depend on complete fixed-epoch3 teacher checkpoints and independently checked target caches. Their fixed family allocation and target formula are already in CONTRACT_R1.json. They have not started yet. All primary scores must use the complete official validation mask, and must be labelled exposed development. No current learned endpoint or leaderboard gain is claimed.
