# Receiving amendment R2: actual execution, 10 November

The scientific source remains commit `c157877d261efaa0a58cd1ae8ba5fc61c914b72d`,
capsule `b917b83edd463760d83829e44f87b57af5473507302e558d36fad3d1034cd01b`.
The four recipes and original finite wave clock are unchanged. This directory
preserves the exact executed receiving helpers, conditional ROOT admission,
independent review, and a native evidence snapshot. `manifest-r2.json` hashes
the copied bytes; formatting these frozen helper copies would invalidate them.

The first receiver exhausted its 600-second CPU preparation allowance before
creating a target cache or launching an SSL profile/fit. It remains a failed
engineering attempt, not a failed quality experiment. R2 uses 16 CPU processes
for independent per-sequence sums and squares, then reduces them in the original
sequence order. This preserves the canonical formula and reduction order:
19,645 target rows fit within its 20,000-row chunk. The actual cache preparation
completed in 66.310 seconds. The first eight actual fitting members also passed
bitwise serial/parallel FP32 mean/std equality before cache publication.

R2 conditional admission is
`b544180c502a94344e12bc94f08f02cced5a1f399a3d46e39d7a3b62197ac103`.
The target normalization identity is
`50741095692b08a2f3cf57fdce00705c15e5a5f215cde66a48eb220b668083b3`;
moments identity is
`dda5ecc40f3e1edb7477787e5036e05ed60344bd6fd75e2c1088d75527f4ef76`.
All 10,325 fitting members and no validation members contribute. Actual source
features supply 202,834,625 masked target rows; task labels and scoring masks are
not read during pretraining. The separate gradient fixture changes task labels
to NaNs and flips masks without changing any parameter gradient.

Both actual 30-update L4 parent profiles and state reloads passed. GPUs 6 and 7
then began the two fresh fixed-2E parent fits. Each fixed-4E child is conditional
on its completed same-seed parent's receipt, checkpoint hash, normalization,
source/config identities, an actual child profile and full-state reload. Its
optimizer, scheduler and counters reset. Transfer excludes the task/hint heads.
There is no validation-quality selection gate and no automatic extra fit.

The native controller is detached and bounded by the original first-profile
clock: 11 hours fitting and one hour collection, combined output cap 6 GB,
4 GB free-space reserve and 128 GiB owned host RSS. SSH disconnection does not
stop it. Foreign processes remain outside its ownership. Receipt timestamps use
the native clock, observed approximately two hours behind the local clock;
neither the server clock nor source events were rewritten.

The snapshot is an observation of execution at collection time, not a completion
claim. Ongoing stage status and scores are reported in PR78. Pretext loss is a
training diagnostic; the full official validation after each supervised child
epoch is the development metric. This validation is repeatedly exposed and
cannot establish an expected public score of 0.70.
