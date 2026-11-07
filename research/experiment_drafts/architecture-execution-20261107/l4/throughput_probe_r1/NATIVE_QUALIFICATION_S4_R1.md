Historical receipt107 cutoff; subsequent successful canonical probes and active
production updates are recorded in [ACTIVE_CONTINUATION_S4_R3.md](ACTIVE_CONTINUATION_S4_R3.md).

Native S4 source preparation and both seed-specific engineering profiles passed.
ROOT approved the measured receipts after verifying the unchanged original
E8/51680 parent checkpoint hashes and exact logged S1 loss/gradient agreement.
Canonical CPU and GPU resume remain pending at this receipt107 cutoff. A fresh
20-update profile does not execute the monitored resume path.

[Builder104](native_receipts/104-build-s4-source.json) returned0 and reports S4
inventory
`28ac81b46517f1923d4ffd9751bd0d615f2657c5721d2e3459726e8c05ef4182`
and patch receipt
`c1cb973296cd0c4cc37ad381eb3a97114560ad7dedea98142b181a218eb02c54`.
It preserves the exact three S3 diagnostic/admission amendments and changes only
sequence_fit_resume.py relative to S3. That file's new SHA is
`b0eed5d6bbf5cc759f6338a48be4099e321a853f6d42bfc05d3763d1fb5f9c8d`.
The cumulative source whitelist remains exactly four files relative to S1.

[Launch105](native_receipts/105-run-s4-finite-profiles.json) records two
engineering profiles and zero scientific fits.
[Results106](native_receipts/106-s4-profile-receipts.json) bind both profiles to
canonical trainer source
`5bbbf68eccdc8c8773ff4b4eda2c4026d9c3f31b145dfc0d683b17d20a1987d0`.
The actual output namespace is gru-monitor-continuation-profiles-r4; the raw
config namespace remains gru-monitor-continuation-r3. These revision labels do
not introduce new fitting recipes.

| Existing seed / GPU | Updates | Native elapsed seconds | Peak reserved bytes | Errors |
| --- | ---: | ---: | ---: | --- |
| 101001 / GPU0 | 20 | 3.420854151248932 | 1306525696 | [] |
| 101002 / GPU1 | 20 | 3.3998799920082092 | 1306525696 | [] |

Both owned executions exited0 without timeout. Both report world1 and data SHA
`ff5c06681d84acd8d9697c5059b784ae9c671067ea9aed802ff5cb891b803201`.
The full unchanged normalized config hashes are in
[NATIVE_QUALIFICATION_S4_R1.json](NATIVE_QUALIFICATION_S4_R1.json).
Device-total memory is the per-rank list[23670685696], rather than a scalar.

[ROOT approval107](native_receipts/107-approve-s4-profiles.json) returned0.
For each seed, paired maximum absolute label-loss and preclip gradient-norm
differences at logged steps10/20 are exactly0.0. The original parent checkpoint
hashes and E8/51680 frontier remain unchanged. The separate approved profile
hashes are:

- 101001:45c7c2bfe80bfae174a18689779caeb38c2a30f71ece4e4e5016748bed7fd681
- 101002:5090e2bafc6987dc554944c5374e6f5476ce1b737d7ee6ece9f398abd0507ae9

Sampled profile parity does not establish full-training bitwise parity or quality
equivalence. ROOT approval still requires real canonical CPU and CUDA
zero-update resume probes and a live supervised full-state archiver before
same-fit continuation. No S3 profile is relabelled, no new scientific fit is
counted, and the original epoch12 endpoint and deadline remain unchanged.

The accepted engineering helper, its three CPU tests and the exact ROOT approval
script are archived byte-for-byte under [native_helpers](native_helpers/).
The .txt suffix keeps historical path-dependent test snapshots from being
accidentally collected or executed. Original names and hashes are in the JSON
record. The copied helper SHA is
`ed7cf023f86b8d2c3bccfbdea1fd0ee80566b2824c47b8999188143f36cafee0`.
The original three CPU tests were rerun from their actual artifact path and
passed in0.18s; they mock the owned executor and perform no signals, CUDA,
model execution or network access. The ROOT approval script was copied and
inspected; the packet actor did not execute it or approve profiles.
