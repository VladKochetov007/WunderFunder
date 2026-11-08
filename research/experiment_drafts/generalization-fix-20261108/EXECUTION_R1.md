# Native L4 execution, 8 November 2026

Six fresh fits admitted after all six actual-shape native30-update profiles
passed with no scientific checkpoint written. Frozen training source is the
previously qualified S4 producer
5bbbf68eccdc8c8773ff4b4eda2c4026d9c3f31b145dfc0d683b17d20a1987d0;
the new postfit reader/SWA code does not mutate that training capsule.

Runtime: /data2/lc-epanov_s/tabdata/runtime/generalization-fix-20261108-r1.
Fits run detached under controller2261449/birth2148815876, launched
7 November22:37:22.615738UTC. Native22:42:47UTC observations confirm all six
have4530–4760 successful optimizer updates, finite loss and no OOM. These are
execution observations, not quality results; no full validation had completed
at that observation. Source/config/data/normalizer/GPU admission is preserved
in ROOT-fit-admission-r1.json and the six approved native profile receipts.

GPU0/1: raw112, no dropout, seeds101101/101102.
GPU2/3: raw112, dropout0.2, same seeds.
GPU4/5: causal event211, dropout0.2, same seeds.
GPU6/7 are reserved for readout/support; no extra fits are authorized.
Foreign root services on GPU4 were preserved.

Each fit is fixed6E/38760updates, B32/C1000, GRU256x2, full20k sequences,
FP32 AdamW3e-4, decay.01, clip1,5%warmup/cosine. All fit only the10325 official
train members and evaluate all1873 official valid sequences/3528273 scored rows.
Train has no is_scored source column; fitting uses explicit all_after_warmup.
Valid remains exposed development evidence, never an untouched final holdout.

The first actual profile started22:27:25.763144UTC. Fit admission/cutoff is
8 November01:27:25.763144UTC; whole wave/collection cutoff02:27:25.763144UTC.
These clocks are not reset by fit launch or readout. Resource guards enforce
owned output4GB, free disk3GB, owned RSS128GiB and host available64GiB.

Fixed endpoint readouts are SWA of exactly E3/4/5/6 (primary) and E6 (secondary),
each standalone and fixed0.9cleanTL3+0.1candidate.24 exported endpoint grades
plus36 per-epoch monitoring looks are development-exposed. Reference baseline
WP0.6923774248169172 is reused, not a new gain. Exact cache transfer verified
1873 file hashes, reference manifest1863e5e6f844c8f172f3fd54879ff908e0e415183766c354482b1ac9c5bfd985.
Historical qualification limitation:
published-legacy-consumed-cache-byte-ancestry-not-reconstructed.

Local dashboard: http://127.0.0.1:8769/; l4 observation root now points to this
wave. Transport retains the existing persistent SSH stream, no new SSH login.
Native full-valid scores, parent hashes, exported predictions and fixed blend
grades are pending; no local/public improvement is claimed.

At22:53:01UTC ROOT admitted and armed separate fixed-readout controller2272381,
birth2148909734. Readout source identity
0042a76f80f685666da7f62f263b2d355d0c35c6eca18a1e0d1fe39097660bab was verified
on the node; native control/l4-readouts-r1/ROOT-admission.json binds it to the
qualified reference. It waits for complete6E parents and runs on spare GPUs6/7.
The live training receiver/source was not edited. Actual E1 full-valid WP for
the six seeds ranges0.66971–0.67485; fixed endpoint comparisons are still pending.
Draft PR71 is live under SemyonEpanov. The separate B300 follow-up is PR70.
