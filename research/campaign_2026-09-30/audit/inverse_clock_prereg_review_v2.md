# Revised inverse-clock preregistration and replay

Source SHA256 `468866b5138a25f7e40967e87fb020f27f1b5b717bb146debbb464120cfd52f2`.
Preregistration SHA256 `59211cbe0fa7b7ecfd17637235df2eee754d26ac70fe455a29fca0f83a0813d0`.
Author results SHA256 `b84947a7dc8b11647152079d45b5a7e3967558f3e2b122622a773f9905666dec`.

The revision addresses unsupported anchors, observed-increment collapse, scale-normalized controls, quiet-column definitions and support/concentration diagnostics. Source fits only selected TRAIN inputs; persisted fit_support/fit_result precede test reading. It excludes current/past clipped endpoints and raw negative increments, keeps zero differences, uses all22price+8dp/dv equality, and applies fixed TRAIN scalar controls and offsets to heldout inputs. Existing diagnostic class records never enter map fitting. Interpolation, analytic gradient, unequal-knot slope curvature, range/slope constraints and68-iteration feasible solve match the stated recipe. No test-driven retuning is present. Files alone do not prove filesystem chronology; retain durable log and frozen launch/source records. The same64/32 panel remains exploratory.

Independent `replay_inverse_clock_v2.py` reads raw current/past feature columns, implements eligibility/interpolation/control/mass calculations independently, and never imports author code, reads targets, or refits. It reproduces32768FIT and432796test rows, all three aggregate metric vectors and all16cell masses at rtol1e-10/atol1e-12. FIT effective cells3.732831268431459 fails>=4; heldout4.021309022198193 passes. Candidate heldout normalized error.6539495138730975, CDF1.1597678334040011, log-survival.993066478732284. Outcome: calculation reproduced; identification gate failed; D3 blocked.

Durable CPU job `c20260930_independent_inverse_replay_v2`:2.44s analysis, peak1102280KiB, one thread, CUDA hidden, shared slice. Actual usage<2GiB; launcher outer hard memory limit was36GiB rather than requested2GiB. This limitation is disclosed; later CPU probes require an explicit nested2GiB scope. No GPU, candidate metric, new data panel or scorer mutation. Do not relax the cell gate after observing the outcome.
