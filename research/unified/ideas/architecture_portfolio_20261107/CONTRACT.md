# Architecture portfolio: planning contract, 7 November 2026

Objective mode: competition. Choose a small, structurally diverse architecture
portfolio with the best defensible chance of improving the complete root
Challenge2 regressor/ensemble, using the user's existing8 B300 and8 L4.
This is literature/evidence review and design only: no fits, GPU profiles,
downloads of training data, rentals, old-job restart or remote process signals.
The older4 November allocations and running notes are expired historical state.

Task-shape source audit: `src/wnn/data.py` declares20,000 rows per complete
sequence and112 raw features.2,000 is a TBPTT chunk, not sequence length.
`event211` is112 raw plus99 derived causal event inputs. The first scout briefs
had stale dimensions; root corrected all scouts before their final handoffs.
Train has no `is_scored`: the current fitting mask is all rows after warmup99;
the supplied scoring mask applies to validation, while all rows advance state.

Evaluator: the existing two-target global clipped weighted Pearson kernel on
the official mask. Use official fitting train10325, validation1873 and
3,528,273 scored rows where the qualified manifest applies. Whole sequences,
causal inputs, exact state reset, fitting-only preprocessing/teacher/initializer
ancestry. Never infer chronology from shuffled seq_ix. Primary validation is
already development-exposed; no random split can replace the official boundary.

Known local endpoint anchors, pending a fresh results check: TL3 WP0.692377425;
the earlier fixed90%TL3+10%LSTM endpoint WP0.692502063. PR61 wider-GRU students
did not improve matched controls. PR62 clipping/auxiliary changes were negative.
The terminal PR61 Transformer/student result is not yet known in this review.
These scores are validation evidence, not public-score forecasts.

Post-freeze source refresh: the actual PR61 terminal comment reports10/12 fits
complete, Transformer scores0.6717207056476808 and0.6729531605803009, and no TF
student fits. The handoff failed on a profile-device/receipt identity check
before target construction. The final R5 release endpoint returns404; the prior
eight-fit release is separate. New PR63–67 describe preparation, not measured
quality results. Raw responses and exact source pins are retained in observations.

Deliverables: three bounded independent literature/representation lenses; a
source-grounded ranked architecture specification; a measured-versus-unknown
evidence table; exact initial comparisons, serving paths and falsifiers; and a
proposed hardware allocation. Approximate parameter counts may be calculated;
throughput, runtime, VRAM and expected quality remain unknown until real profiles.
Do not materialize a large Cartesian architecture/hyperparameter grid.

Verification regime: empirical development validation with selection exposure
and unknown public/private transfer. Literature supports mechanisms, not our
dataset quality. No numeric unseen-score improvement is asserted. Independent
actors do not approve their own proposed methods.

Root owns integration and read-only current-results access. Three independent
scouts each write only their assigned lens note, use primary papers/official
code, do no SSH/GPU work, and do not read competing lens notes before committing
initial proposals. Model policy follows active AGENTS: Luna/max; independence
comes from fresh context and different source/mechanism assignments, not claims
of model diversity. Local support remains bounded to light repository reads.

Time scope: one planning turn. No new compute allocation or monetary budget is
created. A later runnable screen needs frozen recipes, actual resource admission,
profiles, a bounded wall-time allocation and an end-to-end artifact owner.
