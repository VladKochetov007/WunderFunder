---
kind: experiment
status: running
created_utc: 2026-10-17
updated_utc: 2026-10-17
hypothesis_ids: [H-MOMENT-GEOMETRY-20261017-R1]
experiment_ids: [E-MOMENT-GEOMETRY-20261017-R1, E-MOMENT-GEOMETRY-PILOT-20261017-R1]
evidence_paths:
  - research/campaign_2026-10-17/moment_geometry_r1/README.md
  - notebooks/sequence_moment_geometry_20261017.py
---

# Sequence moment geometry: descriptive arcs and causal representations

## Question or claim

The user supplied a scatter of sequence mean versus standard deviation and asked for marimo EDA, causal scaling/guidance/early inference and bounded local experiments directly on main. The screenshot's feature and window are unknown. This explicitly resumes this investigation only; unrelated paused campaigns and other agents' Kaggle work are untouched. Objective mode is competition. Fixed-data measurements have tier B verification; transfer from a tiny pilot to the competition endpoint remains tier C.

## Mechanism and method

The registered [contract](../../campaign_2026-10-17/moment_geometry_r1/contract.json) preceded data outcomes. A seed20261017 sample contains128 complete20000-row sequences per file, all112 raw inputs, warmup and unscored rows included. Known corrupt train group528 is excluded. Prefix lengths32,100,512,2048,8192,20000 use population variance (`ddof=0`). Statistics are descriptive; full-sequence moments never become causal input features. Validation is exposed/unknown rather than pristine. The initial scan ranks112train-fitted fixed-curvature relations; choosing columns, a consensus fit and illustrative sequences afterwards is exploratory selection.

For exactly two values a,b, population variance equals `(mu-a)*(b-mu)`. Every distribution supported on[a,b] instead satisfies the inequality. This proves an arc can be algebraic without adding information.

A more general alternative matters here: let X be a mixture of a point mass at a and a fixed broad distribution Q. Changing the mixture proportion makes E[X²] linear in E[X], yielding `variance = B*mu+C-mu²` again. Its second zero can be a **virtual endpoint**, not a support bound. Different Q or different anchor values can generate multiple arcs/clouds. Stable conditional Q moments are a proposed explanation, not an established market mechanism.

Whole-window [RevIN](https://github.com/ts-kim/RevIN) is related normalization literature, but its [official implementation](https://github.com/ts-kim/RevIN/blob/master/RevIN.py) computes moments over the supplied time axis. Supplying the entire current competition sequence would make early outputs depend on future rows. Our treatment uses expanding current-inclusive moments reset per sequence, retains raw levels and adds a scale floor. It is not a direct reproduction of RevIN and does not borrow its reported gains.

## Evidence and interpretation

The six-example figure reproduces a very similar arc for **i0_p0**, with an analogous shifted arc for i1_p0. This is a plausible screenshot match, not recovered screenshot provenance. A train-only exploratory consensus for i0_p0 gives beta≈−3.66807 and intercept≈−3.20564 in `var=beta*mu+intercept-mu²`;46.1%train and46.9%valid sampled sequences fall within0.002variance units. This adaptive consensus fraction is descriptive, not a generalization score or a test of predictive value. A single least-squares curve fits all points poorly (valid R²≈0.24), supporting the visible multi-regime qualification.

Actual near-arc sequences are not strictly two-valued: examples have8 and23distinct values. Group1793 has92.285%of rows at its two most frequent values, while its most frequent level alone occurs13858/20000 times. A frequent-background-value mixture is plausible. Off-arc examples have hundreds of levels and much weaker two-mode dominance. The six examples were selected on their observed geometry and are not prevalence estimates.

For i0_p0 on the sampled valid sequences, the absolute error of prefix std relative to final std averages0.11033at100 rows,0.08073at512,0.05144at2048 and0.02503at8192. Mean errors are0.13227,0.08382,0.05703and0.02941respectively. Final-statistic reconstruction is a diagnostic with a future-defined reference, not an available serving feature or evidence of target prediction gain.

### Direct visual inspection of all112 features

Four contact sheets were rendered and visually inspected. Most price-like p* columns show a narrow branch/arc with additional off-branch clouds; corresponding i0/i1 shapes often resemble one another. Branch direction and curvature vary by column. Volume-like v* fields more often show broad or multiple clusters. Trade-volume dv* fields show strong column-specific mean/std coupling. Additional a0–a3 are mostly clouds, with a1 means concentrated near zero; a4/a6 show decreasing mean/std association and a7 shows two visible branches. Train/valid colors largely intermingle in this sample. These observations do not identify feature semantics, prove equal distributions or establish latent market classes.

## Frozen predictive pilot

The [pilot plan](../../campaign_2026-10-17/moment_geometry_r1/pilot_plan.json) and [source-qualified membership](../../campaign_2026-10-17/moment_geometry_r1/pilot_membership.json) were committed before target scores. Twelve complete fitting components from folds2/3 contain43 sequences from permitted train and valid data. Eight evaluation components each from folds0/1 contain46 valid sequences total; all their train counterparts are excluded from fitting. One common fitting pool is evaluated on two exposed slices; this is three distinct fitting seeds, not six independent fits.

All arms reuse RNNModel(hidden32,layers1), raw112+existing99event channels and20 extra positions, identical initialization and sample order for seeds3/7/11. Control zeroes20normalized extra channels. Moments retains16 channels (expanding mean/std, train-fitted arc residual and frequent-anchor occupancy for i0_p0,p3/i1_p0,p3) and zeroes the last4. Scaled additionally retains clipped current-minus-prefix-mean/prefix-std with0.05floor. Every preprocessing constant, anchor and scaler is fit only on declared fitting membership. Zeroing happens after normalization, preserving matched width and a true zero suffix.

The fixed budget is2 epochs, batch4,512-row chronological TBPTT, FP32, AdamW(lr0.001,weight_decay0.01), cosine update schedule to1e-5and gradient clipping1. Hidden state resets for each full sequence and detaches at chunk boundaries. No checkpoint selection uses validation. Existing weighted_mse_loss and official global_wp_clip2_v1 are reused; original clipped-label magnitude weights, ±2evaluation clipping and supplied prediction/scored masks remain fixed. Allseed outcomes must be reported. A full30seed/fold campaign is not claimed by this pilot; confirmation and a production-size endpoint remain future steps.

## Limits and validity

No predictive result is available at this note's initial registration. Tiny-data results cannot establish a production teacher/student gain or rule out a scale-dependent mechanism. A conditional component bootstrap describes evaluation sampling given fitted predictions; it does not include training variability, selection or future transfer. Previously inspected folds stay exposed.

The first scan hit its1.4GB child cap before sampled data outcomes because Parquet metadata was opened redundantly. The corrected single-reader scan finished at≈2.6GiB process peak. The task slice remains5decimalGB RAM, with one CPU core inside the existing two-core parent. Subsequent staging peak including file cache was4.502 GB; dropping only this task's completed NPY cache via POSIX_FADV_DONTNEED reduced charged memory to≈2.1GB. This preserves operating headroom without changing the user's limits. No unrelated process is terminated. GPU admission is limited to one tiny worker and≤18%of the6GiBallocator; another project's≈3.24GiB job remains untouched.

Independent Luna/max criticism preceded outcomes. It identified missing input-source/mapping and storage pins. The repair validates the trusted original Parquet footer+trailer hashes/byte/mtime identities, cached target row-group→seq_ix index hashes, graph identities, approved physical scratch and free disk. An actual-producer pass verifies all89 selected sequence IDs, one raw channel, both targets and masks against staged arrays. Causal suffix mutation and prefix-only recomputation agree exactly for the20 new channels. These checks establish the tested input path, not model-quality transfer.

## Decision and next step

Keep raw levels and use soft evidence of regime/occupancy; do not project every sequence onto one arc. Run the frozen pilot and report all pairedseed/fold-slice outcomes. If promising, test the learned representation on the existing stronger student/teacher endpoint with full repetitions and serving measurements. A useful follow-up should distinguish ordinary causal moments, atom occupancy, arc residual and scale normalization rather than attributing a combined gain to the circle itself.

## Source records

- [Complete campaign evidence and reproduction entry](../../campaign_2026-10-17/moment_geometry_r1/README.md).
- [Live marimo artifact](../../../notebooks/sequence_moment_geometry_20261017.py), [data semantics](../../../docs/data_overview.md), [source split provenance](../../campaign_2026-09-29/artifacts/splits/fold0.json).
- [Feature implementation](../../../src/wnn/moment_geometry/features.py), [staging](../../../src/wnn/moment_geometry/staging.py), [provenance checks](../../../src/wnn/moment_geometry/provenance.py).
