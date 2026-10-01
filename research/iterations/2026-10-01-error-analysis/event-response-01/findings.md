# Observed-event response: actual findings

**The event curves do not support a broad “the model reacts too slowly” diagnosis.** Predictions already move substantially at the observed encoded event. This is descriptive evidence against prioritizing a generic faster-update branch; it is not a proof of optimal reaction or irreducible noise.

The diagnostic used 55,040 deterministic input-selected anchors from all860 pure sequences, yielding437,520 valid offset rows. No new model inference, forecast fit, shifted-prediction score, or correction was run. Normalizers come from all1,614,484 scored main-table rows; pooled normalization reproduces the registered student4 score within1e−10. Results by fold/class/target and all predeclared orientations remain in the CSV/JSON.

At offset0, orienting each target by its **own model-output change**, the target-weighted increments from the preceding row are:

| Fold / target | Clipped target increment | Clipped prediction increment | Oriented q = u−ρz |
|---|---:|---:|---:|
| 0 / t0 | .1075 | .1798 | +.0287 |
| 0 / t1 | .1361 | .2045 | +.0058 |
| 1 / t0 | .1248 | .1873 | +.0064 |
| 1 / t1 | .1397 | .1996 | −.0065 |

Prediction changes being larger does **not** prove overreaction: conditioning on prediction direction, weighting, clipping and global calibration all matter. The normalized metric sensitivity is small or mixed at the aggregate anchor, rather than a common large positive signal to amplify every update. Raw-code-oriented curves are reported separately; signs of anonymized codes are not physical price directions.

A narrow descriptive lead survives: class3 / t0 has positive model-oriented anchor q in both folds (+.0804 / +.0912;432 /470 rows with nonzero model direction). Class3 / t1 is +.0186 /−.0072. This is an asymmetric conditional-error observation, not a class-specific rule selected for training or evidence that an event gate would improve WP. It should be read alongside the main analysis’s conditional predictability and error mass, rather than promoted alone.

## Selection matters visibly

**Each offset uses a changing population, not one fixed cohort followed through time.** Both masks and the secondary future-quiet condition can change the retained anchors. The mask-overlap plot makes this explicit: among anchor-scored rows, the fraction also scored at the displayed row declines sharply away from0, whereas only roughly6–9% of all available event-window rows are scored.

The authorized secondary joins exact integer cumulative event counts in the existing main table. All9,974 eligible scored endpoint pairs matched;5,271 had no intervening i0 event. This conditions on future quietness relative to the anchor, so it cannot define an online gate or identify a causal response. At positive offsets its mean oriented target increment is zero, but the subset differs from the offset0 population; do not interpret the plotted drop as a target reversal along the same trajectories.

**The quiet +64 tails contain only5 /14 rows per fold (before target-specific weight concentration).** Large positive/negative tail q values are not persuasive mechanism evidence. No confidence interval, independence of overlapping windows, or all-event representativeness is claimed. Original windows can cross additional events; the secondary removes that particular overlap only for its retrospective selected subset.

## Figures

- [Observed-event target/prediction increments](01-event-response.png): solid curves use anchor-and-offset scored rows; dotted curves additionally select future quietness.
- [Metric sensitivity, separate orientations](02-event-gradient.png): raw-code and model directions remain distinct. Solid/dashed mask populations and sparse quiet tails must not be conflated.
- [Mask overlap and sample selection](03-event-mask-coverage.png): the most important qualification when reading the response curves.

Execution completed once in10.53s, maxRSS383,240KiB, under2CPU/4GiB/300s with CUDA hidden. Worker597351 exited; its scope is inactive/dead, and CPU18/19 are released. SourceSHA `5d062953d8ea1dea3062fcad7d9caf6c827dfa697f97636caa62667161dfd95b`; extraction receiptSHA `8a59292164c52b6b0c6d574d9e4824a306598475dd9d1d33b17e57480c9c7973`.
