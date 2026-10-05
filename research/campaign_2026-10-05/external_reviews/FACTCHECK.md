# Independent CLI reviews: useful ideas, unverified statements

GLM5.3Flash/max produced a blind proposal before receiving the Sol mechanics
audit. DeepSeekV4.1Flash/low then criticized that proposal using the audit's
measured effects. Both final answers are preserved verbatim in separate files;
their agreement is not independent experimental evidence. No hidden reasoning,
credentials, dataset rows or model weights were copied into the repository.

The DeepSeek review has several mathematical and factual errors:

- It says rows are weighted by predicted |y|. The official scorer weights by
  the absolute **observed clipped target**. Predicted mass is only a possible
  internal model output, never a replacement for the official weighting.
- The weighted conditional mean E[|Y|Y|X]/E[|Y||X] does not require a pooled
  learned centering module to become the population weighted-correlation
  optimum up to an affine transformation. Global centering and scale are
  handled by the correlation functional. Prediction clipping and finite
  estimation complicate this, but do not reverse the conditional-mean result.
- The 91% figure is the **difference-mode share of class-3 residual energy**,
  not class 3's population mass. It cannot justify a global class mixture.
- A failed hard-hold predictor does not reject a hazard or event-state model
  that continuously updates its posterior from quiet-row observations.
- Degenerate learned decay is not a scientific falsifier by itself: one useful
  timescale can be predictive. Compare actual fixed-spectrum, no-memory and
  learned-memory predictors under the declared metric.
- Volume/width effects do not prove |Y| predictability or physical book levels.
  Their fitted configurations and target weighting differ.

The GLM review likewise assumes physical labels and column identities not
established by this dataset audit, uses a stale chunk recipe, and proposes a
joint likelihood without fully specifying how per-target weights enter it.

Retain the testable components: multiscale state, content-addressed memory,
small spline nonlinearities, and target-support regularization. Test matched
implementations and nuanced combinations. A review's overbroad rejection does
not authorize closing an entire model family.

OpenCode's two genuine CLI attempts failed with provider HTTP403, including
an actual TUI attempt. No usable OpenCode review was obtained. The failure and
installed catalog are recorded; no provider bypass or paid fallback was used.
