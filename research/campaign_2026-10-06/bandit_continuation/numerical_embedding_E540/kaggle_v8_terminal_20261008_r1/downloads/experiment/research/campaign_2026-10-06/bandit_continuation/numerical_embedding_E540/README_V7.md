# E540 V7 FP32 paired screen

V7 is a new numerical treatment for the frozen E540 pair. It uses FP32 in
both arms with GradScaler disabled. It keeps the 512 selected complete
components, 468 previously exposed evaluation groups, seed 540, clip2-target
WMSE with chunk normalization, the same model shapes and numeric maps, and one
fixed epoch-0 checkpoint. Both profile and fit commands enable --grad-stats;
the measured instrumentation cost is included in profile timing and the
worker's full-fit projection.

V6 remains immutable. Its mounted-input identities and both profiles passed.
The linear FP16 arm attempted 72,996 updates, completed 72,967, and skipped 29.
It therefore failed the strict zero-skip checkpoint contract. The PLE fit and
both prediction exports did not run. V6 produced no paired comparison or grade.

V7 requires both actual T4 profiles to complete 316 FP32 updates, report finite
losses and exactly 316 finite pre-clip GradStats observations, remain within
the reserved-memory limit, and project both full fits and prediction exports
inside the worker deadline. The scaler is disabled by --precision fp32;
--amp-init-scale 1 is an inert argument only and is recorded as not applied.
The full fit must complete exactly the expected update count in both arms with
zero skipped updates and zero nonfinite GradStats steps. The worker separately
checks model/readout checkpoint tensor finiteness and all feature-only
predictions for the frozen 468 groups.

The packet requires an embedded root authorization bound to its code manifest,
recipe, and subset before mounted-data hashing. The current authorization is a
pending placeholder. This packet is code-only preparation: it has not been
uploaded or pushed, and it has no fit, prediction, or grade outcome. Root owns
independent review, provider admission, and grading.
