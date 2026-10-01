# Fifth h128 member: source-only inventory, 1 October 2026

**Defer the exact five-member profile; preserve `student_fixed4.zip` as primary.** No additional training, prediction, metric, packaging or runtime allocation follows from this inventory. The current student4 plus h128 seed5 with matched two-epoch head training has not been tested, and historical small gains do not refute that exact profile.

## Existing assets and missing endpoints

Inspected only existing source/metadata, textual logs/history and directory/file metadata. Remote base `/home/grinchenkoeg/WunderFunder` remained read-only. No checkpoint was loaded, and no label or prediction array was opened, hashed or scored.

For folds0 and1, the base repository contains:

- `artifacts/runs/refs5_f{0,1}/model_ep3.pt`: distinct seed5 h128×2 parents, four training epochs, EMA evaluation. Raw configuration logs are `artifacts/logs/refs5_f{0,1}.log`; their parent recipe uses train+other-valid, EMA1/.999, propensity and since32/.25 weighting.
- `artifacts/runs/ft128s5_f{0,1}h/model_ep0.pt` and `model.pt`: **one-epoch** head-only children, gcorr, mask weighting, frozen `inp,gru`, lr5e−4, auxiliary coefficients .3/.3 and seed5. Their logs are `artifacts/logs/ft128s5_f{0,1}.log`. Existing [M01 provenance](../../2026-09-30-creative/M01-checkpoint-provenance.json), lines187–203 and225–240, independently records the parent identity, one-epoch schedule and cache aliases.
- `cache/pred_ft128s5H_f0_valid.npy` and `cache/pred_ft128s5H_f1_valid.npy`: existing OOF paths, each **299,680,128 bytes** by file stat. Both paths exist under the base repository and `/home/grinchenkoeg/WunderFunder-creative`. This check establishes presence/size only, not a new content-integrity or scoring verification.

No seed5 two-epoch head child/OOF or FULL final seed5 parent/H2 endpoint was found in the inspected base/creative run and cache inventories or final-run configurations. Final ordinary h128 runs include `fin128_s3[ h / h2 ]` and `fin128_s4[ h / h2 ]`, but no `fin128_s5`/`fin128_s5h2`; creative `on_ctl128s*` has seeds3/4 only. The separate `ft128s5_f{0,1}` runs are the subsequent full-network FT stage, not a second head-only epoch.

H2 is a fresh two-epoch schedule from the same parent epoch3, **not** an extra epoch appended to the old H1 child; see [training audit](../../2026-10-01-deep/training.md), line103. Existing H1 caches therefore allow only an explicitly mixed H2/H1 exploratory profile. They do not supply the exact requested H2 profile or a deployable final seed5 artifact.

## Historical counterevidence and its limits

The [September journal](../../../quant-probes-2026-09-29.md) records:

- Lines138–146: the pure h128 seed curve gains about+.00007 from four to five members and+.00005 from five to six, averaged over the corresponding subsets. This is evidence of diminishing returns for that older reference recipe.
- Lines194–201: one-epoch head-FT h128 pair score .70107 and seed3/4/5 triple .70124. This is an older head-FT comparison, not student4 plus a fifth member.
- Line271: adding more plain h128 seeds to old B changes its score by approximately+.00002. That sentence does not establish a fully specified matched-H2 five-member student test.

The current failed double-wide **replacement** likewise does not refute adding a cheaper h128 while retaining the useful h128 seed4. No exact historical test of `student4-H2 + h128s5-H2` was identified.

## Decision at closure

Adding one h128 raises the model-only dense MAC count from1,348,736 to **1,605,376**, still below strong4's1,741,952. This makes runtime feasibility plausible; it does not establish an actual deadline margin for a five-member native package. Extra member state, nonlinearities, memory access and callback behavior are not certified by MAC arithmetic or another package's runtime.

A faithful continuation would require at least two fold H2 fits and their OOF generation, plus a FULL final seed5 parent and H2 fit, followed by native parity/quality and runtime qualification if promoted. That is more than a single cached fixed-profile readout. With no matched final asset and only weak nearby historical marginal gains, preserve the completed primary artifact and defer this untested profile without labeling it scientifically disproved.
