# H230: heterogeneous distillation with an independently trained wide baseline family

This is a separate interaction arm, not an exact replication or a modification of frozen
E506/E520. User wants useful components even when individually weaker and independent
routes around cloud transport stalls. The E521 scratch raw GRU384 seed3 checkpoint is
fit/export/independently graded; seed4 is training. These can replace the missing compact
baseline family in a declared teacher variant once both own checkpoints/exports exist.

Basis: Maxim-inspired heterogeneous recurrent teachers, unnormalized weighted main
sum0.8125, four-family auxiliary coefficient0.1, alpha0.7. Keep strong/full/raw families
and all existing coefficients; only baseline sources change from planned GRU128 pair to
own GRU384 pair at0.03125 each. No teammate weights. Preserve the original raw/clipped
family-target discrepancy explicitly: use reviewed persistent familyloss implementation
commit dde32d0f and `--distill-family-target-mode raw` for the leading variant; the older
clip2 recipe remains separate. Root running trainer/source must not be edited.

Dependencies only for this arm:
1. Complete E521seed4 fixedep3; both teacher train+valid predictions and provenance.
2. Audit every source/initializer/propensity ancestry against evaluated fit-excluded
   fold0 groups. Existing old checkpoints must still satisfy that exact ancestry.
3. Seal this new manifest with all actual hashes, build only necessary teacher caches
   in bounded chunks with measured disk/RAM headroom. Draft null hashes cannot launch.
4. Freeze exact student commands, twoepoch/fixedep1, same initialization/optimizer/masks/
   lossnormalization as declared reference. Independently audit source/objective.
5. Actual fullshape profile for extra family outputs/cache consumption; admit bounded
   fit/export on local or Kaggle with a result owner through independent grading.

Compare equal ownstudent seedpair against existing audited C1 pair and warmparents,
with comparison differences stated. For a family-head effect specifically, use matched
sameheterogeneous main target without familyhead; do not call C1 a family-head ablation.
Measure complete served/student endpoint and costs, not parent-additive gains. Exposed
fold0 is development; no publicforecast from localdeltas.

This arm need not wait for E520's compact baseline for validity: different baseline
capacity is intentionally part of its recipe. E520 still owns the faithful planned
analogue and matching width comparison. H230 is not a reason to cancel healthy jobs.
No sourcecache, studentfit, computeallocation or score is admitted by this draft.
