# E-496 independent source and calibration-resolver review

Review date: 2026-10-06. This review covers prepared materializer/spec source
only. It does not launch calibration, probes, fits, prediction exports, or
scores.

## Verdict

The alpha/horizon and direction blocks pass source/spec review for preparation
only. The fixed and shuffled denominator arms are **NO-GO** until the
calibration resolver proves that the supplied reference mass is the measured
fit-pool estimate, rather than merely a positive scalar with plausible
metadata. Accordingly, the complete E-496 packet does not yet receive an
independent preparation GO.

## Reviewed identity and evidence

| File | SHA-256 |
|---|---|
| `supervision_frontier/DESIGN.md` | `86e17546ac9e2652ef02f8c64287df3ba1acb89c3f3d1bd9e73b61771bb9aa22` |
| `supervision_frontier/materialize_specs.py` | `25e6473d77ece00a58e61b33df82ca45874ecd9974ad09739bf1a51d50f7504b` |
| `supervision_frontier/resolve_calibration.py` | `61752bab070d5841df0b9ca6b0e61d40b675cb872c9249e91eb4906cd5239049` |
| `supervision_frontier/test_materialize_specs.py` | `5bd3b7872aabfef050916ef7997433201d4da8466d5e05971ee428901aa033a3` |
| `supervision_frontier/dependency_graph.json` | `8fc31088cb5111cb972e9f698b622ebff92418f0e8234541090872a38c66b7eb` |
| CPU test receipt | 13 collected cases, 0 failures, 0 errors (root's combined 36-test run) |

The generated dependency graph has 3 CPU calibration nodes, 27 parent fits,
27 heads, 54 predictions, 4 resource-probe plans, and 18 fixed endpoint specs;
`launch_authorized` is false. The source tests cover all six alpha/horizon
cells, endpoint matrices, recipe/denominator structure, stale source/data/cache
and membership rejection, and nested probe-plan binding. The direction target
and weight match the current clipped contrast `(clip(y0)-clip(y1)>0)` and its
absolute clipped magnitude. The CPU calibration recipe strips `--fast`, keeps
`--prop-cache`, sets CPU and zero GPU limits, binds source/data/teacher/cache
and fit-membership identities, and exits before optimizer construction.

## Material blocker: invented mass is accepted

`validate_report()` currently checks that `reference_mass` is finite and
positive, lies inside the reported min/max, and is accompanied by internally
consistent aggregate fields. The report does not include the per-update mass
observations from which these aggregates were computed. I exercised the
synthetic resolver fixture under the one-CPU/2-GiB limit: changing only the
mass to `20.0` and supplying a self-consistent summary (`minimum=0`,
`maximum=40`, `standard_deviation=12`, `zero_mass_updates=0`) caused
`resolve_template()` to write a resolved draft. All source, data, teacher,
cache, fold, and fitting-membership bindings in that fixture remained valid.

Before resolving either fixed or shuffled denominator templates, the report
must carry the measured update-mass observations (or a replayable artifact),
and the resolver must recompute/check their count, finite nonnegative values,
mean, extrema, standard deviation, and zero count against the summary. Add a
negative mutation test proving that a changed scalar and matching summary are
rejected. Keep the report and observation artifact hashes in the downstream
binding. The existing identity checks remain useful, but they do not validate
the estimator output itself.

This finding does not block the independently separable alpha/horizon and
direction drafts once the ordinary controller gates apply. Fixed/shuffled
templates must remain unresolved until the resolver and tests change; any
resolver change invalidates this note's source identity and requires review of
the new hash set.
