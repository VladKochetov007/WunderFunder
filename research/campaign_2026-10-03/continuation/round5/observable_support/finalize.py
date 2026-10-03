from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

OWN = Path(__file__).resolve().parent
REPO = OWN.parents[4]


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1048576):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    """Freeze the two-test audit, with no additional data tests or fits."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = json.loads((OWN / "results.json").read_text())
    for name, expected in result["input_sha256"].items():
        assert sha(Path(name)) == expected
    panel_path = OWN.parent.parent / "round4/peer_innovation/samples.json"
    panel = json.loads(panel_path.read_text())
    assert (
        sha(REPO / "src/wnn/features.py")
        == panel["control"]["source_sha256"]["src/wnn/features.py"]
    )
    seq = result["sequences"]
    total = result["summaries"]["all"]
    fit = result["summaries"]["fit"]
    test = result["summaries"]["test"]
    pure = sum(s["coordinate_motion"]["exact_uniform_increment_rows"] for s in seq)
    events = sum(s["coordinate_motion"]["any22_change_rows"] for s in seq)
    maximum_clock = max(s["packed_peer_clock_max_abs"] for s in seq)
    static_error = max(s["gauge_controls"]["static_relative_shape_max_abs"] for s in seq)
    dynamic_error = max(s["gauge_controls"]["dynamic_relative_shape_max_abs"] for s in seq)
    gauge_scalar = sum(s["gauge_controls"]["dynamic_scalar_clock_changed_rows"] for s in seq)
    gauge_top = sum(s["gauge_controls"]["dynamic_top_pair_clock_changed_rows"] for s in seq)
    time_controls = sum(s["gauge_controls"]["dynamic_shift_boundary_rows"] for s in seq)
    scalar_mismatch_ids = [
        r for r in result["all288_query_clock_IDs"] if r["scalar_age"] != r["any22_age"]
    ]
    records = []
    for s in seq:
        records.append(
            f"{s['component']:5} {s['seq_ix']:10} {s['stratum']:4} {s['query_counts']['scalar_vs_any_age_disagreements']:3} {s['units']['scalar_nonzero_step_RMS_normalized']:.6f} {s['coordinate_motion']['common_projection_energy_fraction']:.6f}"
        )
    report = f"""OBSERVABLE SUPPORT — TWO INPUT-ONLY TESTS, NO EXPANDED PROXY FIT

Narrow findings: the peer scalar clock differs from its block clock, but not at
any original age-zero query in this panel. The production top-pair clock and
any22 clock coincide on all audited post511 rows. Thus the implementation
assumption is empirical on this panel, not a universal encoding identity, and
it does NOT explain away the previously failed freshness-stratum gate.
Common-coordinate gauge controls preserve relative shape while changing scalar
innovation; this limits physical interpretation, not validity of predicting
an explicitly defined encoded observable. No fits, labels, masks or GRU forwards.

PREREGISTRATION / SCOPE / PROVENANCE
PREREG.json froze exactly TWO invariant/control tests and fit/test/sequence
metadata strata before array reads. Existing24 matched_usage stored sequences,
all288 inherited queries, plus all rows>=512 (467,712 rows). These rows are an
INPUT audit domain, not original scored-position domain or a new target census.
Only NPZ members raw, packed and std were accessed. No Y, mask, reference,
response, future proxy outcomes, GRU states or raw parquet were loaded. Original
query IDs and16fit/8test split come from frozen metadata; prior mask-selected
panel exposure remains inherited and is not called fresh confirmation.
One sequence at a time, row order retained; no shuffle, resampling, channel,
filter, horizon or normalization fit. Source-array hashes equal their original
manifest; input files and production feature source rehashed unchanged. The
source SHA equals the frozen panel's production feature hash.

TEST1: SCALAR / TOP-PAIR / ANY22 FRESHNESS
Operational anonymous coordinates: scalar=raw52; top-pair=raw52 OR raw63;
any22=ANY raw52..73 difference from immediately previous row. Every clock begins
at row0 and counts rows since its own last event. Production ev_i1px=packed183
uses TOP-PAIR, not ANY22. Its since_i1px=packed188 is log1p(min(age,10000)).
Top-pair event flags equal packed183 EXACTLY in all24 arrays. Independent clock
log recreation maximum discrepancy {maximum_clock:.9g}, below the predeclared
2e-6 float32 NumPy/Torch transcendental allowance. Clock definitions were not
changed after seeing values. The ordered toy changes a deeper coordinate only,
then second top only, then scalar: flags are respectively ANY-only, TOP+ANY,
and SCALAR+TOP+ANY. It falsifies universal equivalence of these encoding clocks
without reversing or randomizing time.

ALL post511 rows: scalar events55,562; top-pair/any22 events60,358.
4,796 top/any events occur while scalar is idle. Scalar and any22 AGES disagree
on40,448/467,712 rows. Top-pair vsany22 ages disagree on ZERO of these rows.
Thus observed block updates are represented by at least one top coordinate;
scalar-only event age misses some second-coordinate events. This does not identify
instrument prices, order-book physics or an informationally sufficient clock.

ALL288 metadata queries: scalar/top-pair/any22 events all59, no fresh-block/
scalar-idle query. Scalar vsany ages differ on24 queries (fit13/192,test11/96),
while top-pair vsany ages differ on0. Complete IDs/clock values retained in
results.json; mismatching age IDs also in scalar_age_mismatch_IDs.json.
Test subset's fresh counts are17 for ALL three definitions, same17 origins;
age0/age1+ memberships in the failed peer-innovation test are therefore unchanged
if scalar or production top-pair freshness replaces ANY22 freshness at those
queries. This audit does not rescue that closed branch or propose such a refit.
Off-query scalar-vsblock mismatches do not become evidence of predictive gain.

Fit post511 rows311,808: scalar events35,644 vsblock38,334;2,690 block-only events,
scalar-vsblock age mismatches28,096. Test rows155,904:19,918 vs22,024 events,
2,106 block-only events, age mismatches12,352. Counts are descriptive within the
fixed sequences; no row-IID or population interval is asserted.

TEST2: COMMON-COORDINATE MOTION / FROZEN STD / GAUGE CONTROLS
Peer increments use original float32 values represented exactly as float64 for
algebra. The fixed common projection is the arithmetic mean of22 increments;
residual is increment minus that mean. No loading/vector/sign was fitted.
Across ALL original sequence transitions, exact equal-common nonzero increments:
{pure}/{events} any22 event rows. Common-projection energy fraction per sequence
ranges {total["common_projection_fraction_range"][0]:.6f} to
{total["common_projection_fraction_range"][1]:.6f}. This fixed basis is not a
learned latent common factor. Unequal coordinate loadings/nonlinear encodings can
produce a physical common move without an equal-coordinate translation. These
statistics neither identify actual recenter operations nor rule out latent factors.

All24 std arrays are identical. Scalar52 frozen std={seq[0]["units"]["frozen_std52"]:.9g},
a GLOBAL checkpoint feature scale rather than per-sequence tick, physical unit,
volatility gauge or observability guarantee. Nonzero scalar step RMS/std ranges
{total["sequence_nonzero_step_RMS_range"][0]:.6f} to
{total["sequence_nonzero_step_RMS_range"][1]:.6f}; sequence-equal median
{total["sequence_equal_median_nonzero_step_RMS"]:.6f}.
Fit range={fit["sequence_nonzero_step_RMS_range"]}; test range=
{test["sequence_nonzero_step_RMS_range"]}. Test RMS range is contained in fit range;
this simple support descriptor does NOT reveal a gross RMS-unit range failure
explaining the prior negative result. It does not test full distributions, joint
covariate support, latent units/loading or predictive stationarity. Prefix512
versus later RMS and per-sequence quantiles are recorded, not used to route,
standardize, fit or select a representation.

Matched algebraic controls on EVERY sequence, original time/order unchanged:
- Static common translation+8 to all22 peer coordinates: innovations and ALL
  three clocks invariant; relative-shape max error {static_error:.9g}.
- Deterministic causal common translation .125*floor(row/32): same relative shape
  max error {dynamic_error:.9g}, but scalar innovation changes at ALL
  {time_controls} predeclared shift boundaries. Scalar-clock ages change on
  {gauge_scalar} rows, top-pair clock ages on{gauge_top} rows. No future value
  defines a shift and there is no outcome/displacement-matched control selection.
These float64 controls compare encoded observable algebra, not new float32 model
inputs or realistic counterfactual market histories. The unchanged relative shape
cannot identify whether an unobserved common reference or actual joint motion
caused an encoded scalar change. Conversely, in a FIXED supplied encoding the
scalar innovation remains a well-defined causal input and future proxy: gauge
sensitivity alone does not make it leaked, useless or physically interpretable.
No future payoff, target sum or quote decoder enters either test.

SEQUENCE SUPPORT / IDS (mismatching query ages, nonzero-step RMS, common fraction)
component seq_ix      split mismatches normalized_RMS common_energy_fraction
{chr(10).join(records)}
All original288 queries are retained including the prior end-censored19991 query;
future availability is irrelevant to this input-only audit. This is not a larger
proxy panel or selected successful-coordinate subset. Exact fit/test metadata
and source files are included in results.json hashes.

DECISION / WHAT THIS DOES AND DOES NOT JUSTIFY
Keep the closed peer-filter result closed. A universal scalar/any/top clock
identity is falsified by the toy and scalar-vsblock differences; the actual
ANY22-to-production-clock substitution holds on this fixed panel, and fresh
query strata coincide. Use explicit clock names when interpreting that evidence.
Uniform-common-coordinate physical semantics remain unidentified. Fixed std
and contained RMS ranges do not establish full support or physical units.
No newly identified nuisance invalidates the old rejection or justifies expanded
proxy fitting. A later controller-selected branch would need a distinct stated
observable mechanism/support requirement rather than a new horizon/filter sweep.
This ONE audit stops here; no additional variants or representation are promoted.

COST / REPRODUCTION / FROZEN OUTPUTS
Audit body{result["cost"]["wall_seconds"]:.3f}s wall,
{result["cost"]["cpu_seconds"]:.3f}s CPU, peak{result["cost"]["peak_rss_kib"]}KiB.
Only CPUwrapper reviewer1core2GiB900s; no failure/GPU/live edit/install. Audit,
format/lint and finalization bodies ran serially, never >1 active body here.
Reproduce with run.sh; scripts, all IDs, measured outputs, logs, STATE and SHA
provenance in this directory only. Actors preserved. No public-gain, predictive
necessity or physical-law claim. Report frozen for controller review.
"""
    (OWN / "REPORT.txt").write_text(report)
    save_ids = {
        "scope": "all original288 input-only queries; scalar age differs from any22 age",
        "IDs": scalar_mismatch_ids,
    }
    (OWN / "scalar_age_mismatch_IDs.json").write_text(json.dumps(save_ids, indent=2) + "\n")
    state = json.loads((OWN / "STATE.json").read_text())
    state.update(
        phase="report_frozen",
        test_count=2,
        novelty_budget_closed=True,
        scalar_any_clock_query_strata_rescue=False,
        production_any_clock_empirical_equivalence=True,
    )
    (OWN / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    (OWN / "COST.json").write_text(json.dumps(result["cost"], indent=2) + "\n")
    manifest = {
        "input_sha256": result["input_sha256"],
        "own_artifact_sha256": {
            p.name: sha(p)
            for p in sorted(OWN.iterdir())
            if p.is_file() and p.name not in ("PROVENANCE.json", "finalize.log")
        },
        "scope": "exactly2 input-only tests; original stored raw/packed/std only",
    }
    (OWN / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logging.info(
        "Frozen: exact-common rows%s/%s; clock mismatch queries%s; no fit.",
        pure,
        events,
        len(scalar_mismatch_ids),
    )


if __name__ == "__main__":
    main()
