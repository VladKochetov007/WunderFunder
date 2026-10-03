# Puzzle A: a path-law falsifier, with an explicit unresolved denominator

**None of the three frozen laws explains the outside target pair.** The trailing
extrema rule matches many interior pairs, but its loser return is algebraically
incapable of matching every mapped outside loser in this sample. The exact
generator remains unidentified; a hybrid winner-extremum/loser-execution law is
still open. This is TRAIN diagnostic reconstruction, with no forecasting gain.

## What was frozen and actually run

PREREG.json preceded the synthetic generator, decoder fit and test scoring. The
three representatives were fixed-extrema over the next 16 quote events;
independent first +/-2-tick entry-return barriers with a 32-event cap; and
independent one-tick reversals of running bid/ask extrema with a 32-event cap.
Trailing extrema include the entry bid/ask and return the extreme, not the
execution quote. There was no horizon/threshold sweep, family addition, or
class-specific law selection. These are specific representatives, not tests of
every imaginable stopping law.

The old immutable reconstruction panel supplied 64 fit groups and the first 12
test groups, with disjoint original paired-component assignments. Membership
was fixed before new target reads. This panel was previously exposed to
reconstruction; all TRAIN had also been exposed to my earlier target-summary
scan. It is not pristine confirmation or a new component-independence audit.
No validation labels, model checkpoints, external data or live source were used.

A new decoder was fitted on the 64 TRAIN fit groups only: 1,026 accepted current
code-to-class/spread identities and 1,716 directed movement identities. Interior
label differences supervise the movement table, assuming a stable common
reference. Physical quote identity remains unproven. Unlike the old whole-segment
modal anchor, this decoder anchors at the first known spread in the input prefix.
Unknown moves and inconsistent spread/class transitions split the path. No gap
is bridged. Fit rounded-code aliases: zero.
The frozen input-derived spread agrees with 922/922 mapped held-out interior
origin sums; 26 interior origins lack that map. This is a coordinate check,
not evidence that outside movements admit the same common future reference.

Events are the pre-existing a6 float32 endpoint. Origins >=99 are chosen every
fifth event, capped at 128 per sequence by input-only evenly spaced indices.
All chosen origins remain in denominators, including missing classes and broken
paths. Scale is a fixed prior TRAIN quantum selected by the frozen current-code
class map; no test-target class or scale was fitted. Maximum mapped test lattice
residual was 0.00016293 raw units. Identity path copies and two prefix replays per
test sequence passed. Known held-out interior moves: 17,410/17,410 exact, conditional
on eligibility; this does not validate all outside moves or identify physics.

There were 7,822 fit calibration origins and 1,393 test origins. Six affine
coefficients per family were fitted on complete fit origins only, then frozen
before test reads. This secondary calibration produced zero exact test pairs
for every family and supplies no generator-identification claim. No ridge gain
or competition-metric improvement was sought.

## Known-generator and adversarial tests

synthetic.py produces 256 integer random-walk quote paths and known labels for
each of the three families. Each law passes 768 comparisons apiece against an
independently written scalar reference, identity copies, price translation and
bid/ask reflection. This reference was written by the same actor; controller
independent reproduction is still required. Cross-family exact confusion is
1/256 fixed versus barriers, 7/256 fixed versus trailing, and 0/256 barriers
versus trailing. Diagonals are 256/256.

An adversarial pair has equal 16-event extrema but a different early reversal:
fixed-extrema returns [3,-1] for both; trailing returns [3,-1] versus [1,-1].
The initial nondiscriminating order fixture was corrected to introduce the
intended early reversal; its original output is retained. No data law changed.
A second adversarial pair has identical first 32 future events but diverges
later: a long trailing extreme would return 31 versus 34. The bounded engine
correctly leaves that side censored in both. Identical finite prefixes cannot
identify an unbounded stopping law. Unknown-gap fixtures retain unresolved sides.

## Measured pair reconstruction

All exactness uses both raw targets within 0.002. Weight is the sum of absolute
clipped origin targets times that selected origin's plateau length, used only
for diagnostic mass accounting. It is not Global WP. Origin count is the primary
sampling unit here; no independent-row p-values or bootstrap claims are made.

| Frozen law | Complete / all 1393 | Exact pairs | Exact / complete | Exact weight / ALL origin weight |
|---|---:|---:|---:|---:|
| Fixed 16-event extrema | 1057 | 336 | 31.79% | 8.61% |
| Competing +/-2 barriers | 1003 | 0 | 0% | 0% |
| One-tick trailing extrema | 1007 | 584 | 57.99% | 20.03% |

All 336 fixed and all 584 trailing exact pairs are interior. Outside origins
number 445; complete outside pairs are 302, 279 and 260 respectively, with ZERO
exact pairs for all three laws. Fixed matches the outside winner alone on 161/302;
trailing on 149/260; neither matches any eligible outside loser. Barriers match
41/279 winners and 2/279 losers but no complete pair. The apparently promising
58% trailing number therefore does not explain the economically heavier outside
geometry. Its weighted raw sum MAE is 1.139 versus 0.952 for fixed extrema, so
more exact low-weight pairs also do not establish a better approximate mapping.

Matched common-complete-origin controls (aligned exact versus control exact):
fixed shift997: 310 versus139 on938; fixed shuffle:311 versus136 on925;
trailing shift997:537 versus227 on892; trailing shuffle:520 versus241 on857.
Barriers:0 versus0 on756 for shift,0 versus1 on751 for shuffle. Aligned extrema
have a real conditional association with labels, but that is insufficient for
an exact target law. Full-denominator controls and all reason masses are saved;
common-coverage comparisons never replace the full panel.

## Unknown, missing and censored mass

| Law | Unknown transition | Horizon cap | Sequence end | Missing current class | Resolved pair |
|---|---:|---:|---:|---:|---:|
| Fixed |264|0|18|54|1057|
| Barriers |131|196|9|54|1003|
| Trailing |213|109|10|54|1007|

Corresponding label-weight shares for fixed are 8.287%,0%,2.651%,1.168%,87.893%;
barriers 5.779%,4.437%,0.390%,1.168%,88.226%; trailing
7.490%,6.609%,0.390%,1.168%,84.342%. Categories are a declared priority for pair
unresolved reasons; per-side reasons and lags remain in sampled_origins.json.
An absent known hit does not mean an absent physical hit. The decoded sample
cannot establish full generator coverage.

E-317's winner-extremum clue survives conditionally: among 417 mapped winner
sides, 241 exit-side target levels were touched within the known bounded path;
all 241 were running extrema at first touch. The remaining176 did not have a
known touch within32; another28 outside origins lack current spread/class.
Target-implied levels are diagnostic labels only and never model inputs.

## Horizon-independent negative result

BOUND_PREREG.json froze an adaptive algebraic check before its count. If the long
running maximum includes entry bid, its returned payoff is >=-entry spread.
The short running minimum including entry ask obeys the same bound. This holds
at any stopping time, for any reversal threshold, even beyond our 32-event cap.

417/417 mapped outside losers fall strictly below that input-derived bound;
28/445 outside origins have missing spread and remain unknown. Synthetic known
trailing labels have zero bound violations. Thus extending a horizon cannot
rescue this extrema-return family, conditional on the supervised spread map.
Example: TRAIN group1131, seq239686751, step750 has fit-code class3, spread1,
minimum extreme return -0.65687, supplied loser -1.970603. Execution-price
stopping and hybrid winning-extremum/losing-execution mechanisms are not rejected.
The bound is the stronger reusable falsifier; zero eligible outside exact pairs
was not merely a censored-path artifact.

## IDs, provenance, failures and cost

Test (group/component/seq):1131/1695/239686751;8488/462/1734866141;
10038/2614/2040244411;4094/1826/836756439;8497/732/1736263665;
6489/746/1333389311;7758/1245/1589015502;6457/2268/1326024571;
1473/388/310399649;1765/637/370750816;5635/1754/1159057298;
8477/991/1732301769. Exact fit IDs/hashes are in fit_results.json; all test
sampled steps, per-sequence input/target hashes and components are in results.json.
The frozen source panel/hash, decoder table hash and calibration hash are saved.
Source context records E-273 events, E-303 current-state failure and E-317 matured
feedback leak/remaining stopping clue. Requested E-269/270 ledger entries concern
model merges/heads rather than a target-path generator; no extra baseline
inspection followed them. Old reconstruction failures were not used as new gains.

Two NumPy-integer export errors and the initial lint errors are preserved. Export
recovery reused byte-identical calibration and regenerated byte-identical raw
sampled outputs. It changed no law, scale, horizon or cohort. Final scripts pass
ruff and synthetic verification. The coverage figure was inspected; it explicitly
shows zero outside exact matches. Raw NaN diagnostics are preserved, with strict
JSON/null companion artifacts for reuse. No resource failure occurred.

Every computation used run_cpu.sh round2-firstpassage, one core/2GiB, CUDA empty.
Fit:5.13s wall,5.11s CPU; verified fixed test replay about3.05s wall. Peak reported
RSS about1,058,512KiB (~1.01GiB). Initial calibration/export costs did not survive
as a timed result; role-filtered controller logs provide rounded scope wall
estimates including failed commands. cost.json states those limits explicitly.
No GPU time, installs or live edits. COMMANDS.txt is the runnable reproduction
sequence; PROVENANCE.json binds final files. This actor does not approve promotion.

## Ranked long-term follow-ups

1. **Hybrid pair law:** predeclare a winning running-extremum return and a losing
   execution return, with one competing stopping event and one fixed barrier.
   The matched control returns extrema for both sides. Kill on outside-pair
   transport failure, explicit loser-side mismatch or comparable shifted paths.
2. **Decoder falsification:** generate anonymous relative codes from synthetic
   paths with changing common future references; fit the same interior-difference
   decoder and compare inferred moves to known true moves. This can quantify the
   common-reference contamination that conditional held-out exactness misses.
3. **Order-preserving-range adversaries:** compare order-shuffled event paths
   holding entry quotes and extrema fixed. First use known generators; then a
   frozen supplied-data panel. Kill a proposed stopping rule if the advertised
   order observable fails to distinguish it from range-only alternatives.
4. **Transport with censoring:** controller chooses components excluded from all
   decoder fit/previous mechanism tests; keep all origins and report family
   identifiability by fit-input-derived class. Reject claims whose apparent
   coverage depends on selecting decodable prefixes. Do not fit eval scales.
5. **Prefix-only representation:** after an independently reproduced law, use
   known past moves, causal running swings, event age and decoder-unknown flags
   as inputs; future stop quantities can be TRAIN supervision only. Compare a
   fixed predictor with/without those inputs under equal training cost. No actual
   target feedback, future extrema, future path or stop labels enter inference.

No GPU contrast proposed yet: the exact outside-pair mechanism and decoder
interpretation remain unresolved. These are the next bottlenecks, not ridge gain.
