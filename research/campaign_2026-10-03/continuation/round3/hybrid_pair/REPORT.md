# Hybrid pair continuation: a narrow conditional match, two rejected exact laws

**Neither frozen representative reconstructs the target generator.** Independent
reversals match zero complete target pairs. A shared profitable-reversal exit
matches 74 outside pairs, but fails the other 186 computable outside pairs and
most interior pairs. The matches concentrate on historically common offset-two,
small-winner geometry. There is no inference/training proposal or physics claim.
The independent decoder skeptic's files have not been read.

## Frozen rules and resources

PREREG.json preceded all synthetic and supplied-data scoring. Exactly TWO new
representatives and the old extrema-only matched control were tested. Every rule
uses the same first one-tick reversals and the fixed32-event cap. No tuning,
scale/class fitting, affine adjustment, offset-two insertion or horizon sweep.

1. **Independent hybrid:** each side uses its own first reversal. A strictly
   positive running-extreme return is returned at the extreme; a zero/nonpositive
   extreme is returned at that side's executable stop quote. Both stops required.
2. **Shared profitable exit:** select the chronological first profitable reversal
   among the two sides' first stops, return its running extreme and the opposite
   executable return at THAT SAME event. If both first stops are nonprofitable,
   select the earlier stop (declared extreme/tie rules apply), returning that
   side's extreme/opposite executable quote. This fallback requires both first
   stops known; no profitable candidate or fallback can cross an unknown step.
3. **Extrema-only control:** original independent reversal extremes for both sides,
   including entry quotes, unchanged from round2.

Profitability and selection use the decoded path only, never the supplied target
winner. Frozen TRAIN code class/spread and prior TRAIN quanta determine scale.
The decoded coordinates are supervised and conditional, not identified physical
quotes. Unknown moves, spread inconsistencies and unanchored origins remain
unresolved. If an unknown step precedes a prospective stop, scanning terminates
and later known islands cannot repair it.

The exact previously exposed12-component TRAIN panel and1393 origins were reused:
445 outside and948 interior. Component assignments come from the original frozen
panel; this is not fresh independent confirmation. All raw arrays/IDs/steps/labels
agree with round2 hashes. No validation, checkpoints, new decoder fit or baseline
inspection. Every computation used run_cpu.sh round3-hybrid-pair, one core/2GiB,
CUDA empty; no GPU/install/live trainer/queue edits or writes to round2.

## Known-generator and adversarial evidence FIRST

256 seeded integer random-walk quote paths produce known labels from a separately
written scalar specification. Engine/reference, identity, translation, reflection
and unknown-gap checks each pass768 cases (three laws times256 paths). All256
complete known-family positives reconstruct exactly on their own generator.
Cross-family exact matches: independent/shared31/256; independent/extrema0/256;
shared/extrema113/256. These overlap rates establish non-equivalence, not physics.

Same-range paths [0,2,4,2,0] and[0,2,1,4,0], padded through32, give:
independent [3,-3] versus[1,-3]; shared [3,-3] versus[1,-2]; extrema-only
[3,-1] versus[1,-1]. All depend on event order. The initial non-equivalence
assertion checked fixture A, where the hybrid laws agree; it was corrected to
predeclared fixture B BEFORE any TRAIN read, with no law/fixture/parameter change.
The traceback is preserved. Monotone caps stay unresolved; a first-future-event
unknown followed by valid events yields NO finite outputs. Unknowns are not skipped.

## Raw reconstruction and residuals

Both raw target errors must be <=.002 for a pair match. All selected origins stay
in denominators. Label weight is abs(clipped targets) summed over both sides times
that origin plateau length, for diagnostic coverage only; it is not Global WP.

| Rule | All exact /1007 complete | Outside exact /260 complete | Interior exact /747 complete |
|---|---:|---:|---:|
| Independent hybrid |0|0|0|
| Shared exit |129|74|55|
| Extrema-only |584|0|584|

Outside shared exactness is74/445 ALL outside origins (16.63%), or74/260 complete
(28.46%). It covers27.89% of all outside origin label weight. Counting winner-side
matches alone would be misleading: both hybrids match149/260 outside winners,
independent matches0 losers, shared matches86 losers, but only74 complete pairs.

Unweighted raw MAE (t0,t1), outside/interior respectively:

| Rule | Outside MAE | Interior MAE | Outside RMSE |
|---|---|---|---|
| Independent |(.7571,.6756)|(.6516,.6433)|(1.1713,1.0526)|
| Shared |(.7460,.7086)|(.3423,.3868)|(1.1715,1.1080)|
| Extrema-only |(.9600,.8648)|(.0868,.0868)|(1.2389,1.1303)|

Complete-case outside sum/difference MAE: independent(.9837,1.4327),
shared(.7326,1.3828), extrema(1.0034,1.8248). Shared and independent improve some
outside raw residuals while worsening interior reconstruction, which does not
establish a better predictor or an exact law. Per-origin signed raw residuals,
null missing sides, selected events and required-prefix lags are saved in
sampled_outputs.json. No post-score corrections were applied.

## Controls and denominators

On identical common-complete OUTSIDE cases, shared aligned/negative exact pairs:
shift99766/5 on229 origins; shuffled65/5 on211. Independent gives0/0 in both.
All-origin and law-specific coverage controls are saved separately; paired
complete comparisons do not replace ALL-origin mass. Interior shared alignment
is weak: shift48/47 on663, shuffle48/55 on646, versus extrema-only584/747.
This kills a strong interior explanatory claim. No independent-row p-values.

All three rules happen to have identical aligned eligibility here:
1007 complete,213 unknown-transition,109 cap-censored,10 sequence-end,54 missing
origin class. Corresponding total-label weight shares are84.342%,7.490%,6.609%,
.390%,1.168%. Outside:260 complete,98 unknown,56 capped,3 ended,28 unmapped;
weight81.302%,8.923%,8.192%,.437%,1.146%. Interior:747 complete,115 unknown,
53 capped,7 ended,26 unmapped. Per-side reasons remain explicit; the pair summary
uses priority missing/unknown/end/cap. Censoring is not failure-to-hit in reality.

5572 extrema-control checks (1393 origins times4 path conditions) agree exactly
with frozen round2 predictions, lags and reasons. Identity copies agree for both
new rules. Every finite pair was verified on a wholly known required prefix;
no reconstructed outside pair depended on skipping an unknown transition. Two
input-prefix replay cuts per sequence also agree. Raw source arrays are identical
to the parent selected-data hashes. Scoring/source-format replay changes no outputs.

## Deduplication narrows the claim

E-272 already contradicted fixed offset2 generally, especially classes0/1.
E-302's clipped first-adverse law was partial with large unknown mass. Neither
formula was used here. DEDUP_PREREG.json froze an adaptive descriptive check
before computing diagnostic winner W and excess losing offset K from the frozen
input-code spread/quantum and supplied labels. No third law was added.

ALL74 shared outside matches have K=2:70 have W=1,4 have W=2. None of85 eligible
outside origins with W>=3 match. Even W=1,K=2 is not universal: many mismatch.
The260 computable outside origins contain258 K=2 and just2 K=4; neither K=4
origin matches. Thus this exposed decodable panel barely tests the non2 tails
that broke the historical fixed-offset law. Partial matches cannot revive it.
Class-derived outside exact/complete/all counts:
class0:0/2/35; class1:3/11/52; class2:8/91/147; class3:63/156/183;
unmapped:0/0/28. Missingness strongly favors the easy recurring coordinate classes.
Exact and failed witness paths, input-derived classes and component counts are
in summary.json. The appropriate conclusion is a small-winner shared-event
association in this coordinate, not a new general payoff formula.

## IDs, hashes and cost

Unchanged test group/component/seq IDs:
1131/1695/239686751;8488/462/1734866141;10038/2614/2040244411;
4094/1826/836756439;8497/732/1736263665;6489/746/1333389311;
7758/1245/1589015502;6457/2268/1326024571;1473/388/310399649;
1765/637/370750816;5635/1754/1159057298;8477/991/1732301769.
All selected steps and raw-array SHA256s are in results.json. Frozen table SHA256
43d9d1e256e9ede1a06ddfd58429ff3c3cdf50ccf815ec07a677c977d70467bb
is checked before/after every raw replay, along with frozen decoder, selection
and parent sampled outputs. source_hashes.json and PROVENANCE.json bind final
source/artifacts; PREREG and dedup hashes are recorded in measured results.

Known-generator run ~.15s; initial raw scoring2.63s wall/2.75s CPU, peakRSS
1,077,080KiB (~1.03GiB); subsequent unchanged-panel witness/source replays have
separate measured costs. cost.json and role-filtered controller logs preserve
these costs; no resource failure or GPU use. Synthetic assertion/lint failures
and initial scores are retained. Final lint and synthetic checks pass. All
reproduction commands are in COMMANDS.txt. All writes are in this round3 branch.

## Decisions and ranked next steps

Both exact-law claims are rejected, not merely uncertified by unknown mass.
Conditional shared-event association survives the shift controls but is heavily
restricted by W/K geometry. No generator identity follows from74 perfect
coordinate-based matches. No prefix-only training representation is proposed
now: both reports must freeze and the controller must compare the independent
skeptic's decoder assumptions before any representation/full-training proposal.

1. **Audit-gated quote coordinate:** compare the frozen independent decoder audit
   after both reports freeze. Kill physical/stopping interpretation if common
   future-reference contamination can explain the inferred excursions. Preserve
   supplied-coordinate identities separately from physical inference.
2. **Rare-tail identifiability:** freeze a component panel without decoder-success
   selection, with input-derived class strata and ALL unknown mass. Require a
   proposed loser execution law to transport on non2 tails; no eval-scale fitting.
3. **Exit-event identifiability:** on validated coordinates, predeclare a single
   event selector that must distinguish W>=3 episodes from equal-range reorder
   controls. Existing shared reversal is the required control. Kill on paired
   mismatch or comparable shifted paths; do not optimize horizon by scores.
4. **Interior/outside compatibility:** demand one law explain both strata under
   fixed parameter count. This pass traded584 interior matches for55 without
   solving outside tails. Any future branch selector must come from path inputs,
   never supplied target signs or labels used for mature feedback.

GPU/live training proposals: none. Unresolved decoder validity, censored/unknown
mass and the failed general pair law are the concrete blockers.
