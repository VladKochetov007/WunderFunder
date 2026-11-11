# Cards04–06: B300 implementation amendment

The 11 November request adds cards04–06 to this same draft PR. It supersedes
these cards' earlier separate-implementation-PR notes. Twelve additional fresh
fits are prepared, zero executed. Together with cards01–03 the packet contains
24 configured fits; cards07–10 remain proposals. Each card has two controls and
two treatments at seeds204101/204102, no reused control and no teacher fit.

The native recipe stays H768/L24/12 heads, context1024, gradient horizon256,
batch128, accumulation4, four full-data epochs, fixed finalE4, two B300 ranks,
Muon/AdamW, LR3e-4, warmup5%, cosine floor0.1, BF16/BF16x9. All optimizer,
precision and compile settings are inherited unchanged. These are mechanism
experiments, not speed optimizations or claimed quality gains.

## Implemented mechanisms

| Card | Control | Treatment |
|---|---|---|
|04 | Final-head WMSE | Heads after blocks6/12/18; primary +0.1 times mean auxiliary WMSE |
|05 | 24 independent blocks; initialize13–24 by value-copying1–12 | 12 unique blocks applied twice, with24 independent logical KV caches |
|06 | Primary WMSE | Primary +0.05 pairwise loss on within-chunk pairs(t,t−32) |

Card04 uses the exact primary mask, clipped targets and absolute-target weights.
Each head uses the same global primary denominator, summed across all ranks and
microchunks in the accumulation window. Auxiliary parameters use isolated
initialization. They are returned explicitly by the DDP forward during training;
evaluation executes only the final head. Export first strictly restores the full
training checkpoint, then removes the auxiliary heads from the inference object.

Card05's control is a distinct initialization, not an old plain-control score.
The treatment registers each of the12 unique blocks once. The unchanged Muon
builder groups these unique parameters; gradients from both logical applications
sum naturally and are not divided by two. Both arms still execute24 logical
layers and retain24 separate KV histories. This couples parameter count and
summed gradients; it is not an isolated proof of regularization or speed.

Card06 clips labels at±2, requires both rows under the primary training mask and
absolute label difference≥0.5, uses the fixed lag32 and temperature1, and weights
pairs by sqrt(abs(y_t)*abs(y_u)). It reduces separate target denominators across
the whole accumulation window and every rank before backward. The two normalized
target losses are averaged, including a zero term for any empty target. Pairs
never cross a chunk or sequence edge. Labels enter loss calculation only.

## Isolated integration

The canonical checkout remains pinned to PR75
`8626000a4b965763a70011a0291a891aa6535d7e`. No existing `src/`, `scripts/`, native
training config, optimizer implementation, validation or scorer is edited.
Cards01–03 keep their canonical epoch runner. Cards04–06 select an explicit
packet-local `TrainingEngine`; no monkeypatches, source rewriting or hidden
label-to-model channels are used.

- `auxiliary_loop.py` and `auxiliary_backward.py` provide the additional objective
  seam while calling canonical sequence sampling, batching, primary WMSE,
  precision contexts, state detachment, DDP synchronization and optimizer updates.
- `lifecycle.py` and `profile.py` accept that engine explicitly. They reuse
  canonical checkpoint, resume, RNG, monitor, epoch-index and final-output helpers.
- `depth_models.py` reuses factory-built temporal blocks. The new traversal exposes
  intermediate heads or maps24 logical positions to12 parameter sets.
- `auxiliary_reporting.py` records primary and combined losses separately, per-head
  WMSE, per-target pair loss/mass/counts and weighted pair concordance. Official WP
  still comes exclusively from the unchanged scorer.

The explicit adapters are needed because the pinned trainer has no auxiliary-loss
callback. They are new experiment code, not changes to an existing running job.
Numerical equivalence of the control path is a receiver obligation, not established
by source inspection. Optional scientific diagnostics such as per-depth gradient
norms and intermediate-head WP need separate receiver analysis; they are not
claimed as automatically emitted training metrics.

## Receiver and provenance gates

[Commands](B300_COMMANDS.md) use the existing single-stage, owned, timeout-bounded
launcher. All six cards require fresh source-bound checks/profiles at this source
revision. Earlier source receipts remain historical; never edit their hashes or
mix checkpoints from another revision into a pair. Runs already using the earlier
PR81 commit must keep their original checkout.

Receiver checks include explicit loss formulas, masked/empty cases, unequal-rank
and microchunk denominator consistency, card04 head gradients and head-removal
parity, card05 copied/tied initial function, unique optimizer ownership and24 cache
histories. A small synthetic one-update probe compares the new control engine
against the canonical loop: losses, counts, parameters, optimizer and scheduler
state. This is a single-device engineering check; the full-size two-rank profile
remains mandatory. These checks are prepared for B300, not executed locally. Each full-size
cell still needs its own50-update profile and reload/resource review before fit.
The early-warmup profile does not certify peak-LR stability or a complete runtime.

The same official split is repeatedly exposed development data. Report every
paired seed/target delta and incomplete/negative result; no new fold or statistical
significance is created by these12 additional fits.

MIDAS recorded a preparation-only route for cards04–06. Its original request
containing publication was routed to the native external workflow and retained
as evidence; Git publication follows the user's explicit instruction. The existing
MIDAS receipt exporter now recognizes all six cards, including `copied_depth`.
No MIDAS research-agent execution, GPU run, evidence import or new metric is claimed.
