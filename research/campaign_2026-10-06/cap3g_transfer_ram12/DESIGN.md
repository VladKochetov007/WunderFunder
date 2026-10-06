# Cap-3G full-recipe transfer capsule

Status: 12 GB host-cap admission variant of the B16 capsule; source-hash closeout
and root admission remain required. It preserves the 18 GB v2 capsule unchanged.
No resource profile, fit, prediction export or score has run from this capsule.

This capsule carries the registered fixed GELU/LeakyReLU readout transfer
(E-493/H-121) and stopped-bootstrap original-label reanchor (E-494/H-124) into
the full GRU256 parent/head recipe. It preserves the existing treatment
definitions and uses one freshly fitted common control for both comparisons.

## Shared recipe and changed baseline

The previous B32/T1000/H256 control profile observed 3,611,295,744 device bytes,
which exceeds the current 3,000,000,000-byte campaign ceiling. This capsule
therefore uses **batch 16 for every parent arm**. The shared control is named
`common_b16`; it is a newly registered baseline and must not be described as an
unchanged B32 control. The smaller batch changes the number of optimizer updates
per epoch and the gradient-noise scale. Those effects are common to all arms;
any result supports only the treatment contrast under this B16 recipe.

Parent arms use the same fold-specific teacher/cache lineage, seeded batch
ordering, normalization membership, H256, two GRU layers, T1000, four epochs,
fixed epoch 3, Muon RNN scope, EMA teacher 1.0/decay .999, structural and next
auxiliaries .3/.3, propensity power 2/epsilon .03, true-mask .03, and
since-weight 32/floor .25. Distillation uses the cached seed-3/seed-4 teachers,
constant alpha .5, and bootstrap horizon 8 except in the separately registered
bootstrap-off control. The final-head seed remains 3; each head freezes input
and GRU, uses batch 16/T2000, one fixed epoch, AdamW at .0005, and inherits its
parent's readout and supervision settings.

The four parent arms are:

- `control`: GELU readout, reanchor 0, bootstrap horizon 8.
- `fixed_mix`: fixed 50/50 GELU and LeakyReLU(.01) readout, reanchor 0,
  bootstrap horizon 8.
- `reanchor010`: GELU readout and the registered .1 original-label contribution
  on surviving stopped-bootstrap rows, horizon 8.
- `bootstrap_off`: GELU readout, reanchor 0, horizon 0.

No treatment is combined with another. `control` supplies both experiments at
each seed/fold. The fixed complete endpoint is the equal prediction mixture of
parent epoch 3 and head epoch 0; parent-only endpoints are retained. No weights
or checkpoints are selected using evaluation outcomes.

The screen is seed 3/fold 0. Reserved confirmation blocks are seed 4/fold 0
and seed 3/fold 1. Each block has explicit parent and complete screen files,
2,000 paired component-bootstrap draws and the prior PUB143 class mixture as a
diagnostic only. Report the frozen global metric, the second target, class and
event-age diagnostics, each block separately, and runtime after actual outputs
exist.

## Resource profiles and gates

The four seed-3/fold-0 resource-profile drafts use the actual training loader,
teacher caches, objective, EMA, optimizer and model width. They warm up once and
measure two steps. Each uses at most 2 CPUs, 12,000,000,000 host bytes,
2,500,000,000 allocator bytes, 2,800,000,000 observed-device stop bytes, and
600 seconds. These are proposed remote resource requests, not measurements or
admission. A profile above either device threshold fails the affected shape and
must be preserved as a censored resource result.

Parent, head and prediction fit drafts use the same 2.5 GB allocator and 2.8 GB
observed-device ceilings; they remain unadmitted. Root owns quota, host/device
headroom checks, sealing, reservation and launch. Before sealing, refresh every
listed input identity at the provider, check current source hashes against
`source_manifest.json`, and confirm no local start is at or after the contract
cutoff. Profile both changed common control and each treatment before fitting.

The dependency inventory distinguishes inputs present in this checkout,
intentionally missing downstream model/prediction artifacts and external
inputs. Only root/provider transport may fill those dependencies. Any source
change after closeout invalidates this capsule's hashes and requires a new
reviewed identity. Existing October 5 capsules remain untouched.
