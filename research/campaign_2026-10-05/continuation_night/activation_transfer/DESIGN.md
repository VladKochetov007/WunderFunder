# Fixed readout transfer: registered before implementation

Status: registered, no scientific fits admitted. Parent contract: ../CONTRACT.md.
Hypothesis H-121 / experiment E-493 (proposal CN-ACT-01): the E488 fixed readout
mixture improves the complete strong predictor, rather than only the H64
prototype. Source: reflect_05_day/RESEARCH_DIRECTION_AUDIT_2026-10-05.md:82
and campaign_2026-10-05/context_mixture_duration/E488_DESIGN.md.

Treatment replaces only head[1] with 0.5*GELU(x)+0.5*LeakyReLU(x,0.01).
Embedding and native GRU remain unchanged. No parameters or random draws are
added; same seed yields identical initial parameter bytes and state-dict keys.
Default GELU stays the historical nn.GELU module. Checkpoint args must retain
readout_activation and head-only fitting must inherit it, rejecting conflicts.

Two fresh arms: gelu, fixed_mix. No dynamic gate or combined reanchor arm.
The fresh unchanged control is shared with E-494 at identical seed/fold;
its complete recipe and immutable predictions must match exactly.
Seed3/fold0 is the screen; seed4/fold0 and seed3/fold1 are reserved confirmations.
Each parent uses full fold-excluded fitting data, H256, two GRU layers, B32,
T1000, four epochs/fixed ep3, Muon RNN scope, cached fold-specific since
teachers alpha .5/bootstrap8, EMA1/.999/eval-EMA, structural/next auxiliaries
.3/.3, propensity power2/floor.03, since32/floor.25. Both arms use seeded
batch ordering and identical fitting-only normalization membership. Parent seed
varies; final-head seed stays3. Head: frozen inp,gru; valid fitting pool; B16,
T2000, one AdamW/global-correlation epoch, LR.0005, fixed ep0, mask weights.
Complete endpoint is the fixed equal parent/head ensemble; also report parent.
Recipe sources are the exact seed4 public parent/head provenance files and
scripts/continuation_recipes.py; fold lineages replace final-data ancestors.

Primary metric is the frozen global_wp_clip2_v1 on common excluded components.
Report both targets, class/event-age slices, 2,000 paired component bootstrap
draws, seeds separately, and runtime. A null/reversal limits this treatment;
it does not reject every activation mechanism. Public .7000 is not forecast.

Before fit: independent source review, exact default output/gradient parity,
state-key/RNG parity, checkpoint roundtrip and head inheritance, full-shape
resource probe and current quota/admission. No production-data allocation is
authorized by this card alone. Prefer Kaggle given observed local headroom;
one worker, measured RSS/VRAM and raw samples before launch. Probe unchanged
and candidate at B32/T1000 with production-width inputs, objective, EMA and
optimizer; measure loader allocations separately. Preserve >=8GB host
MemAvailable and all parent contract limits. Stop on breach and preserve it.

Native/ONNX export currently assumes GELU in some paths. The fixed_mix model
is research-only until exporter/native support, Python3.11 parity and <=52min
one-CPU timing pass. Parameter-key compatibility does not establish functional
native compatibility. Do not send this checkpoint through an unchanged exporter.

Plots/tables belong in eda/; independent scorer owns metrics. Root imports
ledger_packet.json after checking IDs and source hashes. No root ledger writes
or scientific training are delegated to this actor.
