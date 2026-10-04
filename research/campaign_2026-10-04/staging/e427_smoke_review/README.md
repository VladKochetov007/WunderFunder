# E427 full-shape production smoke drafts

This namespace contains three matched, one-epoch-initialized smoke commands
for the fixed E427 aligned, stale, and no-op arms. Each uses the original
batch size 32, sequence length 20,000, chunk size 1,000, exact fixed E424
parent, fit-only context statistics, teachers, propensity cache, objective,
optimizer, EMA, and AMP settings. The trainer exits after two real loader
batches, before any checkpoint or score is written. The wrapper records an
audit JSON only; it does not promote the result to a fit.

The smoke adds two CUDA checks before the first optimizer attempt: it compares
the E427 model with its zero-extension parent on the first full `[32,20000,211]`
batch, and it changes rows 512 onward in a real input prefix to test that the
earlier context rows remain exactly unchanged. After two batches it reports
attempted and completed optimizer steps, AMP skips, EMA steps, student and EMA
constraint-audit counts, final gate gradient/weight signal and finiteness,
peak host/CUDA memory, parent/statistics hashes, and the inherited
normalization identity. Aligned and stale must receive finite real update
signal; no-op must keep the added pathway inactive. All arms must stay under
24 GiB host RSS, 4.2 GB CUDA allocated, and 4.8 GB CUDA reserved.

`--no-eval-during-fit` prevents the trainer from constructing or running an
evaluation loader. The inherited `--eval-source holdout` is retained solely
because E427's matched-recipe validator fixes that argument; the smoke never
loads holdout sequences or computes a holdout metric. The full-fit recipe has
prior teacher-lineage exposure and is not an untouched-generalization claim.

These three specs are drafts only. They require root-side review and admission
after queued local GPU work is complete. No smoke, full fit, prediction export,
or score has been launched from this packet.
