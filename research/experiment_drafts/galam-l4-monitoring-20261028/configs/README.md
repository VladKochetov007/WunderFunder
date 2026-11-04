# Prepared matched recipes

Six complete JSON recipes materialize the [three-arm/two-seed plan](../PLAN.json):
full-lstm-s1001/s1002, full-gru-s1001/s1002 and local-lstm-s1001/s1002.
All use5epochs, FP32/AdamW lr1e-4 wd0.01, 5%warmup/cosine, hidden64,
attention64, chunk64, batch16 and all-epoch checkpoint retention. Batch16 is an
unprofiled starting candidate, not a measured optimum or resource admission.

Monitoring evaluates all official valid and a fixed source-qualified32-member
training diagnostic cohort; exact fitting members and cohort are recorded by
the actual receiver. Initial settings are assumptions, not Denis's lost recipe.
Every future batch/precision/source change needs its own hashed recipe/profile.

The recurrent/branch variants deliberately alter decoder channel count and total
parameters. `scripts/check_galam.py --config CONFIG` reports the effective graph,
active fusion order, inactive placeholders and parameter count without fitting.
Local-only inactive global/ConvMixer fields cannot be tuned silently. The source
template's graph claims do not turn these adaptations into original GALAM code.

No recipe has been trained, staged remotely or dispatched.

| Arm | Seed1001 | Seed1002 | Effective parameters |
|---|---|---|---:|
| Full LSTM | [recipe](full-lstm-s1001.json) | [recipe](full-lstm-s1002.json) | 376426 |
| Full GRU | [recipe](full-gru-s1001.json) | [recipe](full-gru-s1002.json) | 286698 |
| Local-only LSTM | [recipe](local-lstm-s1001.json) | [recipe](local-lstm-s1002.json) | 195398 |

[Inspection receipt](INSPECTION.json) records exact file hashes, loader success,
random-weight parameter counts and effective/inactive topology for all six
recipes. It executes no forward or optimizer step and establishes no quality.
