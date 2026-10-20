# Parent-fit R2 terminal verifier V2

This V2 packet preserves the V1 verifier and uses the actual, exact-stopped RNG diagnostic to check the saved CUDA generator state. It binds diagnostic report SHA `0bb2a87b9796872d5e26666ea7398bca69b7f0a0ce27bb0895714b5ed654598b`, capture acceptance SHA `655c5f12db5e166cfabf68c343aeb10d709f95db4abffeaf2925b413226837ec`, and exact-stop SHA `8e935356da08ac249e47389bb65bc5597134e6442ab9d914da1e7ec256b73c67`. The checker requires runtime PyTorch `2.14.0+cu130`, the diagnostic's pinned CPU RNG signature and single CUDA `uint8[16]` signature, and the same signatures in the bounded loaded training state. Source runtime/provenance checks remain in the full checker; the diagnostic is not parent acceptance.

The packet targets the already captured parent run and source `68ff682977b822f38351aac0f621146f5f13b440` (167 files). It stages only the pinned checker/helpers and small JSON evidence. Fresh support gates, one CPU-only verifier submit, and durable unknown-outcome recording are in the body. No model, optimizer, data, or prediction arrays are read by preparation. The existing profile ancestry is transitively bound by the source-qualified parent-fit admission; this packet does not repeat that profile.

The focused test file exercises the 16-byte uint8 CUDA signature, wrong length/dtype/list cardinality, digest mismatch, and runtime/report Torch-version mismatch using tiny synthetic tensors and the small diagnostic report. The local checker also runs the cold metadata-only payload-read sentinel. It does not run unchanged training tests or read model/data arrays locally.

The separate source-qualified runtime import proof is recorded by [`parent_fit_terminal_cold_import_proof_v3.py`](../parent_fit_terminal_cold_import_proof_v3.py) and [`PARENT_FIT_TERMINAL_COLD_IMPORT_V3.json`](../PARENT_FIT_TERMINAL_COLD_IMPORT_V3.json). The exact command was `.venv/bin/python research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_cold_import_proof_v3.py`; the receipt pins the checker, S5 helper, support helper, all 26 imported `wnn` modules, and the diagnostic report. Its read sentinels remained empty and CUDA was not initialized.

Run from the repository root:

```sh
.venv/bin/ruff format research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_check_v2.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v2
.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_check_v2.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v2
.venv/bin/python research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v2/prepare_parent_fit_terminal_verifier_v2.py
.venv/bin/python research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/parent_fit_terminal_verifier_v2/check_parent_fit_terminal_verifier_v2.py
```

The preparation is local only. Root owns independent review, fresh remote gates, one submission, verification acceptance, and exact stopping.
