# H232 V7B short-slug terminal delivery

- Kernel: `vladkochetov07/h232-gru384-b8-amp4096-v7b-shortslug`
- Provider status: **KernelWorkerStatus.ERROR**; worker phase `failed`.
- Fit: all four requested epochs completed; fixed epoch 3 was selected. The trainer recorded 59,680 update attempts, 59,654 completed updates and 26 AMP skips. No score or prediction was produced.
- Failure: post-fit guard raised `ProfileAbort: frozen fit membership overlaps the fold-0 evaluation mask`. The frozen fold check is source-qualified and clean: train fit 10,535 / valid fit 1,405 / valid evaluation 468; exact manifest matches; zero valid-fit/evaluation rowgroup or component overlap. The worker's post-fit condition compares bare integer IDs across source namespaces, yielding 463 train-versus-valid aliases.
- Checkpoints: all five checkpoint files downloaded and their hashes match trainer provenance; selected `model.pt` SHA-256 `1e26ab6853b5147def2da958f5b63cb1da49e7b3dee5b4b43022615bb3164686`; epoch-3 snapshot SHA-256 `aaa731da1692ee5083c80296fd665b0a60815c9dc18430457f702cf779c2b829`.
- Collection: `kaggle kernels output` and `kaggle kernels logs` both returned exit 0; 146 downloaded files (51943065 bytes) are listed and hashed in the JSON receipt. No prediction output file was present.
- Next action: root review/admission for a delivery-only export successor using the selected checkpoint; no retraining is required by this membership audit. Root retains push and grading ownership.

The JSON receipt contains the full downloaded-file SHA-256 inventory and the 463 source-qualified alias pairs.
