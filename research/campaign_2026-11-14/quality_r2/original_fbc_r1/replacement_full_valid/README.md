# Exact replacement ZIP full-valid receiver

Output of the [fixed three-member candidate check](../run_replacement_cpu.sh), bound by [replacement_receipt.json](../replacement_receipt.json). The native receiver writes `predictions.npy` incrementally; it is not a completed endpoint until `report.json` records successful validation and a matching prediction hash. The final report includes official full-valid score, A-B-A/warmup checks, peak RSS and measured one-CPU timing. All valid sequences are exposed development data; no portal or public-transfer claim follows from this report.
