# E465 AdamW FP32 CPU parity transport v2 draft

This private, CPU-only packet is the isolated retry for the v1 aggregation `IndexError`, which happened after all 7,460 JAX and Torch updates had run. The revised checker differs from the frozen v1 checker by one line only: NumPy advanced-index selection now receives `list(indices)` instead of a tuple, so selecting grouped moment errors indexes one-dimensional `error_by_value` correctly. The numerical calculations, schedule, tolerances, optimizer, and model/data-free scope are unchanged.

The four-file source archive contains the worker, revised checker, and byte-identical E465 `rnn.py` and `onecycle.py`. Its builder pins the checker SHA `fbdddd791856250d965159dcce20a95c557aa023602aa10594a232a7b5d9a9af`, rnn SHA `294eff11d906fe434dba2c4d0ed4741ad2191f756c4a35204474cb229ff939dc`, and OneCycle SHA `f857fe857a361249c2742e3486cd32b846e0f47413a5c4db2ae1fb6c65509ded`. No model checkpoint or raw data is attached.

Runtime requirements and limits match the reviewed v1 packet: Python 3.13.15, JAX/JAXlib 0.11.1, NumPy 2.1.3, Torch 2.11.0+cpu; one worker CPU and 2 GiB runner memory inside an external 4 GiB RSS limit; 300-second worker and 600-second whole-job wall deadline; GPU, TPU, and Internet disabled. The runtime pin provenance is recorded in the source manifest. The worker requires the checker to report exit 0, passed status, 7,460 updates, the exact rnn/OneCycle hashes, and no loaded model or data.

Two source-only tests are included outside the uploaded archive. One extracts the actual `maxima` expression from the v2 checker AST and evaluates it with a tiny NumPy vector to verify the three grouped maxima; another asserts the complete v1-to-v2 source diff is exactly the indexing change. The copied worker contract tests check pass, fail, missing result, and mutated source cases. They are drafted for the root-bounded check; no JAX/Torch/model/data computation has been run locally.

The packet is not uploaded or launched. Root should verify the new closure and runtime inventory before remote execution. The earlier v1 packet, failed attempt, and results remain preserved unchanged.
