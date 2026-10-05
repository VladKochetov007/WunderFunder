# E463 existing TPU job drain

Observed 2026-10-05T00:30:02.688658+00:00: `vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1-runner` is **QUEUED**. Terminal outputs collected: **False**.

This collector polls and downloads only the already-submitted preflight. It starts no kernels, performs no scientific retry, and does not launch research work. Provider completion and raw self-reported checks require independent interpretation; neither alone establishes full-model TPU or public-score success.

Raw status, output paths and small result hashes are in `checks/e463_pause_drain_collection.json`.
