# E488 actual verify-only v1 path failure

The frozen independently reviewed v1 verifier ran once against the actual fit receipts under scripts/limited (one CPU,6GiB). It failed in check_scientific_sources before reading any test-role group or loading checkpoint predictions: source path resolved to `research/research/campaign_2026-10-05/activation_context_mixture/fit_context_mixture_v2.py`.

The code conflated file and directory parent indices. `Path(__file__).resolve().parents[2]` points to research; the repository is the file's parents[3], equivalently HERE.parents[2]. The original CLI help and synthetic fixtures did not exercise actual source paths. The v2 source fixes that one line, adds an actual-root/file-existence fixture and clean/tmp CLI regression, and preserves original v1 source/review. Full fits, sealed science contracts and numerical gates are unchanged. A fresh independently reviewed actual verify-only attempt is required before scoring.
