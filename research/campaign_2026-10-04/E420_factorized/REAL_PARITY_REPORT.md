# E420 actual W384/r128 native parity

The zero-update SVD export and real native parity completed. Three E409 streams of20,000rows each; all output rows checked. Saved FP32 factors reproduce stored effective dense products exactly. Native first-two-output max absolute difference 2.38418579e-06, absolute global-WP difference 7.71307596e-09; unchanged3e-4/5e-6 gates pass. Native factor bytes round-trip exactly; portable/SIMD and v2/v3 outputs are bitwise equal; reset/replay passes.

Actual runtime storage: 4,706,432bytes total streamed weights, INCLUDING 3,145,728factor bytes. Scratch 16,716, recurrent state 3,072; freed import-only dense recurrent matrices 7,077,888. Do not add the factor subset twice or compare only factors with an entire quantized dense predictor. Parity model retains23 training outputs; deployment timing will retain two scored outputs.

Original W384 WP0.8290104313 versus zero-update SVD0.8276324925 on these three arbitrary runtime streams (SVD delta−0.0013779387). Raw difference0.26066357 reflects a changed approximate model; it is not the implementation parity reference. This is approximation context, not held-out model confirmation or a width-gain estimate. No fitting or checkpoint selection occurred. Native timing remains pending a quiet CPU window.

Export peaked449MB under2GiB. Parity finished 190.26seconds, peak 752,418,816bytes, noOOM. The first parity attempt stopped pre-inference because a relative argv path differed from the bound absolute E409 path. It is preserved; the corrected path identifies the same hashed data.

Full source/runtime hashes: `e420_full20k_native_parity_v3.sealed.json`; numerical report: `full20k_native_parity.json`.
