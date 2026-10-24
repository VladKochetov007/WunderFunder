# GPU profile packet hash audit

**Scope:** metadata and source bytes only. No imports, tests, cache reads,
profile, fit, Kaggle API call, or upload were performed. The GPU candidate
remains unadmitted.

## Verified against actual local bytes

- `source_overlay_cuda_r1.wnn` is 389,120 bytes with SHA-256
  `8d5c2567589876855256e7009a5be23a9f4b56ff366956da072daf21a49b2716`.
  Its 43 tar members consist of the 42 files declared by
  `SOURCE_OVERLAY_MANIFEST_GPU_R1.json` plus `overlay_manifest.json`.
- All 42 declared member hashes match the bytes read directly from the tar
  payload (zero mismatches). The embedded `overlay_manifest.json` hashes to
  `c977ca865e2e40da4fe7f38d7ab53c533edd8f85dfd4df6266ea2c1a581e3fef`,
  matching the actual local manifest file.
- The overlaid `SOURCE_SHA256_GPU_R1.json` payload hashes to
  `bf034ffc8ac0212f5a52d7d5b9c57a11bb54750275147aa70d601c0a7b9795fc`,
  matching the hash pinned in `SOURCE_CLOSURE_GPU_R1.json`. Its 66 module
  pins were compared to actual current checkout files: 64 match directly;
  the two expected CUDA-profile replacements differ in the base checkout,
  and their actual archived payloads match the declared profile hashes:
  `age_readout_cli.py` `4962b60511b1f79911f435dcd5190dc3f3a0fc3f58218eb8232ec59bd2d8e4d3`
  and `age_readout_runner.py` `77e3904c1d289af6e09b49d8062bb38495a308b82d8067869e354b7bef5a5c74`.
  No source path was missing from the current checkout.
- `kernel_driver_template.py` is 47,220 bytes and hashes to
  `8e653edc7370e5be046bae8068cf53dea854e6fc8f19921adba6c5c1ce73473b`,
  matching the closure receipt. The 66-module profile source manifest pins
  the same machine recipe SHA as the CLI.
- The actual local machine recipe,
  `research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json`,
  hashes to `0a67fe12c520e56bcf14a3a1efecb5030f3aa7ce932101814c9dd1c675c930fd`.
  Its `folds` and `cells` entries both specify F0/seed1001 and F1/seed1002.
  The first GPU packet's F0/seed1201 label was wrong. The corrected
  [KERNEL_RECIPE_R2.json](KERNEL_RECIPE_R2.json) uses seed1001; R1 is retained
  unchanged as the original erratum (R1 SHA-256
  `2745facbf52eadc85cae6695932b08fb98f8c47b820641f9be29fe023191fdca`;
  corrected R2 SHA-256
  `ad424a874d326336442ffd1a8417dfb7f70a1e9e4aacc0242c38280dd41e1878`).
- The admission template's transport and export-receipt hashes match actual
  local sidecar files: `SUFFIX_TRANSPORT_R1.json` hashes to
  `1ff63aaf02cf94176677e19b1bfebdce03404fe9ce8c800669953451dfcb8c6a`, and
  `SUFFIX_EXPORT_RECEIPT_R1.json` hashes to
  `c838527f734d7a2b791a704c1fa3ac900457b0d4b0876bf0d0a0d35b281fc40c`.
  The referenced 60-entry bundle manifest itself hashes to
  `ea241506eadd0a34f437c699bab3b7784a69b818b710e3d016c9b3599774f99e`,
  matching the admission template. The parent R3 source manifest file hashes
  to `6ce40d6e4980caf5d7b41e47f104e4a845a30f557c151d60c5ca3852024a1e5a`,
  matching the recorded parent closure.

## Limits and status

The existing private base bundle is pinned in the packet as 1,143,552,000
bytes, SHA-256
`971871550f50be91c5afe1134e1012a3ba3b9c94e2157b4f8b6ed17a62442ef8`. The
manifest file's own SHA was verified above, but that large external archive
was not downloaded or independently rehashed here. Its archive size and SHA
remain prior artifact pins, not results of this audit.

The separate R3 CPU profile most recently failed before extraction, so it
provides no cache reconstruction timing. GPU admission, upload, and execution
remain pending root review. The corrected seed does not change the machine
recipe, source overlay, numerical implementation, or profile scope.
