Preserved preparation source revisions, before GPU or competition data execution:

- d09dde17ccf349699f46adf21cd6c643c6a8e49f52d1ec707100b0d5ce6b0d76: initial implementation, cpu-check-attempt01.json.
- 5cb4b2e0b3a400a5c6f7f2b535afeb5a6a26c492624eb50d8aba406a13895dd8: added immutable scalar buffers loading guard; cpu-check-attempt02.json.
- 707f134fb29db3ee9464af6f9eedae3b4fd8397898802aa27ac8f7672d6b6f86: require exact integer buffer dtype, finite float32 parameters, reject invalid scalar load before mutating scalar buffers, require external frozen fit-array hashes in load_scalar_model; cpu-check-attempt03.json and critic independent current-source checks.

Interpolation, custom table backward, forward AMP residual dtype and initialization are unchanged across these revisions. Reconstructed historical sources were byte-hash verified. No GPU or empirical read authorized before this record.
