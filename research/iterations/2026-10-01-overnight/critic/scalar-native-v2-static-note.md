# Independent SCALAR native derivative static review

This review inspects source and actual compiled instructions only, on CPU37. No model/module import, generated inference, empirical data, timing, GPU, or CPU18 work was performed. The running baseline3 full-valid audit is unaffected.

The first prospective derivative replaces two volatile interpolation temporaries by ordinary floats inside GCC `optimize("fp-contract=off")`. At the exact modules below, the aggregate remains a distinct out-of-line function called from both the Python scalar-feature binding and `Predictor::step`. It contains no fused multiply-add instruction; its double fraction has two scalar-double subtractions and one division followed by one double-to-float conversion. The build uses `-O3`, architecture flags, `-mtune=generic`, `-fno-trapping-math`, `-fno-math-errno`, with no fast-math, associative-math, or LTO flag. This bounds the inspection to the actual binaries, rather than assuming an attribute is effective.

| ISA | Exact module SHA256 | Interpolation/sum instructions per feature | Final scale |
| --- | --- | --- | --- |
| v2 | `03d7a8e32aa159c82b1af40a6376f553055729e32ed9e478c559c9d1e4310d88` | 16 scalar subtract,16 scalar multiply,32 scalar add | Four packed SSE multiplies |
| avx2 | `b0fe37fff1d3599e521d765ee798a1e104181bf33769aca2744497360b8f5d95` | 16 scalar subtract,16 scalar multiply,32 scalar add | Two packed AVX multiplies |
| v3 | `d1492ca1f5a0537e00548ec6e3b81b7ef407ae988cbe001b0c354efa4ebf9a51` | 16 scalar subtract,16 scalar multiply,32 scalar add | Two packed AVX multiplies |

Thus the separate-rounding boundary is visible in machine code, but **the interpolation and ordered sum remain scalar in all three builds**. There is no packed subtract or packed add in aggregate. Removing volatile alone does not demonstrate core SIMD or a speed gain. A possible output/table alias is a plausible compiler obstacle, not something established by these instruction counts. The JSON receipt contains exact function disassembly hashes, all arithmetic instructions, opcode counts and callers; SHA256 `b2fa3fe5dd5671b5cae3326383e718f477f7c3707bd125f2ebbdb87cc1e6e455`.

## Output-only restrict proof, before the next source edit

Root authorized a single prospective output-only alias correction within the same un-timed derivative if callers establish disjointness. Static inspection of the complete `native_v2/include` and `native_v2/src` tree finds exactly two calls to `aggregate`:

1. `ScalarEmbedding::add` declares `alignas(64) float value[16]` in its own automatic storage and passes that array as output. The row comes from a checked Python input array; knots/table are owned member vectors. Neither can occupy this newly allocated automatic array. `Predictor::step` is the sole `add` callsite; it passes the raw row and a private encoded-state buffer.
2. The `scalar_features` Python binding first copies weights into owned native vectors through `to_weights`/`ScalarEmbedding::from`, then creates a fresh `py::array_t<float>` for output. Each row passes a separate span of that new output allocation. The API has no caller-supplied output-buffer argument, and no view into member-vector storage is exported.

The public Predictor/Ensemble `step` and `run` bindings also allocate their outputs internally. The native aggregate method is not separately exposed to Python. Consequently an **output-only** `float* __restrict__ out` declaration is compatible with the current supported call paths. No extra runtime rejection is needed for an alias the exposed API cannot construct through valid allocations. An external C++ consumer directly calling this header method would inherit the new output-nonalias precondition; that is outside this fixed extension interface.

This proof does **not** justify disjointness between the two read-only interpolation endpoints. They intentionally coincide for `count=1` and right-endpoint cases. It also does not prove that GCC will choose SIMD after the correction, that all floating environments match, or that runtime will improve. The actor must preserve this intermediate build, rebuild once, rerun the previously fixed literal/full-model tests and obtain final assembly inspection. No empirical timing has been allocated by this review.
