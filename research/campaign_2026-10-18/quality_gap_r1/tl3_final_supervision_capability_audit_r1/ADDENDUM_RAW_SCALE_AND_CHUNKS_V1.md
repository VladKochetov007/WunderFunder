# Addendum: tiedlead raw scale and chunk boundaries

This addendum supplements the static capability audit without changing its R1 README, source bindings, or manifest. It records source semantics only; the archived source snapshot does not prove the runtime behavior or exact software environment of any historical fit.

The pinned archived `wnn_nn.py` (`a6e13c2aceedfd648a6cf3dcee0621f22bd0530af42b350a1d66a48267124bc5`, lines 82–92) begins the tied encoder path with `raw = xn * self.std + self.mean`. Thus the shared instrument/lead branches consume reconstructed pre-normalization feature values, while the ordinary input projection consumes normalized `xn`. A canonical adapter must preserve that scale split; applying the shared tied encoder directly to normalized `xn` would change its inputs and would not reproduce the recovered architecture.

The same source constructs `d` with a zero row followed by differences only within the current `encode()` argument (line 86). Consequently, the first row of every `encode()` call has zero delta. Under chunked forward execution, the first row of each chunk resets this delta even though the recurrent hidden state is carried between chunks. By contrast, the archived `lead_features` block is assembled in `train_rnn_rs.py` before chunked forward calls (SHA-256 `29c385f665d7fc8a2e25454fa33ad58d081a7db4bd2b0269d4f8ad153c29a79a`, lines 511–517 and 575–581), so its causal event history is sequence-wide and does not reset at chunk boundaries.

Before fitting, the adapter review should include tiny deterministic tests that:

1. Set nontrivial `mean/std` and verify each tied branch receives `xn * std + mean` values, while the main projection receives normalized inputs.
2. Verify `d[0] == 0` and `d[t] == raw[t] - raw[t-1]` within one encode call, then split the same sequence at a known boundary and verify the next chunk's first `d` is zero despite a supplied carried recurrent state.
3. Verify the separately appended lead feature remains based on sequence-wide past events across that same split, with no future-row dependence.

The chunk reset is a recovered behavioral detail, not an assertion that arbitrary chunk sizes are equivalent. Any choice to carry deltas across chunks instead would be a deliberate compatibility change and must be recorded and tested separately. No adapter code, tests, data, or model execution were performed for this addendum.
