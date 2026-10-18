# Exact deployed R01 v4 source for successor review

Root collected this snapshot through the existing SSH operator console on 18 October 2026. `deployed-review.tgz` is 102,725 bytes, SHA256 `be625fedec2771c128ab8db323824c0a3610070fcf0ba477cfb3596574261713`. Transport chunks are retained in `transport_chunks.json`; decoding reproduced the server archive byte count and SHA256.

`source/EXECUTION_MANIFEST.json` is the actual frozen v4 manifest. All included `code/`, `control/`, executable, panel, policy, contract and recipe bytes match its pins. The four large canonical fold payloads were deliberately excluded from this review archive; their original hashes remain in the manifest and the unchanged server files remain authoritative. This is a source review snapshot, not a standalone runnable data package.

The snapshot supports the [fourth-process measurement and queue successor review](../density4_measurement_r1/README.md). No active controller, source file, original fit or recipe was changed while collecting it.
