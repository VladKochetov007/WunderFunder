# R10 startup preparation

This directory contains a local-only, one-off R10 CPU launch body and import-smoke bootstrap. The remote body admits only an empty campaign cgroup tree, checks the existing parent limits and memory headroom, verifies persistent scratch and selected-GPU ownership, validates the fixed R3 metadata identities, stages only sealed R10 paths, then binds one systemd launch to the returned InvocationID. It does not inspect NPY payloads or import the canonical WNN package.

Before root assembly, substitute the frozen preparation-manifest SHA-256, canonical data.py SHA-256 string literal, runtime module-map Python literal, and sealed CFG/inline-source representations in the matching placeholders. CFG paths must equal the manifest noncanonical paths plus PREPARATION_MANIFEST.json, IMPLEMENTATION_REVIEW.json, TEST_RECEIPT.json, and README.md, all below idea_screens_r10/. The R10 CLI is invoked with the accepted sample SHA and preparation-manifest SHA; membership is fixed internally by R10.

No remote action was run while preparing these files. Root owns independent review, assembly, admission, launch, capture, and acceptance.

## Actual completion

The one reviewed launch ran successfully at03:50:55–03:51:26UTC, invocation8aa5dbfc71e946188eaac5611a4a8e74, then was captured and stopped. [Capture V2 review](CAPTURE_STOP_REVIEW_V2.json) preserves the single `systemctl show --all` correction for an omitted empty property; V1 source and failed raw transport remain in the terminal records. No compute retry. [Captured records](../attempt_v1/README.md) bind the resulting receipts and exact stop.

The assembler is the original preparation-time script, not a relaunch instruction. For offline byte reconstruction after navigation updates, the staged root README bytes are PREPARED_README.md; TRANSPORT_BINDINGS.json gives their original hash. Never rerun the launch or replace the completed create-only paths.
