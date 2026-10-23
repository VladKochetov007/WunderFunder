# NLL verifier capture files

This directory contains the exact bounded response and extracted metadata receipts for the successful NLL terminal verifier capture. The raw response is [STDOUT.raw.json](STDOUT.raw.json), 54,641 bytes, SHA-256 `62688aa0f92ac8933f11745a32ecb8b2f461869c71b366a9ca7fa7def3819be7`. Its transport receipt is [TRANSPORT.json](TRANSPORT.json); the response was persisted before parsing and no retry was performed.

The three extracted files are [VERIFICATION.json](VERIFICATION.json), [VERIFIER_ADMISSION.json](VERIFIER_ADMISSION.json), and [VERIFIER_LAUNCH.json](VERIFIER_LAUNCH.json). The fourth embedded member, [unit.log](unit.log), is empty. Their source joins are preserved inside the raw response. [ROOT_CAPTURE_BINDING.json](ROOT_CAPTURE_BINDING.json) binds the capture reader and the exact admission/launch identities and hashes.

This capture is an integrity/provenance result, not a quality evaluation. For a concise local-derived summary and links to the acceptance, stop, and authority records, see the [NLL capture observation](../../capture/OBSERVATION.json) and its [README](../../capture/README.md). Those summary files do not replace the raw response or extracted receipts in this directory.
