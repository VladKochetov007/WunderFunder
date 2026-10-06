# W384 GPU v3r1 independent packet review

Review date: 2026-10-06. This verdict applies only to the immutable private
packet identity below. **GO for packet/source handoff and controller admission
review; no provider push or launch is authorized by this note.**

## Packet identity

| File | SHA-256 |
|---|---|
| `kernel.py` | `8fd9e46b794e79558930c7f829d26345970ec5ad1d4755d35b5ae8d532a72481` |
| `kernel-metadata.json` | `3bda12d0ee93c36e17fc0ff63a255f4b7b6facbf68a64a27134c05f4754de5b1` |
| `review_packet.json` | `1d56353cb93e812bbf0e91eb15b42cf7377f462ac0786b8ef3a2ebb7530007eb` |
| `SOURCE_RECEIPT_V3R1.json` | `8bac7da6a97be100c3615fb2c806c5f36f47acd89741b05a39a315df2937861c` |

The review packet's kernel, metadata, manifest, payload, and source identities
match current bytes. I decoded all five embedded source files from `kernel.py`;
their SHA-256 values match both the embedded hash table and packet metadata.
The W384 source manifest and original payload remain bound to
`4518e583e9eab4bcaa5c5eb499958e9329748191006e68cd89a101d12cee1958` and
`498c0a5d1105a9fbbe34d02fffb7d4a6c12cd5d2f6b10f8ef647b7bac03bb90a`.

## Review findings

The material blocker from v3 is fixed. The phase runner assigns fresh
`serialv3r1` probe and fit identities, refuses preexisting run directories
before probing, and derives the manifest from the exact fit directory. It
requires `model_ep3.pt`, `model.pt`, `provenance.json`, `history.json`, and
`training_state.pt`; it hashes every regular top-level file while checking
device/inode/size/mtime stability, then includes both validation prediction
exports. The tests exercise the fresh names and fail separately when each
required fit file is missing.

The width recovery remains loader-only: its embedded source follows runtime
`wnn.data.SEQUENCE_LENGTH` for the short fixture and binds the production
source to 20,000 rows. Row-group metadata, decoded offset and final length,
sequence IDs, and target/mask alignment remain checked. The derived trainer
hash is unchanged from v2; the packet reports no scientific argv or split
change. The private kernel metadata keeps internet and TPU disabled, has no
competition attachment, and contains no launch step.

I independently ran the current transport suite with one CPU, a 2-GiB memory
cap, CUDA hidden, and plugin autoload disabled: 17 passed in 21.97 seconds.
Ruff passed on `phase.py`, `test_transport.py`, and `build_packets.py`. The
packet also binds the legacy staged-preflight receipt (`legacy_cpu_tests.xml`,
36 passed, zero failures/errors); I verified its recorded SHA-256 and did not
rerun that unchanged suite.

The packet records `provider_recovery_attempt=2` of 2 and explicitly leaves
controller review and reservation required. No W384 v3r1 provider action,
preflight, scientific fit, prediction, probe, or score occurred during this
review. Root/controller must make any execution decision under the existing
reservation and budget gates.
