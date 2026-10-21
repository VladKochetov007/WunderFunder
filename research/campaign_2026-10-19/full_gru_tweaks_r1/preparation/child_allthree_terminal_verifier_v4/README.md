# AllThree terminal verifier wrapper V4

This packet prepares a bounded local wrapper for checker V2 after the first
actual support attempt failed its strict NLL-stop schema gate. The V1 support
invocation remains preserved as failed evidence and is excluded from all V4
actions. The new support identity is
`computer-vision-homework-fg19-child-allthree-verify-v2.service`, with private
output directory `fg19_child_allthree_verify_v2`.

The template is based on the frozen V3 wrapper (707,678 bytes,
`12f357f2127abded88b0e89d5ba08323310bcc5337a980f89a2ce90b4ec57151`). It
changes the support unit/output directory, private checker filename and pin,
and root-binding schema. It retains the seven-file closure, exact read-only
receipt map, source/split/cache/profile/parent identities, path guard, resource
caps, and restricted decoder. After writing durable admission and launch
receipts and verifying a valid support invocation, the outer process emits a
single JSON line with exactly `unit`, `outcome`, `admission_base64`, and
`launch_base64`. It reads those two receipts with the existing stable no-follow
reader, caps each at 1 MB and their combined size at 4 MB, and makes no
self-read or body-hash claim.

The create-only renderer required the checker V2 source review and root
preparation, the seven-code private-import proof, and the strict actual native
terminal/raw pair. It created `REMOTE_SUBMISSION_V4.py` at 710,826 bytes / SHA-256
`c186e7b4473bc11681f61c55a741f17b2fdab828c7906090ef10574a8aa2df90`; the exact
binding and gate results are in `V4_RENDER_BINDING.json`. That native pair is pinned at 717 bytes / SHA-256
`da91b782652e98843fdadeadcb41ab2c3f67492ead0e95c108ef18c85916f27a` and
82,591 bytes / SHA-256
`0d9ba24a1e241ee22090247711f5e875def72faf5b052cf149d227cdca2f262a`. The
renderer does not query B, stage or submit the wrapper, read fit model/state or
data/cache payloads, or start a fit.

Files in this packet:

- `REMOTE_SUBMISSION_V4_TEMPLATE.py` is the V3-derived create-time template
  with empty payload, fixed-pin, and final-binding slots.
- `render_child_allthree_terminal_verifier_v4.py` verifies the exact local
  source and evidence gates, then creates the body and render-binding receipt
  exclusively; it never overwrites either output.
- `test_child_allthree_terminal_verifier_v4_contract.py` contains focused
  static AST fixtures for the adaptation, receipt-map preservation, and
  bounded stdout reply ordering. It does not import or execute the wrapper.
- `LOCAL_CHECKS.md` records the narrow local checks and their limits.

The rendered body has not been staged or executed. It requires independent
exact-body review and a separately reviewed V2 result-capture consumer before
any future support action.
