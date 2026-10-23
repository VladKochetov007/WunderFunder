# Reader V6 review snapshot preservation

The announced reader review snapshot has been preserved separately at `INDEPENDENT_SOURCE_REVIEW_V6_ANNOUNCED_SNAPSHOT.json` (9,673 bytes, SHA-256 `5e067c1d9972be3f00f462a075084b5ea315e3708bdb18669a0cf52be1a2d8ed`). It was reconstructed by reversing the exact two reviewed-file row replacements recorded in the local Codex session change log and verified against the announced byte count and digest. The original review bytes are not claimed to be restored at the canonical path.

The current canonical review is `INDEPENDENT_SOURCE_REVIEW_V6.json` (9,525 bytes, SHA-256 `844fb16d236a6778179f55edfe37a553809097691b19060869309c0c07bbeeec`), which binds the V14 producer and corrected V14 binding. `INDEPENDENT_SOURCE_REVIEW_V6_V2.json` is an identical preservation copy of that current review. Runtime consumers must use only the review path and digest explicitly pinned by their own versioned source.
