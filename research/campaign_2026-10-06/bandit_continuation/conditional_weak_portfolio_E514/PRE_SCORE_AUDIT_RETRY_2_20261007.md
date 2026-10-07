# E-514 frozen-partition replay guard repair

Recorded 2026-10-07T03:03:49.189313Z before any comparison score. After the notebook graph repair, the source audit reached the frozen partition write-once guard and stopped because direct dictionary equality compared integer in-memory chain IDs with string JSON object keys. The stored assignment was not rewritten, and no comparison metric ran. See `PRE_SCORE_PARTITION_REPLAY_FAILURE_20261007.json`.

Canonicalize the reconstructed document through JSON encode/decode before comparing it with the existing partition. The retry still uses the original sensors-derived seed and chain assignment; the canonical equality check must pass against the frozen `E514_CHAIN_PARTITION.json`. No candidate, lambda, endpoint, data split or score rule changes. The eventual v2 source manifest will bind both this repair note and the partition hash.
