# E508 footer range diagnosis

Root independently inspected producer `scripts/build_campaign_splits.py:31` and E508 consumer `raw_teacher_pair/packet_worker.py:321` on2026-10-07. Producer hashes Parquet footer metadata plus its8-byte trailer; consumer starts8bytes later. This is a validator bug, not evidence that the mounted data changed.

Bounded local byte-range checks returned:

| Input | Footer metadata bytes | Wrong consumer range SHA256 | Producer range SHA256 (matches frozen registry) |
|---|---:|---|---|
| train.parquet | 60,437,285 | cb7786010e10f13f989b83277a4c901d44621b5b7357224b67d9a456cb22e564 | 071a60cc9e72a95b013fe6784318b92a6fb82b7c95b6371d1f21a0e5d04e3bfb |
| valid.parquet | 10,686,926 | b2a6ccbca68438e160c0b2ed4b438c4c0e35ce5fcb11842af97f9f207997457e | 4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b |

Cloud operator independently reproduced the valid-file calculation. CorrectedV3 must verify actual trailer length and magic, hash the producer's exact range in chunks, and persist expected/observed fingerprints and discovered input paths before failure. Local parity does not yet prove Kaggle-mounted bytes match; the corrected provider run must establish that. PreserveV1/V2 terminal artifacts and original packets.
