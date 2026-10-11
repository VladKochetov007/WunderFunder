# H382 R3 Parquet footer correction

R2 is preserved exactly as pushed. Root's actual Kaggle startup review found that
its raw Parquet verifier sought back `footer_bytes` only, omitting the four-byte
little-endian footer-length field and trailing four-byte `PAR1` magic. The
project producer `scripts/build_campaign_splits.py::source_identity` records a
SHA-256 over `footer_bytes + 8` bytes. With the same mounted R2 raw files, the
short R2 range produced train digest `cb7786…` instead of the registry digest
`071a60cc9e72a95b013fe6784318b92a6fb82b7c95b6371d1f21a0e5d04e3bfb`, and
validation digest `b2a6…` instead of
`4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b`. The
producer's split and portable registry agree on both full identities.

R3 changes only the startup verifier: it checks the trailer length and `PAR1`
magic, hashes from `file_size - (footer_bytes + 8)` through EOF, records the
observed range and digest, then enforces the unchanged registry digest. It does
not alter source, fold membership, model inputs, recipe, optimizer, fit/export
limits, or scientific endpoint. The fixed fold-0 split/registry and current V6
prediction fixtures remain bound unchanged.

A local contract check extracts the exact `source_identity` function from the
producer and the exact `parquet_footer_identity` function from the R3 worker,
then compares their identities on a small real PyArrow-written Parquet fixture.
It also checks the packaged actual split/registry identities against their
recorded train/valid footer lengths and hashes. Root separately verified the
corrected `footer_bytes + 8` range against the mounted train and valid Parquet
files plus all eleven mounted fold-0 assets. No full raw-file rehash is claimed
by the small-fixture check.

R2 remains a separately dispatched attempt and is not edited or canceled by
this packet. R3 is an identified prepared correction only; dispatch still
requires R2 terminal reconciliation, a fresh quota/admission check, and root's
independent review of the final exact packet and source.
