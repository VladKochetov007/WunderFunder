# Research byte artifacts

- [Eight-L4 exact source R9](quality8-source-r9-20261028.tar.gz) preserves the
  delivered source/test/config bytes of commit50053909, including receiving line
  endings, for training-source SHA `f9685716d64ecb5ceef2ec821781fa4078ff21661201edfcdac8292867990230`.
  No dataset, credential value or model weight is included. Re-extract these
  bytes when replaying the strict source identity; Git newline conversion can
  otherwise produce a different byte digest.
- [Eight-L4 engineering evidence R4](quality8-engineering-r4-20261028.tar.gz)
  contains eight original unapproved profile receipts, sampled diagnostics/logs,
  independent resource review, source audit/index/split metadata and the two
  excluded startup-failure descriptions. These are engineering observations,
  not scientific endpoint scores or weights.

[Artifact hashes and sizes](../experiment_drafts/galam-l4-quality8-20261028/launch/ARTIFACTS.json),
[active experiment](../experiment_drafts/galam-l4-quality8-20261028/launch/README.md)
and [first intermediate scientific epoch](../experiment_drafts/galam-l4-quality8-20261028/launch/FIRST_EPOCH_RESULT.json)
provide provenance and current status. Other historical artifact files in this
directory retain their original campaign references.
