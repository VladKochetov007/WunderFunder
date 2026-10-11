# Paid artifact preservation coverage reconciliation (R1)

This bounded, read-only pass reconciles the lane-B master inventory `b4e496252c4b…94c6` (2,736 rows; 31,355,639,635 bytes). The complete collection gate remains **false**. This report does not claim every manifest row was collected.

| Class | Rows | Bytes | Interpretation |
|---|---:|---:|---|
| Receipt-qualified rescued or preserved | 1,270 | 4,245,326,500 | Selected archive/state/checkpoint/prediction receipts match manifest SHA and size; destinations were checked by `lstat` for regular-file type and expected size. |
| Local, size-matched, not receipt-verified here | 1,410 | 18,718,757,920 | Local path and size match; no selected receipt supplied an accepted SHA match. |
| Missing prediction pairs with checkpoint candidates | 46 files / 23 pairs | 6,893,063,192 | Re-export candidates only after exact source, predictor, dataset and checkpoint selection are rebound. |
| Missing prediction pairs without a located checkpoint | 10 files / 5 pairs | 1,498,492,023 | LSA epoch outputs below; bounded search found no matching local checkpoint/provenance. |

The five missing pairs are `vt_f1_lsa_ep3`, `vt_f1_lsa_ep7`, `vt_f1_lsa_ep11`, `vt_f1_lsa_ep15` and `vt_f1_lsa_ep19` (`pred_…_valid.npy` plus `.json`). This does not prove the weights are globally irrecoverable. The other 23 missing pairs have checkpoint rows and local history/provenance files, but their predictions were not regenerated and byte identity is not claimed. No non-prediction log/history/provenance/control row was found missing in this manifest mapping.

All 149 archive checkpoint rows (725,061,662 bytes) have a SHA coverage PASS; the 1,179-member artifact archive and 22-state archive bind to the master manifest. Twenty selected critical prediction files across H390 s3/s4, H391, SWN/SWN25, H388 s3/s4/s5 and OOD splice/spmix also match their receipts and manifest rows. Their receipt hashes are listed in the JSON. This reconciliation did not rehash binary files.

The H390 `s3_r3` and `s4_r3` `model_ep5.pt` files remain distinct by run ID and SHA; each is separate from that run's `model.pt`. Nothing was overwritten or coalesced. The reported instance credit-expiry exit is an operational stop, not a scientific finding.

Source: [`PAID_PRESERVATION_COVERAGE_RECONCILIATION_R1.json`](PAID_PRESERVATION_COVERAGE_RECONCILIATION_R1.json).
