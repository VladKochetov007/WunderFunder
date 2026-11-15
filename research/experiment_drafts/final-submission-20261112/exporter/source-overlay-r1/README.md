# Source4 consumer compatibility repair

- `scripts/package_final_tl3.py` is the hash-pinned successor consumer; the historical checkout stays unchanged.
- `repair-receipt-r1.json` records source/repair identities and the preparation-time status.
- `test_source4_history_compat.py` tests exact historical fitting counters and rejects real validation fields. The actual seven-test result is in [build evidence](../../build/ROOT_COMPAT_TEST_RUN_R1.json).
