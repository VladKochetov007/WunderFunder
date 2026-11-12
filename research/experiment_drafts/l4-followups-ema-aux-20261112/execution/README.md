# Execution and readout provenance

Frozen ROOT controller, clock, admission, owner, fit-result and first readout bytes live under `pinned/` and `receipts-r1/`. Later progress is in `receipts-r2/`; it does not replace or mutate the R1 package. `PUBLICATION_PINS.json` lists SHA-256 and byte length for each frozen file, excluding itself. `.gitattributes` disables line-ending conversion on all pinned and receipt paths.

## Frozen identities

- Follow-up controller `pinned/followups/root_fits_r4.py`: SHA-256 `7a0277192ff1e98c2b44391a0573b64b32d0df206ee8711add95b1df0600b870`; follow-up source SHA `29fb45164667e69a71d4be2c696731d9824884d2d51674f37d2a633bd17d9511`; ROOT admission `receipts-r1/l4-publication-admission-receipts-r1/followups/ROOT-fit-admission-r2.json`, SHA-256 `c8ae408781b42cb029d55c997107fc56dd20722d16c65e4ecc53ad794f9b857a`.
- EMA controller `pinned/ema/root_fits_ema_r2.py`: SHA-256 `c1fbb0f4a3888a6ddcdabdd737a3a489f97713b4812db0cee62e2421c64540ef`; EMA source SHA `33574f496a2640e1a4c14690c8858eeaad5075e21d5b96042b746d35ad55810a`; ROOT admission `receipts-r1/l4-publication-admission-receipts-r1/ema/ROOT-fit-admission-ema-r2.json`, SHA-256 `f84e496b7bce86d5c39963782eb3650ea3aba34b1ff33dd6ac9cb1701ce2d0ef`.
- Fixed clock `pinned/CLOCK_R1.json`: SHA-256 `0181e9889a45d0636e3b9981ddb15e17fb5cd26429068bb0ee37f78d09d71270`.
- Canonical R3 receiver `pinned/readout/canonical_readouts_r3.py`: SHA-256 `09bd67033ba59943ac4c19a3f5ffcba4c1a9e20ca2d5ad8a2cd877d65972fae6`; unchanged helper `pinned/readout/helper/readout_r2.py`: SHA-256 `fadb34135dbf3ac6b5f3c5ab4d316e892d9ea16e6094bbc999cf5344ba94eb68`. The receiver repair receipt records source unchanged and the per-cell helper namespace fix.
- Admission receipt archive `pinned/l4-publication-admission-receipts-r1.tar.gz`: SHA-256 `f8fcac229618426d4ac108476d30abaf09c9a0bd23761d6dfdee3286f098b695` (30,873 bytes). Its 31-file manifest was verified against every copied file.

## Results and repairs

The R1 bundle preserves the original admissions, per-cell fit results, owner/completion receipts and first canonical readout results. It records four completed Muon/since cells, two completed raw EMA controls, and two timed-out/partial EMA treatments. The partial treatment results do not contain committed E6 checkpoint weights.

`receipts-r2/049-readouts-canonical-tail.json` records all eight first-cohort R3 readouts as complete, with full-valid WP / target scores within `1e-5` of their epoch monitors. In that same R3 run, the four completed Muon/since cells and two completed raw EMA controls were rejected because the prepared source data could not be found under `<fit-root>/project/datasets/`; it did not compare against the right path even though the pinned data root was present.

The repair uses exact symlinks from the follow-up and EMA fit roots' `project` paths to `/data/lc-epanov_s/tabdata-stem-successors-20261111-r1/project`, matching the ROOT-pinned data source. `receipts-r2/050-start-layout-repaired-canonical-r4.json` records both targets, the unchanged R3 receiver SHA, and all eight preseeded first-cohort results with verified receipt/model hashes. It starts fresh follow-up and EMA readout output roots; their R4 results remain pending in this snapshot. Keep R3 rejection receipts intact.

No new fit was launched from this publication packet. Fit sources, admissions, profiles and endpoint identities remain ROOT-owned; this document only records their exact evidence.