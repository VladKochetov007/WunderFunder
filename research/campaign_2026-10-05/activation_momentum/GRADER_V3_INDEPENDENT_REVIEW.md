# E483 versioned grader v3 review

Root inspected v2→v3: a single checked canonical full arm key now drives score storage/export ordering, with an explicit per-seed short-key to full-key mapping for unchanged27 contrast formulas. No recurrence, inference, normalizer, fit proof or scoring arithmetic changed. Root independently reran all9 tests under1CPU/2GiB/no swap. The added synthetic end-to-end score_all plus write_outputs test now assembles all18 arms/27 contrasts and checks every export dimension, using synthetic checkpoints and shortened synthetic sequence length only. All9 tests pass, including the direct real sealed-spec identity contract with no row loading.

Frozen scorer SHA1be6039a6a11553470d93736e2bb20470f1516d603fdc902cbf61f5c7ea2cb58; tests643608a904de22e4f1646d217d4472c05eab6b099a060b6d133d89a9105fdbd4.
GO for the actual v3 fit-only proof and a separately sealed explicit CPU scoring attempt after it passes. Preserve both earlier incident records, frozen versions, v2 sealed score attempt and raw log. Previous v2 inference read the exposed panel but wrote no results; it is not counted as18 successful grades.
