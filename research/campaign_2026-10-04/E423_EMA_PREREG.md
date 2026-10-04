# E423 EMA retention in rows

Run the same fold0/seed3 combo for4 epochs, batch32, chunk1000, fixed epoch3. Only intervention: EMA decay is interpreted over2000 reference rows, so the1000-row optimizer update uses sqrt(0.999)=0.999499874937461. The unchanged reference uses0.999 each update. Same TRAIN/development mixture, teachers, alpha0.5, stopped bootstrap8, Muon, auxiliary objectives and schedule. Evaluation only after fitting, no epoch selection.

Compare paired component Global WP with unchanged combo; report all class deltas. One previously exposed fold/seed: no independent/public claim. If source-identity inspection reveals any control drift, report it and fit a same-source control before interpreting the comparison. Longer-credit and pooled-loss accumulation experiments remain separately planned.

Same allocations as already successful batch32 combo:8CPU24GiBhost,4.2GBallocator,4.8GBobservedstop. Guard stops resource violations. Full code closure and inputs sealed; interrupted attempts remain preserved. This is a predictive experiment, not a short harness benchmark.

A fresh same-source full-schedule control is queued after the EMA arm, followed by common-FP32 full-valid exports and paired scoring. Historical combo is descriptive only. Controller `checks/run_ema_sequence.py` stops on any guard incident; it never automatically acknowledges or restarts failed fits.
