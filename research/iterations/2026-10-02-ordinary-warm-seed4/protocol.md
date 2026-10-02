# Conditional seed4 ordinary-warm replication

Only activate through a separate root allocation after seed3 ordinary-warm results justify replication. Preparation authorizes no training, OOF, score, final production fit or public/GitHub action.

Primary estimand is **seed4 warm4 minus the ORIGINAL seed4 four-member ensemble**, with `cont_sh256_s4h2` and unchanged `on_ctl128s3,on_ctl128s4,on_ctlc500`, equal quarters in that order. Operational comparison with current seed3 student4 is secondary. This avoids confusing warm adaptation with seed replacement. The original completed seed4 checkpoint/OOF/receipt provenance is copied into baseline-cache-manifest.json; no new baseline fit is needed.

Start each fold from original `cont_sh256_s4_fF/model_ep3.pt`, preserve all12 ordinary state tensors and mean/std. One new all-layer parent epoch lr.0005, unchanged original h256 seed3+4 teachers/alpha.5/EMA1, batch32/chunk2000/masks/aux. Fixed parent EMAep0; strict own-parent load, then2epoch H2 with `inp,gru` frozen, batch16 and fixed ep1. Fourfits/twoOOF, seed4 only, no optimizer/epoch/alpha sweep.

train_warm.py and check_checkpoint.py are byte-identical to active W. Predictor changes only the s4 name guard; generated test changes only manual_seed3→4. Factory consumes and discards the same interval-extra constructor draws at both stages, retains only the ordinary model, and CPU parity initialization uses fork_rng. All103 resolved training args are reconstructed independently from frozen parser declarations and compared against original seed4 and W seed3 recipes. Differences from W are names, seed and own/original parent paths only.

No checkpoint tensors, target arrays, predictions or quality values are read during preparation. Parent hashes and baseline cache hashes come from completed seed4 inspection/bindings, cross-checked against the already frozen alpha.7 seed4 reserve metadata. Expected allocated duration12–22minutes on one A4000 lane, resource allocation/finish/grader to be reused separately by root only if activated.
