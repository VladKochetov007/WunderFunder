# Distillation facts (public, our setup)
Task: causal per-row regression on market streams (211 inputs/row; 2 targets t0,t1; metric = pooled weighted Pearson, weights |clip(y,2)|, rows>=99). Model: Linear+GELU encoder -> 2-layer GRU 256 (or 384) -> head [GRU;skip] -> 2 outputs + aux. Fold 1 validation, one-run noise sd ~0.002, seed sd ~0.001.
Teachers (3 epochs scratch): plain h256 ~0.6660; muon-all h384 mean 0.670 (5 seeds .6702/.6715/.6703/.6695/.6679); prop0 ~0.6678.
Student pipeline today: warm-start from a parent checkpoint (0.666 class), 3 epochs AdamW lr 1e-4, loss = 0.7 * MSE(student, mean teacher cached outputs) + 0.3 * label loss (weighted), teachers are 2 seeds' cached predictions on train split. Results: student from 2 h256 teachers 0.66938; from 2 h384 teachers 0.66836; NO teacher continuation (same schedule) 0.66861 => teacher signal worth only +0.0008.
Diverse pair blends (h256+h384 members) give +0.003 over a single; seeds of the same recipe give ~+0.001; members share 96-99.9% of residual.
Residual EDA: error concentrated in market class 3 (50% weight, 69% residual SS, local corr .44 vs .75-.81); 94% of residual is per-run level offset between label runs (labels step only at book events); label leads features ~1 row; t0/t1 difference mode under-dispersed.
Serving constraint: final model must run CPU int8 within ~13 portal-min per model (GRU 256 ok; 384 ~2x cost). Training-time compute is relaxed. Teachers may be much larger / ensembles; student must be small.
Queued/untested: out-of-sample path distillation (augmented random paths fed to teachers, student trained on teacher outputs); multi-teacher; hidden-state matching.
Not available: any external data or pretrained weights beyond supplied data.
