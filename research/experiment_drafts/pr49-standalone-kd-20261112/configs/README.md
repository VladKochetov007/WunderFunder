# Source-matched cold GRU controls and KD treatments

The four JSON files use seeds203501/203502 and KD coefficients0/0.2. Within each seed only `kd_weight` changes. Both arms use the same fitting-only cache and fixed six-epoch recipe. `../PANEL_MANIFEST_R2.json` binds each exact file SHA-256, committed source, teacher lineage and unchanged validation policy. These are frozen launch configs, not a tuning grid or a restart instruction.
