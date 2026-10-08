import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    mo.md("# H274 blend lab — fold0 excluded-468 prediction panel")
    return


@app.cell(hide_code=True)
def _():
    import os
    import json
    import hashlib
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    import sys

    # Add src to path for metrics
    sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/src')

    import marimo as mo

    return Path, np, pd


@app.cell(hide_code=True)
def _(Path):

    ROOT = Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
    CAMPAIGN = ROOT / 'research/campaign_2026-10-06/bandit_continuation'
    EXEC = CAMPAIGN / 'continuous_execution_E529'

    # Prediction files to load
    PREDICTIONS = {
        'H272_parent': {
            'path': EXEC / 'exports/H272_PARENT_V2_CLEANFOLD_VALID/parent.npy',
            'wp_reported': 0.6810293773471175,
            'description': 'H272 parent (control baseline)',
        },
        'H272_hidden_only': {
            'path': EXEC / 'exports/H272_PARENT_V2_CLEANFOLD_VALID/hidden_only.npy',
            'wp_reported': 0.6810293773471175,
            'description': 'H272 parent hidden-only',
        },
        'H272_multiplicative': {
            'path': EXEC / 'exports/H272_PARENT_V2_CLEANFOLD_VALID/multiplicative.npy',
            'wp_reported': 0.6810293773471175,
            'description': 'H272 parent multiplicative',
        },
        'H233_hidden_only': {
            'path': EXEC / 'exports/H233_C1S3_E1_VALID_V2/pred_hidden_only.npy',
            'wp_reported': 0.681617903425,
            'description': 'H233 hidden-only',
        },
        'H233_additive': {
            'path': EXEC / 'exports/H233_C1S3_E1_VALID_V2/pred_additive.npy',
            'wp_reported': None,
            'description': 'H233 additive',
        },
        'H233_conditional_residual': {
            'path': EXEC / 'exports/H233_C1S3_E1_VALID_V2/pred_conditional_residual.npy',
            'wp_reported': 0.681618158484,
            'description': 'H233 conditional residual',
        },
        'H233_multiplicative': {
            'path': EXEC / 'exports/H233_C1S3_E1_VALID_V2/pred_multiplicative.npy',
            'wp_reported': 0.681617205539,
            'description': 'H233 multiplicative',
        },
        'H272_direct_hidden_only': {
            'path': EXEC / 'exports/H272_C1S3_DIRECT_LABEL_E1_VALID_V1/pred_hidden_only.npy',
            'wp_reported': None,
            'description': 'H272 direct label hidden-only',
        },
        'H272_direct_multiplicative': {
            'path': EXEC / 'exports/H272_C1S3_DIRECT_LABEL_E1_VALID_V1/pred_multiplicative.npy',
            'wp_reported': None,
            'description': 'H272 direct label multiplicative',
        },
    }

    print(f'Available prediction sets: {len(PREDICTIONS)}')
    print(f'Models: {len(set(k.split("_")[0] for k in PREDICTIONS))}')

    return EXEC, PREDICTIONS, ROOT


@app.cell(hide_code=True)
def _(EXEC, ROOT, np, pd):

    from wnn.gmetric import global_wp, reference_global_wp

    # Load evaluation masks and labels
    eval_mask_path = EXEC / 'exports/H272_PARENT_V2_CLEANFOLD_VALID/evaluation_group_mask.npy'
    eval_mask = np.load(eval_mask_path)
    eval_indices = np.where(eval_mask)[0]

    scored_mask_path = EXEC / 'exports/H272_PARENT_V2_CLEANFOLD_VALID/scored_mask.npy'
    scored_mask = np.load(scored_mask_path)

    print(f'Evaluation indices: {len(eval_indices)} rows out of {len(eval_mask)}')
    print(f'Scored mask shape: {scored_mask.shape}')

    # Load labels and row weights from valid data
    valid_parquet = ROOT / 'datasets/valid.parquet'
    df_valid = pd.read_parquet(valid_parquet, columns=['t0', 't1', 'step_in_seq'])
    print(f'Valid set shape: {df_valid.shape}')

    # Extract labels for evaluation rows
    targets = df_valid[['t0', 't1']].values
    print(f'Targets shape: {targets.shape}')
    print(f'Targets dtype: {targets.dtype}')

    return eval_indices, global_wp, scored_mask, targets


@app.cell(hide_code=True)
def _(PREDICTIONS, eval_indices, global_wp, np, pd, scored_mask, targets):

    # Load all prediction files, keeping only evaluation group rows
    predictions_loaded = {}
    verification_results = []

    for model_name, meta in PREDICTIONS.items():
        pred_path = meta['path']
        if pred_path.exists():
            # Load prediction array (full 1873 rows)
            pred = np.load(pred_path).astype(np.float32)
            # Keep only evaluation group rows (468 rows)
            pred_eval = pred[eval_indices]
            predictions_loaded[model_name] = pred_eval
        
            # Compute official WP score
            wp_computed = global_wp(targets[eval_indices], pred_eval, scored_mask[eval_indices])
        
            verification_results.append({
                'model': model_name,
                'wp_reported': meta['wp_reported'],
                'wp_computed': wp_computed,
                'match': (abs(wp_computed - (meta['wp_reported'] or wp_computed)) < 1e-8) if meta['wp_reported'] else None,
                'description': meta['description'],
            })
        
            print(f'{model_name}: computed WP = {wp_computed:.10f}')
        else:
            print(f'{model_name}: FILE NOT FOUND at {pred_path}')

    df_verify = pd.DataFrame(verification_results)
    print(f'\nLoaded {len(predictions_loaded)} prediction sets')
    print(f'Shape: {list(predictions_loaded.values())[0].shape}')

    return


if __name__ == "__main__":
    app.run()
