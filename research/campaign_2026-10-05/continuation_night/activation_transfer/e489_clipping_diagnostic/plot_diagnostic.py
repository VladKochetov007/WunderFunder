"""Plot measured clipping coverage and optimizer-parity diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    here = Path(__file__).resolve().parent
    report = json.loads((here / 'actual_v1/report.json').read_text())
    seeds = report['seeds']
    output = here / 'eda/v1'
    output.mkdir(parents=True, exist_ok=True)
    labels = [str(row['seed']) for row in seeds]
    positions = np.arange(len(seeds))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    norms = [row['native_preclip']['preclip_global_norm'] for row in seeds]
    axes[0].bar(positions, norms, color=['#3675a9' if value > 1 else '#c88739' for value in norms])
    axes[0].axhline(1, color='black', linestyle='--', label='Clipping threshold')
    axes[0].set(xticks=positions, xticklabels=labels, ylabel='Global gradient norm', title='Original three seeds: clipping coverage')
    axes[0].legend()
    for offset, key, label in [(-0.18, 'native_custom_raw_update_deltas', 'Raw step'), (0.18, 'native_custom_clip1_update_deltas', 'Clipped step')]:
        values = [max(row[key]['adamw']['max_abs_error'], 1e-12) for row in seeds]
        axes[1].bar(positions + offset, values, width=0.36, label=label)
    axes[1].axhline(2e-5, color='black', linestyle='--', label='Original absolute-error limit')
    axes[1].set(xticks=positions, xticklabels=labels, ylabel='Maximum absolute AdamW update difference', title='Native versus custom recurrence: one update', yscale='log')
    axes[1].legend(fontsize=8)
    figure.suptitle('E499 diagnostic only: no training fit or score')
    figure.savefig(output / 'clipping_and_update_parity.png', dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    main()
