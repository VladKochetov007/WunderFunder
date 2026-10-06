"""Training-only scale diagnostic; does not evaluate a new predictor."""
from pathlib import Path
import json
import numpy as np
from wnn.data import SequenceFile

R = Path.home()/'WunderFunder-creative'
D = R/'research/iterations/2026-10-05-wave28'
groups = np.load(R/'cache/train_targets.npz')['row_group']
ids = np.linspace(0, len(groups)-1, 64, dtype=int)
names = ['d256bc1k_f0', 'w6family_full', 'w19family_twostream', 'orgbase', 'w19teacher']
arrays = {n: np.load(R/f'cache/pred_{n}_train.npy', mmap_mode='r') for n in names}
prop = np.load(R/'cache/propcache_prop_f0_train.npy', mmap_mode='r')
sf = SequenceFile(R/'datasets/train.parquet')
moment = {n: np.zeros((2, 3)) for n in names+['true_clipped']}
for i in ids:
    g = int(groups[i])
    y = np.clip(sf.read(g).targets, -2, 2).astype(np.float64)
    q = np.asarray(prop[g], dtype=np.float64)**2+.03
    q[:99] = 0
    w = abs(y)*q[:, None]
    for name in moment:
        p = y if name == 'true_clipped' else np.asarray(arrays[name][i], dtype=np.float64)
        assert np.isfinite(p).all()
        moment[name] += np.stack([w.sum(0), (w*p).sum(0), (w*p*p).sum(0)], -1)
report = {'scope': '64 fixed training sequences, no new model validation or labels from held-out data. Weights abs(clip(y))*(prop^2+.03), first99 excluded. Squared target energy is an initialization-scale diagnostic, not the actual learned family-head loss.', 'families': {}}
for name, m in moment.items():
    mean = m[:, 1]/m[:, 0]
    ms = m[:, 2]/m[:, 0]
    report['families'][name] = {'mean': mean.tolist(), 'weighted_rms': np.sqrt(ms).tolist(),
                                 'weighted_std': np.sqrt(np.maximum(ms-mean*mean, 0)).tolist()}
energy = np.array([moment[n][:, 2].sum() for n in names[:4]])
report['four_family_target_energy_fraction'] = dict(zip(names[:4], (energy/energy.sum()).tolist()))
(D/'family-scale.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
