"""Label-free diagnostic: did cloned units actually diverge during training?"""
from pathlib import Path
import json
import numpy as np
import torch
from train_rnn import make_model, build_inputs
from wnn.data import SequenceFile

R = Path.home()/'WunderFunder-creative'
D = R/'research/iterations/2026-10-05-wave22'
torch.set_num_threads(2)
sf = SequenceFile(R/'datasets/valid.parquet')
inputs = [build_inputs(torch.tensor(sf.read(g).features[:2000])[None]) for g in (0, 5, 12)]


def stats(rows):
    x = np.concatenate(rows).astype(np.float64)
    a, b = x[:, :128], x[:, 256:]
    ac, bc = a-a.mean(0), b-b.mean(0)
    cor = (ac*bc).sum(0)/np.sqrt((ac*ac).sum(0)*(bc*bc).sum(0)+1e-30)
    return {'clone_correlation_quantiles': np.quantile(cor, [0, .25, .5, .75, 1]).tolist(),
            'relative_clone_rms_difference': float(np.sqrt(np.mean((a-b)**2)/np.mean(a*a))),
            'fraction_clone_correlation_gt_099': float(np.mean(cor > .99))}


report = {'scope': 'Label-free mechanistic audit on first2000 rows of valid sequences0,5,12. Measures clone divergence, not forecast utility or a fresh validation score.', 'results': {}}
for seed in (3, 4):
    for stage, name, epoch in (('initial', f'w22wide_init_s{seed}', 0), ('trained', f'w22_wide_s{seed}', 1)):
        ck = torch.load(R/f'artifacts/runs/{name}/model_ep{epoch}.pt', map_location='cpu', weights_only=False)
        model = make_model(ck['args'], ck['n_in'], 23).eval()
        model.load_state_dict(ck['model'])
        hidden, readout = [], []
        with torch.no_grad():
            for x in inputs:
                z = model.encode((x-model.mean)/model.std)
                o, _ = model.gru(z)
                f = model.head[1](model.head[0](torch.cat((o, z), -1)))
                hidden.append(o[0].numpy())
                readout.append(f[0].numpy())
        report['results'][f'{stage}_s{seed}'] = {'gru': stats(hidden), 'readout': stats(readout)}
(D/'widening-utilization.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
