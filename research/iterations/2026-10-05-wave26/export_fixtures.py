from pathlib import Path
import json
import numpy as np
import torch
from train_rnn import make_model, build_inputs
from wnn.data import SequenceFile

R = Path.home()/'WunderFunder-creative'
D = R/'research/iterations/2026-10-05-wave26'
torch.set_num_threads(2)
ck = torch.load(R/'artifacts/runs/w19distilled_s3/model_ep1.pt', map_location='cpu', weights_only=False)
sf = SequenceFile(R/'datasets/valid.parquet')
x = np.stack([sf.read(g).features[:2000] for g in (0, 5)])
probes = {'x': x}
for variant in ('uniform', 'soft', 'saturated', 'zero'):
    torch.manual_seed(126)
    mode = 'uniform' if variant == 'uniform' else 'soft'
    model = make_model(dict(ck['args'], family_class_mode=mode), ck['n_in'], 23).eval()
    missing, unexpected = model.load_state_dict(ck['model'], strict=False)
    assert sorted(missing) == ['class_res.bias', 'class_res.weight'] and not unexpected
    with torch.no_grad():
        if variant != 'zero':
            model.class_res.weight.normal_(0, .03)
            model.class_res.bias.normal_(0, .03)
        if variant == 'saturated':
            model.head[2].weight[17:21].zero_()
            model.head[2].bias[17:21].copy_(torch.tensor([1000., -1000., -1000., 999.]))
        probes[variant] = np.stack([model(build_inputs(torch.tensor(xx)[None]))[0][0, :, :2].numpy() for xx in x])
    state = model.state_dict()
    w = {k: v.numpy().copy() for k, v in state.items() if k in ('mean', 'std') or k.startswith(('inp.', 'gru.', 'head.', 'class_res.'))}
    w['class_gate.weight'] = w['head.2.weight'][17:21].copy()
    w['class_gate.bias'] = w['head.2.bias'][17:21].copy()
    w['head.2.weight'] = w['head.2.weight'][:2].copy()
    w['head.2.bias'] = w['head.2.bias'][:2].copy()
    w['meta.class_head_mode'] = np.array([0 if mode == 'uniform' else 1], dtype=np.float32)
    np.savez_compressed(D/f'fixture-{variant}.npz', **w)
np.savez_compressed(D/'fixtures-probe.npz', **probes)
print('Nonzero, uniform, saturated and zero residual fixtures exported', flush=True)
