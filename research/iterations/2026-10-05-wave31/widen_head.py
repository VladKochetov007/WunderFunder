from pathlib import Path
import json
import copy
import torch
from train_rnn import make_model, build_inputs
from wnn.data import SequenceFile

R = Path.home()/'WunderFunder-creative'
D = R/'research/iterations/2026-10-05-wave31'
torch.set_num_threads(2)
sf = SequenceFile(R/'datasets/valid.parquet')
raw = [torch.tensor(sf.read(g).features[:2000])[None] for g in (0, 5)]
report = []
for seed in (3, 4):
    parent_name = 'd256bc1k_f0' if seed == 3 else 'd256bc1ks4_f0'
    parent = R/f'artifacts/runs/{parent_name}/model_ep3.pt'
    ck = torch.load(parent, map_location='cpu', weights_only=False)
    control_args = copy.deepcopy(ck['args'])
    control_args.update(family_teachers='d256bc1k_f0,w6family_full,w19family_twostream,orgbase')
    args = copy.deepcopy(control_args)
    args.update(family_head_hidden=512, name=f'w31head_init_s{seed}')
    torch.manual_seed(seed)
    old = make_model(control_args, ck['n_in'], 23).eval()
    old_rng = torch.get_rng_state().clone()
    missing, unexpected = old.load_state_dict(ck['model'], strict=False)
    assert not unexpected and all(k.startswith('family_head.') for k in missing)
    torch.manual_seed(seed)
    wide = make_model(args, ck['n_in'], 23).eval()
    assert torch.equal(old_rng, torch.get_rng_state())
    src, dst = old.state_dict(), wide.state_dict()
    with torch.no_grad():
        for key in dst:
            if not key.startswith(('head.', 'family_head.')):
                dst[key].copy_(src[key])
        dst['head.0.weight'][:256].copy_(src['head.0.weight'])
        dst['head.0.bias'][:256].copy_(src['head.0.bias'])
        for key in ('head.2', 'family_head'):
            dst[key+'.weight'].zero_()
            dst[key+'.weight'][:, :256].copy_(src[key+'.weight'])
            dst[key+'.bias'].copy_(src[key+'.bias'])
    wide.load_state_dict(dst)
    errors = []
    with torch.no_grad():
        for xx in raw:
            x = build_inputs(xx)
            a, ha = old(x)
            b, hb = wide(x)
            assert torch.equal(ha, hb)
            err = float((a-b).abs().max())
            ferr = float((old.family_output-wide.family_output).abs().max())
            assert max(err, ferr) < 5e-5
            errors.append({'all_output_max_abs': err, 'family_output_max_abs': ferr})
    wide.zero_grad()
    pred, _ = wide(build_inputs(raw[0][:, :100]))
    pred[..., :2].square().mean().backward()
    assert wide.head[2].weight.grad[:, 256:].norm() > 0
    out = R/f'artifacts/runs/w31head_init_s{seed}'
    assert not out.exists()
    out.mkdir()
    torch.save({'model': wide.state_dict(), 'args': args, 'n_in': ck['n_in'], 'epoch': 0}, out/'model_ep0.pt')
    report.append({'seed': seed, 'parent': str(parent), 'hidden': 256, 'head_hidden': 512,
                   'constructor_rng_matches_control': True, 'gru_state_bit_exact': True,
                   'fresh_output_receives_gradient': True, 'parity': errors})
(D/'widening-parity.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
