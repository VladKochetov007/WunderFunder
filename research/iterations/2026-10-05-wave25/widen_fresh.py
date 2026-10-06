"""Preserve the h256 function while adding 128 independently initialized units."""
from pathlib import Path
import json
import copy
import torch
from train_rnn import make_model, build_inputs
from wnn.data import SequenceFile

R = Path.home()/'WunderFunder-creative'
D = R/'research/iterations/2026-10-05-wave25'


def main():
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sf = SequenceFile(R/'datasets/valid.parquet')
    raw = [torch.tensor(sf.read(g).features[:2000])[None] for g in (0, 5)]
    report = []
    for seed in (3, 4):
        parent = R/f'artifacts/runs/w19distilled_s{seed}/model_ep1.pt'
        ck = torch.load(parent, map_location='cpu', weights_only=False)
        old = make_model(ck['args'], ck['n_in'], 23).eval()
        old.load_state_dict(ck['model'])
        args = copy.deepcopy(ck['args'])
        args.update(hidden=384, name=f'w25fresh_init_s{seed}', widening_parent=str(parent))
        torch.manual_seed(20261006+seed)
        wide = make_model(args, ck['n_in'], 23).eval()
        src, dst = ck['model'], wide.state_dict()
        with torch.no_grad():
            for key in ('mean', 'std'):
                dst[key].copy_(src[key])
            for key in ('inp.weight', 'inp.bias'):
                dst[key][:256].copy_(src[key])
            for layer in range(args['layers']):
                for kind in ('ih', 'hh'):
                    key = f'gru.weight_{kind}_l{layer}'
                    for gate in range(3):
                        block = dst[key][gate*384:gate*384+256]
                        block.zero_()
                        block[:, :256].copy_(src[key][gate*256:(gate+1)*256])
                    key = f'gru.bias_{kind}_l{layer}'
                    for gate in range(3):
                        dst[key][gate*384:gate*384+256].copy_(src[key][gate*256:(gate+1)*256])
            dst['head.0.weight'][:256].zero_()
            dst['head.0.weight'][:256, :256].copy_(src['head.0.weight'][:, :256])
            dst['head.0.weight'][:256, 384:640].copy_(src['head.0.weight'][:, 256:])
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
                a, _ = old(x)
                b, _ = wide(x)
                err = float((a-b).abs().max())
                ferr = float((old.family_output-wide.family_output).abs().max())
                assert max(err, ferr) < 5e-5, (seed, err, ferr)
                errors.append({'all_output_max_abs': err, 'family_output_max_abs': ferr})
        # Confirm zero output weights can immediately learn from the fresh features.
        wide.zero_grad()
        out, _ = wide(build_inputs(raw[0][:, :100]))
        out[..., :2].square().mean().backward()
        fresh_gradient = float(wide.head[2].weight.grad[:, 256:].norm())
        assert fresh_gradient > 0
        target = R/f'artifacts/runs/w25fresh_init_s{seed}'
        assert not target.exists()
        target.mkdir()
        torch.save({'model': wide.state_dict(), 'args': args, 'n_in': ck['n_in'], 'epoch': 0}, target/'model_ep0.pt')
        report.append({'seed': seed, 'parent': str(parent), 'hidden': 384,
                       'initial_new_output_gradient_norm': fresh_gradient, 'parity': errors})
    (D/'widening-parity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
