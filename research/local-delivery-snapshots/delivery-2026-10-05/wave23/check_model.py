from pathlib import Path
import json
import torch
from train_rnn import make_model, build_inputs
from wnn.data import SequenceFile

R = Path.home() / 'WunderFunder-creative'
D = R / 'research/iterations/2026-10-05-wave23'
torch.set_num_threads(2)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
ck = torch.load(R/'artifacts/runs/w19distilled_s3/model_ep1.pt', map_location='cpu', weights_only=False)
x = build_inputs(torch.tensor(SequenceFile(R/'datasets/valid.parquet').read(0).features[:2000])[None])
base = make_model(ck['args'], ck['n_in'], 23).eval()
base.load_state_dict(ck['model'])
with torch.no_grad():
    reference, _ = base(x)
    routing = reference[..., 17:21].softmax(-1)
    routing = routing / routing.square().sum(-1, keepdim=True).sqrt()
    assert (routing.square().sum(-1)-1).abs().max() < 1e-6
report = {}
for mode in ('uniform', 'soft'):
    torch.manual_seed(3)
    model = make_model(dict(ck['args'], family_class_mode=mode), ck['n_in'], 23).eval()
    missing, unexpected = model.load_state_dict(ck['model'], strict=False)
    assert sorted(missing) == ['class_res.bias', 'class_res.weight'] and not unexpected
    with torch.no_grad():
        got, _ = model(x)
        assert torch.equal(reference, got)
        model.class_res.weight.normal_(0, 0.01)
        model.class_res.bias.normal_(0, 0.01)
        full, _ = model(x)
        prefix, h = model(x[:, :1000])
        suffix, _ = model(x[:, 1000:], h)
        err = (torch.cat((prefix, suffix), 1) - full).abs().max().item()
        assert err < 5e-5, err
    model.zero_grad()
    pred, _ = model(x[:, :100])
    pred[..., :2].square().mean().backward()
    assert model.class_res.weight.grad.norm() > 0
    # Main regression loss must not bypass the auxiliary class supervision.
    assert model.head[2].weight.grad[17:21].abs().max() == 0
    report[mode] = {'zero_init_bit_exact': True, 'chunk_causality_max_abs': err,
                    'gate_detached': True, 'routing_squared_norm': 1.0,
                    'parameters': sum(p.numel() for p in model.parameters()),
                    'residual_gradient_norm': model.class_res.weight.grad.norm().item()}
assert report['uniform']['parameters'] == report['soft']['parameters']
(D/'model-check.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
