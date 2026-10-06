"""One root-allocated generated CUDA feasibility probe; no data/checkpoint I/O."""
import copy
import hashlib
import json
from pathlib import Path
import time
import torch
from minimal_rnn import MinimalRNN

torch.set_num_threads(1)
torch.manual_seed(3)
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
torch.cuda.set_per_process_memory_fraction(.50)
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')
torch.cuda.reset_peak_memory_stats()
begin = time.perf_counter()

# Full untruncated resident prefix and raw current/next-batch reservation.
# Values are generated on GPU; no host copy or competition input is involved.
prefix = torch.randn(32, 20000, 211, device=device)
raw = torch.zeros(32, 20000, 112, device=device)
prefetched_raw = torch.zeros_like(raw)
targets = .2 * torch.randn(32, 20000, 23, device=device)
model = MinimalRNN(n_in=211, hidden=256, layers=2, n_out=23).to(device)
teacher = copy.deepcopy(model).eval().requires_grad_(False)
optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01, fused=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.001, total_steps=15000,
                                               pct_start=.05, final_div_factor=100)
scaler = torch.amp.GradScaler('cuda')
state = teacher_state = None
records = []
for step in range(3):
    x = prefix[:, step*2000:(step+1)*2000]
    y = targets[:, step*2000:(step+1)*2000]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        reference, teacher_state = teacher(x, teacher_state)
        teacher_state = teacher_state.detach()
    with torch.autocast('cuda', dtype=torch.float16):
        out, state = model(x, state)
        state = state.detach()
        # Generated main/aux loss exercises all23 outputs; not a forecast objective qualification.
        main = (out[..., :2].float() - y[..., :2]).square().mean()
        aux = (out[..., 2:].float() - y[..., 2:]).square().mean()
        consistency = (out[..., :2].float() - reference[..., :2].float()).square().mean()
        loss = main + .3*aux + consistency
    optimizer.zero_grad(set_to_none=True)
    old_scale = scaler.get_scale()
    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    scaler.step(optimizer); scaler.update(); scheduler.step()
    with torch.no_grad():
        for destination, source in zip(teacher.parameters(), model.parameters()):
            destination.lerp_(source, .001)
    torch.cuda.synchronize()
    record = {'step': step, 'warmup': step==0, 'seconds': time.perf_counter()-t0,
              'loss': float(loss.detach()), 'gradient_norm': float(norm.detach()),
              'scale_before': old_scale, 'scale_after': scaler.get_scale(),
              'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'model_finite': all(bool(torch.isfinite(p).all()) for p in model.parameters()),
              'gradients_finite': all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())}
    records.append(record); print(json.dumps({'step': record}), flush=True)
    assert torch.isfinite(loss) and torch.isfinite(norm) and record['model_finite'] and record['gradients_finite']
    assert record['scale_after'] >= old_scale, 'Probe requires finite actual updates; no overflow retry'

result = {'status': 'PASS_GENERATED_GPU_MEMORY_FINITE_GRADIENT', 'steps': records,
          'model_source_sha256': hashlib.sha256((Path(__file__).parent/'minimal_rnn.py').read_bytes()).hexdigest(),
          'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'shape': [32, 2000, 211], 'hidden': 256, 'layers': 2, 'outputs': 23,
          'full_prefix': list(prefix.shape), 'raw': list(raw.shape), 'prefetched_raw': list(prefetched_raw.shape),
          'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
          'total_seconds': time.perf_counter()-begin, 'torch_version': torch.__version__,
          'device': torch.cuda.get_device_name(0), 'GPU_total_bytes': torch.cuda.get_device_properties(0).total_memory,
          'flags': {'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                    'float32_matmul_precision': torch.get_float32_matmul_precision()},
          'allocator_fraction': .50, 'real_data_or_trained_checkpoints_read': False,
          'forecast_fit_or_metric': False, 'precision_or_native_parity_claimed': False,
          'limitations': ['Synthetic squared main/aux loss, not the full forecast objective/loader.',
                         'Current prototype retains finite/nonnegative state host-sync checks.',
                         'Forward autocast gate projections differ from FP32 streaming; AMP/native parity unqualified.',
                         'Three steps do not qualify multi-epoch allocator growth or full-training resource use.']}
print(json.dumps({'result': result}), flush=True)
