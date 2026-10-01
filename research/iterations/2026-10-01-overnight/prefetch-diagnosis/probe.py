"""Generated-only allocator-stream diagnostic. Never imports the forecast trainer."""
from pathlib import Path
import argparse, ast, hashlib, json, os, time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TRAINER = ROOT / 'research/iterations/2026-10-01-overnight/training/train_aux.py'
TRAINER_SHA = 'f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_prefetcher():
    assert sha(TRAINER) == TRAINER_SHA
    source = TRAINER.read_text()
    matches = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'CudaPrefetcher']
    assert len(matches) == 1
    node = matches[0]
    return '\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno]) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['per_epoch', 'reuse'], required=True)
    ap.add_argument('--allocation', type=Path, required=True)
    ap.add_argument('--allocation-sha256', required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    assert sha(a.allocation) == a.allocation_sha256
    allocation = json.loads(a.allocation.read_text())
    assert allocation['prefetch_generated_pair_authorized'] is True
    assert not allocation.get('competition_inputs_authorized', False)
    assert not allocation.get('forecast_fits_authorized', False)
    plan = json.loads((HERE / 'plan.json').read_text())
    assert allocation['plan_sha256'] == sha(HERE / 'plan.json')
    for rel, expected in plan['source_sha256'].items():
        assert sha(ROOT / rel) == expected, rel
    assert not a.output.exists()
    assert os.environ.get('PYTORCH_ALLOC_CONF', '') == ''
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '') == ''
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.memory.get_allocator_backend() == 'native'
    device = torch.device('cuda:0')
    # Native allocator behavior is left unchanged: no fraction, empty_cache,
    # or allocator configuration. Boundaries and external NVML monitor abort
    # on limits without first reclaiming or resizing allocator reservations.
    torch.manual_seed(2026100161)
    torch.cuda.manual_seed_all(2026100161)
    scope = {'torch': torch, 'time': time}
    extracted = extract_prefetcher()
    assert hashlib.sha256(extracted.encode()).hexdigest() == plan['extracted_class_sha256']
    exec(compile(extracted, str(TRAINER) + ':CudaPrefetcher', 'exec'), scope)
    CudaPrefetcher = scope['CudaPrefetcher']

    def tensor_hash(value):
        return hashlib.sha256(memoryview(value.detach().numpy()).cast('B')).hexdigest()

    fixture_started = time.perf_counter()
    generator = torch.Generator(device='cpu').manual_seed(2026100162)
    batches = []
    fixture = []
    for batch_id in range(2):
        x = torch.empty((32, 20000, 211), dtype=torch.float32, pin_memory=True)
        x.uniform_(-1., 1., generator=generator)
        y = torch.empty((32, 20000, 2), dtype=torch.float32, pin_memory=True)
        y.uniform_(-.1, .1, generator=generator)
        scored = torch.zeros((32, 20000), dtype=torch.bool, pin_memory=True)
        scored[:, 99 + batch_id::2] = True
        prop = torch.empty((32, 20000), dtype=torch.float32, pin_memory=True)
        prop.uniform_(.1, .9, generator=generator)
        values = (x, y, scored, prop)
        batches.append(values)
        fixture.append({'batch_id': batch_id, 'tensors': [
            {'name': name, 'shape': list(v.shape), 'dtype': str(v.dtype),
             'bytes': v.numel() * v.element_size(), 'sha256': tensor_hash(v), 'pinned': v.is_pinned()}
            for name, v in zip(['x', 'y', 'scored', 'prop'], values)]})
    fixture_seconds = time.perf_counter() - fixture_started
    fixture_sha = hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest()

    class FixedLoader:
        def __len__(self):
            return 32
        def __iter__(self):
            for i in range(32):
                yield batches[i % 2]

    class Consumer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            init = torch.Generator(device='cpu').manual_seed(2026100163)
            self.weight = torch.nn.Parameter(torch.empty(211, 2).uniform_(-.02, .02, generator=init))
            self.bias = torch.nn.Parameter(torch.tensor([.001, -.001], dtype=torch.float32))
        def forward(self, features):
            return features @ self.weight + self.bias

    model = Consumer().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=0., weight_decay=0., foreach=False, fused=False)
    loader = FixedLoader()
    records = []
    all_traces = {}
    current = None

    def memory():
        stats = torch.cuda.memory_stats()
        return {name: int(stats[name]) for name in [
            'allocated_bytes.all.current', 'reserved_bytes.all.current',
            'allocated_bytes.all.peak', 'reserved_bytes.all.peak',
            'inactive_split_bytes.all.current', 'inactive_split_bytes.all.peak',
            'num_alloc_retries', 'num_ooms']}

    def rng():
        return {'cpu': tensor_hash(torch.get_rng_state()), 'cuda': tensor_hash(torch.cuda.get_rng_state()),
                'fixture_private_cpu': tensor_hash(generator.get_state())}

    # Exactly one initial sync and one explicit sync after each of four epochs.
    # All trace readback occurs at these epoch boundaries, never per batch.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial_state = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    initial_rng = rng()
    baseline_memory = memory()
    loop_started = time.perf_counter()
    for epoch in range(4):
        before = memory()
        epoch_started = time.perf_counter()
        if a.arm == 'per_epoch' or current is None:
            current = CudaPrefetcher(loader, device)
        stream_id = int(current.stream.cuda_stream)
        device_trace = {k: [] for k in ['batch_checksum', 'features', 'target', 'loss', 'gradient', 'update']}
        for index, (dev, scored_cpu) in enumerate(current):
            x, y, scored, prop = dev
            features = x.mean(dim=1)
            target = y.mean(dim=1)
            sample_weight = (scored.float() * prop).mean(dim=1)
            prediction = model(features)
            loss = ((prediction - target).square().sum(dim=1) * sample_weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            device_trace['gradient'].append(torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()]))
            optimizer.step()
            device_trace['update'].append(torch.cat([p.detach().reshape(-1) for p in model.parameters()]))
            # Device traces are evidence, not a collision-free full GPU tensor hash.
            # Full pinned CPU tensor hashes and exact loader order are also bound.
            checks = [v.view(torch.int32).sum(dtype=torch.int64) for v in (x, y, prop)]
            checks.append(scored.sum(dtype=torch.int64))
            device_trace['batch_checksum'].append(torch.stack(checks))
            device_trace['features'].append(features.detach().clone())
            device_trace['target'].append(target.detach().clone())
            device_trace['loss'].append(loss.detach().clone())
        assert index == 31
        torch.cuda.synchronize()
        epoch_wall = time.perf_counter() - epoch_started
        after = memory()
        epoch_trace = {}
        for name, values in device_trace.items():
            array = torch.stack(values).cpu().numpy()
            all_traces[f'epoch{epoch}_{name}'] = array
            epoch_trace[name] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                                 'sha256': hashlib.sha256(memoryview(array).cast('B')).hexdigest()}
            assert np.isfinite(array).all(), name
        records.append({'epoch': epoch, 'stream_id': stream_id, 'batch_ids': [i % 2 for i in range(32)],
                        'memory_before': before, 'memory_after': after, 'epoch_wall_seconds': epoch_wall,
                        'rng': rng(), 'trace': epoch_trace})
        print(json.dumps({'epoch_boundary': records[-1]}), flush=True)
        assert after['reserved_bytes.all.peak'] <= (5 << 30), 'natural allocator peak exceeded boundary limit'
    loop_seconds = time.perf_counter() - loop_started
    for name, value in model.state_dict().items():
        all_traces['initial_' + name] = initial_state[name]
        all_traces['final_' + name] = value.detach().cpu().numpy().copy()
        assert np.isfinite(all_traces['final_' + name]).all()
    assert any(not np.array_equal(all_traces['initial_' + name], all_traces['final_' + name]) for name in initial_state)
    trace_path = a.output.with_suffix('.npz')
    assert not trace_path.exists()
    np.savez(trace_path, **all_traces)
    final_memory = memory()
    result = {'status': 'COMPLETE', 'arm': a.arm, 'source_sha256': sha(__file__),
              'allocation_sha256': a.allocation_sha256, 'plan_sha256': sha(HERE / 'plan.json'),
              'trainer_sha256': TRAINER_SHA, 'extracted_class_sha256': hashlib.sha256(extracted.encode()).hexdigest(),
              'generated_only': True, 'competition_inputs_read': False, 'forecast_fits': 0, 'WP_queries': 0,
              'fixture': fixture, 'fixture_sha256': fixture_sha, 'fixture_seconds': fixture_seconds,
              'fixture_total_bytes': sum(v['bytes'] for b in fixture for v in b['tensors']),
              'initial_rng': initial_rng, 'initial_memory': baseline_memory, 'epochs': records,
              'final_memory': final_memory, 'peak_reserved_bytes': final_memory['reserved_bytes.all.peak'],
              'loop_wall_seconds': loop_seconds, 'sum_epoch_wall_seconds': sum(r['epoch_wall_seconds'] for r in records),
              'trace_file': str(trace_path.relative_to(ROOT)), 'trace_file_sha256': sha(trace_path),
              'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
              'device_name': torch.cuda.get_device_name(), 'allocator_backend': torch.cuda.memory.get_allocator_backend(),
              'allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'deterministic_algorithms': True,
              'GPU_allocator_fraction': None, 'GPU_boundary_reserved_limit_bytes': 5 << 30, 'synchronization': 'one explicit initial plus four explicit end-of-epoch; boundary-only trace readback',
              'loop_wall_includes': 'prefetcher construction, all four transfer/consumer/trace loops, epoch synchronization, trace readback, memory/RNG logging; excludes fixture/model setup and final NPZ write'}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
