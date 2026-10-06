"""Fixed minGRU INT8/VNNI bridge; stateful raw112 input, two FP32 outputs.

No quality/AMP/package qualification. run() carries BOTH event-feature and
recurrent state; reset() starts a new sequence. state() is diagnostic recurrent
state only, not a restorable snapshot of the event-feature history.
"""
from pathlib import Path
import ctypes as C
import hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
FORECAST = HERE.parent
NATIVE = FORECAST.parent / '2026-10-02-mingru-readiness/native-v2'
LIBRARY = NATIVE / 'generated-01/optimized.so'
ARCH_SHA = '16585c0fdcd6cf16b63234c64ea7a02d620a8f705f933fac4324c0a880a234a6'
LIB_SHA = '76dab8992cefcbd1b92e45d7a1a5bb7b36cb0b691154c52986a0206c45fe8ef7'
CPP_SHA = '36bc528281d2e767b8a0a33a3c836fbb78730e5db1211fbde69e4b18d1e0f8a9'
FP = C.POINTER(C.c_float)
FLAT_COUNT = 422 + 256*212 + 256*513 + 2*257 + 4*256*257


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def checked(x, shape, name):
    if not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.shape != shape:
        raise ValueError(f'{name}: expected float32 {shape}')
    if not np.isfinite(x).all():
        raise ValueError(f'{name}: nonfinite')
    return np.ascontiguousarray(x)


def pack_state(state, n_out=23):
    """Exact native-v2 order. O23 is trimmed ONLY at the final output matrix/bias."""
    if n_out not in (2, 23):
        raise ValueError('Only generated O2 or trained O23 layout is supported')
    shapes = {'mean': (211,), 'std': (211,), 'inp.weight': (256, 211),
              'inp.bias': (256,), 'head.0.weight': (256, 512), 'head.0.bias': (256,),
              'head.2.weight': (n_out, 256), 'head.2.bias': (n_out,)}
    for layer in range(2):
        for name in ('gate', 'candidate'):
            shapes[f'cells.{layer}.{name}.weight'] = (256, 256)
            shapes[f'cells.{layer}.{name}.bias'] = (256,)
    if set(state) != set(shapes):
        raise ValueError('Unexpected/missing state tensors')
    pieces = []
    for name, shape in shapes.items():
        value = state[name]
        if hasattr(value, 'detach'):
            if value.device.type != 'cpu':
                raise ValueError('State tensors must already be on CPU')
            value = value.detach().numpy()
        value = checked(value, shape, name)
        if name == 'std' and not (value > 0).all():
            raise ValueError('Nonpositive normalization scale')
        pieces.append((value[:2] if name.startswith('head.2.') else value).reshape(-1))
    return checked(np.concatenate(pieces), (FLAT_COUNT,), 'packed state')


def load_checkpoint(path, expected_sha256):
    """Load an explicitly hash-bound, trusted local checkpoint on CPU; no model construction."""
    import torch
    if len(expected_sha256) != 64 or sha(path) != expected_sha256:
        raise ValueError('Checkpoint identity mismatch')
    if sha(FORECAST / 'minimal_rnn.py') != ARCH_SHA:
        raise ValueError('Architecture source changed')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    args = ck['args']
    expected = dict(arch='mingru', hidden=256, layers=2, enc='linear', dropout=0.,
                    architecture_source_sha256=ARCH_SHA,
                    recurrent_state='positive FP32 tensor [layers,batch,hidden]')
    if ck['n_in'] != 211 or ck.get('readout') is not None or any(args.get(k) != v for k,v in expected.items()):
        raise ValueError('Checkpoint architecture/config mismatch')
    flags = ('raw_only', 'online_norm', 'quant_feats', 'quant_i1', 'heuristics',
             'lookahead', 'prop_input', 'class_heads', 'atom_head', 'state_reset')
    if any(args.get(k) for k in flags):
        raise ValueError('Unsupported input/head/state configuration')
    return pack_state(ck['model'], 23)


def ptr(x):
    return x.ctypes.data_as(FP)


class NativeMinimal:
    """One owned sequence state. Calls must not overlap between threads."""
    def __init__(self, flat):
        self.handle = None
        flat = checked(flat, (FLAT_COUNT,), 'flat')
        if sha(LIBRARY) != LIB_SHA or sha(NATIVE/'native.cpp') != CPP_SHA:
            raise ValueError('Fixed tested native library/source mismatch')
        lib = self.lib = C.CDLL(str(LIBRARY))
        lib.prototype_error.restype = C.c_char_p
        lib.fast_create.argtypes = [C.c_int]*4+[FP,C.c_longlong,C.c_int,C.c_int]
        lib.fast_create.restype = C.c_void_p
        lib.fast_reset.argtypes = [C.c_void_p, FP]
        lib.fast_run.argtypes = [C.c_void_p, FP, C.c_int, FP, C.c_int]
        lib.fast_state.argtypes = [C.c_void_p, FP]
        lib.fast_destroy.argtypes = [C.c_void_p]
        lib.fast_destroy.restype = None
        self.handle = lib.fast_create(211,256,2,2,ptr(flat),len(flat),1,2)
        if not self.handle:
            raise RuntimeError(lib.prototype_error().decode())

    def _ok(self, code):
        if code != 0:
            raise RuntimeError(f'native error {code}: {self.lib.prototype_error().decode()}')

    def _live(self):
        if not self.handle:
            raise RuntimeError('NativeMinimal is closed')

    def reset(self):
        self._live()
        self._ok(self.lib.fast_reset(self.handle, None))

    def state(self):
        self._live()
        out = np.empty((2,1,256), np.float32)
        self._ok(self.lib.fast_state(self.handle, ptr(out)))
        checked(out, (2,1,256), 'state')
        if (out < 0).any():
            raise RuntimeError('Negative recurrent state')
        return out

    def run(self, raw):
        self._live()
        if not isinstance(raw, np.ndarray) or raw.ndim != 2 or len(raw) > 2**31-1:
            raise ValueError('Expected [rows,112] array fitting the C ABI')
        raw = checked(raw, (len(raw),112), 'raw112')
        out = np.empty((len(raw),2), np.float32)
        if len(raw):
            self._ok(self.lib.fast_run(self.handle,ptr(raw),len(raw),ptr(out),1))
        return checked(out, (len(raw),2), 'output')

    def close(self):
        if self.handle:
            self.lib.fast_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()
