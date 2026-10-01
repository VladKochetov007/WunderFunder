"""CTXMEM01 causal event-clock recurrence; no dataset or model I/O."""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.signal import lfilter

SCHEMA = 'ctxmem01-hadamard16-ema4-32-v1'
RANK, WIDTH, HIDDEN = 16, 32, 128
LAMBDA_FAST = float.fromhex('0x1.ae89f995ad3adp-1')
LAMBDA_SLOW = float.fromhex('0x1.f50765b6e4540p-1')
HADAMARD_ROWS = (68, 93, 98, 43, 72, 42, 49, 44, 7, 104, 76, 95, 26, 64, 41, 65)
PROJECTION_SHA256 = '0ed6109334d68ec914d5c5e3da6b7f0ed898db05828ace54c76532a4c443f20b'


def projection_from_rows(rows):
    rows = np.asarray(rows, dtype=np.int64)
    if rows.shape != (RANK,) or len(set(rows.tolist())) != RANK or rows.min() < 0 or rows.max() >= HIDDEN:
        raise ValueError('Expected 16 unique fixed Hadamard rows in [0,128)')
    hadamard = np.ones((1, 1), dtype=np.int8)
    while len(hadamard) < HIDDEN:
        hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])
    return (hadamard[rows].astype(np.float64) / np.sqrt(128.0)).astype(np.float32)


def projection_sha(matrix):
    return hashlib.sha256(np.ascontiguousarray(matrix, dtype='<f4').tobytes()).hexdigest()


def fixed_projection():
    matrix = projection_from_rows(HADAMARD_ROWS)
    assert projection_sha(matrix) == PROJECTION_SHA256
    return matrix


def metadata(arm):
    if arm not in ('memory', 'current'):
        raise ValueError('Unknown arm')
    return {'schema': SCHEMA, 'arm': arm, 'rank': RANK, 'feature_width': WIDTH,
            'projection_rows': list(HADAMARD_ROWS), 'projection_sha256': PROJECTION_SHA256,
            'projection_matrix': fixed_projection().tolist(),
            'lambda_fast_hex': LAMBDA_FAST.hex(), 'lambda_slow_hex': LAMBDA_SLOW.hex(),
            'memory_dtype': 'float64', 'head_feature_dtype': 'float32',
            'projection_dtype': 'float32', 'projection_tf32': False,
            'first_row': 'initialize fast=slow=q0', 'event_column': 178,
            'event_rule': 'raw0!=previous_raw0 AND raw11!=previous_raw11',
            'control': 'both histories computed; head uses duplicate current q',
            'reset': 'new sequence only; preserve across arbitrary chunk boundaries'}


def empty_state(batch):
    return (np.zeros((batch, RANK), np.float64), np.zeros((batch, RANK), np.float64),
            np.zeros(batch, dtype=bool))


def event_histories(q, event, state=None):
    """q is float32 BxTx16. Carries actual m, never lfilter delay or m/lambda.

    Event flags already belong to the causal full-sequence input construction.
    First row of a new sequence initializes both memories, independently of flag.
    Every call computes both histories, including the current-q control arm.
    """
    q = np.asarray(q, dtype=np.float32)
    event = np.asarray(event, dtype=bool)
    if q.ndim != 3 or q.shape[2] != RANK or event.shape != q.shape[:2]:
        raise ValueError('Expected q[B,T,16], event[B,T]')
    if not np.isfinite(q).all():
        raise ValueError('Nonfinite contextual projection')
    batch, steps, _ = q.shape
    old = empty_state(batch) if state is None else state
    fast, slow, initialized = (np.array(value, copy=True) for value in old)
    if fast.shape != (batch, RANK) or slow.shape != (batch, RANK) or initialized.shape != (batch,):
        raise ValueError('Memory state batch mismatch')
    if fast.dtype != np.float64 or slow.dtype != np.float64 or initialized.dtype != np.bool_:
        raise ValueError('Memory must carry float64 states and bool initialization flags')
    result = np.empty((batch, steps, WIDTH), np.float32)
    if not steps:
        return result, (fast, slow, initialized)
    row_positions = np.arange(steps)
    for b in range(batch):
        if not initialized[b]:
            fast[b] = slow[b] = q[b, 0].astype(np.float64)
            initialized[b] = True
            indices = np.flatnonzero(event[b, 1:]) + 1
        else:
            indices = np.flatnonzero(event[b])
        gather = np.searchsorted(indices, row_positions, side='right')
        selected = q[b, indices].astype(np.float64)
        for offset, decay, memory in [(0, LAMBDA_FAST, fast), (RANK, LAMBDA_SLOW, slow)]:
            values = np.empty((len(indices) + 1, RANK), np.float64)
            values[0] = memory[b]
            if len(indices):
                # DF-II-transposed delay is lambda*m_previous. Carry the final
                # output, not zf (which equals lambda*m_final) and no division.
                filtered, _ = lfilter([1.0 - decay], [1.0, -decay], selected, axis=0,
                                      zi=(decay * memory[b])[None, :])
                values[1:] = filtered
                memory[b] = filtered[-1]
            result[b, :, offset:offset + RANK] = values[gather].astype(np.float32)
    return result, (fast, slow, initialized)


def selected_features(q, histories, arm):
    if arm == 'memory':
        return histories
    if arm == 'current':
        return np.concatenate((q, q), axis=-1)
    raise ValueError('CTXMEM arm must be memory or current')
