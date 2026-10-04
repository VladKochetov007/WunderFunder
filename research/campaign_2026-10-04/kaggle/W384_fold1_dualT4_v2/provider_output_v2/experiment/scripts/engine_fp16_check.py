"""Per-row profile, runtime calibration and fp16 parity for the native streaming GRU engine.

Runs in a torch-free interpreter against a wnn_native build given by --lib (a directory holding
wnn_native.abi3.so), so the shared .venv is never touched:

    PY=research/campaign_2026-10-01/engine/.venv/bin/python
    taskset -c 15 $PY scripts/engine_fp16_check.py calib submissions/X/X.zip --seqs 3
    taskset -c 15 $PY scripts/engine_fp16_check.py profile --lib DIR --seqs 1
    $PY scripts/engine_fp16_check.py parity --lib DIR --seqs 3 --out parity.json

Timings replay the scorer's per-row loop (utils.ScorerStepByStep: DataPoint construction, output
validation and copy) around the callback, so they are comparable with the scorer-harness
numbers in research/artifacts/fallback_runtime_audit.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import (  # noqa: E402
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    TARGET_COLUMNS,
    DataPoint,
    GlobalAccumulator,
)

VALID = ROOT / "datasets" / "valid.parquet"
# The incumbent public package's two GRU128 members (0.6622 public, 36m27s).
H128 = [ROOT / "submissions" / "final_since2_gclp" / "pkg" / f"weights_{i}.npz" for i in (0, 1)]


class Cycles:
    """User-space core cycles of this thread (perf_event_open), so timings survive the laptop's
    frequency swings; `None` readings when the kernel refuses the counter."""

    def __init__(self):
        import ctypes
        import struct

        self._unpack = struct.Struct("Q").unpack
        libc = ctypes.CDLL(None, use_errno=True)
        # perf_event_attr v0: PERF_TYPE_HARDWARE, CPU_CYCLES, exclude_kernel | exclude_hv.
        attr = ctypes.create_string_buffer(struct.pack("IIQQQQQIIQ", 0, 64, 0, 0, 0, 0, 0x60, 0, 0, 0))
        self.fd = libc.syscall(298, attr, 0, -1, -1, 0)

    def read(self):
        return self._unpack(os.read(self.fd, 8))[0] if self.fd >= 0 else None


CYCLES = Cycles()


def load_native(lib: Path):
    sys.path.insert(0, str(lib))
    module = importlib.import_module("wnn_native")
    if Path(module.__file__).resolve().parent != lib.resolve():
        raise SystemExit(f"imported {module.__file__}, not {lib}")
    return module


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as w:
        return {k: np.ascontiguousarray(w[k], dtype=np.float32) for k in w.files}


def synth_member(
    template: dict[str, np.ndarray], hidden: int, seed: int, cell: str = "gru"
) -> dict[str, np.ndarray]:
    """Random GRU or LSTM member of any width at the per-matrix scale of a trained GRU128.

    Matrices whose fan-in grows with the hidden size are scaled by sqrt(128 / hidden), like
    PyTorch's default init; the input projection and biases keep the trained spread.
    """
    rng = np.random.default_rng(seed)
    base = template["inp.bias"].shape[0]
    shapes = {
        "inp.weight": (hidden, template["inp.weight"].shape[1]),
        "inp.bias": (hidden,),
        "head.0.weight": (hidden, 2 * hidden),
        "head.0.bias": (hidden,),
        "head.2.weight": (2, hidden),
        "head.2.bias": (2,),
    }
    gates = {"gru": 3, "lstm": 4}[cell]
    for k in range(2):
        shapes |= {
            f"gru.weight_ih_l{k}": (gates * hidden, hidden),
            f"gru.weight_hh_l{k}": (gates * hidden, hidden),
            f"gru.bias_ih_l{k}": (gates * hidden,),
            f"gru.bias_hh_l{k}": (gates * hidden,),
        }
    out = {"mean": template["mean"], "std": template["std"]}
    for key, shape in shapes.items():
        scale = float(template[key].std())
        if key.startswith(("gru.weight", "head.")) and key.endswith("weight"):
            scale *= np.sqrt(base / hidden)
        out[key] = (rng.standard_normal(shape) * scale).astype(np.float32)
    return out


def merge_heads(trunk: dict[str, np.ndarray], heads: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """One member whose head.0 stacks k heads (width k*H) and averages them exactly, the layout
    of scripts/package_submission.py 'A|B' members. `trunk` supplies every non-head tensor."""
    k = len(heads)
    out = {key: v for key, v in trunk.items() if not key.startswith("head.")}
    out["head.0.weight"] = np.concatenate([h["head.0.weight"] for h in heads], axis=0)
    out["head.0.bias"] = np.concatenate([h["head.0.bias"] for h in heads])
    out["head.2.weight"] = np.concatenate([h["head.2.weight"][:2] for h in heads], axis=1) / k
    out["head.2.bias"] = sum(h["head.2.bias"][:2] for h in heads) / k
    return {key: np.ascontiguousarray(v, dtype=np.float32) for key, v in out.items()}


def with_head(trunk: dict[str, np.ndarray], head: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {**trunk, **{k: v for k, v in head.items() if k.startswith("head.")}}


def read_sequences(path: Path, groups: list[int]):
    pf = pq.ParquetFile(path)
    for g in groups:
        table = pf.read_row_group(g, use_threads=False)
        seq = int(table["seq_ix"][0].as_py())
        feats = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        y = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        need = table["need_prediction"].to_numpy().astype(bool)
        mask = table["is_scored"].to_numpy().astype(bool) & need
        yield seq, feats, y, need, mask


def synthetic_sequence(seed: int = 3):
    """20k rows whose 112 columns are random walks moving on ~30% of rows (no labels)."""
    rng = np.random.default_rng(seed)
    moves = (rng.random((SEQUENCE_LENGTH, 112)) < 0.3) * rng.uniform(-1, 1, (SEQUENCE_LENGTH, 112))
    feats = np.cumsum(moves, axis=0).astype(np.float32)
    need = np.arange(SEQUENCE_LENGTH) >= 99
    return 10**6 + seed, feats, None, need, None


def harness(model, seq, feats, need, cycles=None):
    """utils.ScorerStepByStep's per-row loop for one sequence; returns (seconds, predictions).
    Adds the loop's core cycles to cycles[0] when given."""
    predictions = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
    c0 = CYCLES.read()
    started = time.perf_counter()
    for step in range(SEQUENCE_LENGTH):
        point = DataPoint(seq, step, bool(need[step]), feats[step])
        value = model.predict(point)
        if not need[step]:
            if value is not None:
                raise ValueError("return None during warm-up AFTER updating state")
        else:
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError("return two finite scores on every required row")
            predictions[step] = value
    seconds = time.perf_counter() - started
    if cycles is not None and c0 is not None:
        cycles[0] += CYCLES.read() - c0
    return seconds, predictions


class Constant:
    """Python-only floor: the callback returns a constant."""

    out = np.zeros(2, dtype=np.float32)

    def predict(self, dp):
        return self.out if dp.need_prediction else None


class PyEnsemble:
    """submission/solution_ensemble.py's callback (governor bookkeeping included)."""

    def __init__(self, model):
        self.model, self.seq_ix, self.rows, self.win_step = model, None, 0, 0.0

    def predict(self, dp):
        if dp.seq_ix != self.seq_ix:
            time.perf_counter()
            self.seq_ix = dp.seq_ix
            self.model.reset()
        t0 = time.perf_counter()
        output = self.model.step(np.ascontiguousarray(dp.state, dtype=np.float32))
        self.win_step += time.perf_counter() - t0
        self.rows += 1
        if not dp.need_prediction:
            return None
        return output


class NativeCallback:
    """The whole callback in Rust: `predict` is the native bound method."""

    def __init__(self, model):
        self.predict = model.predict


def ensemble(native, members, precision, **kw):
    coefs = [np.float32(1.0 / len(members))] * len(members)
    return native.EnsemblePredictor(members, coefs, fast_activations=True, precision=precision, **kw)


def cmd_calib(args):
    """Run a packaged solution through the harness (as the scorer would)."""
    groups = list(range(args.seqs))
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(args.zip) as archive:
            archive.extractall(tmp)
        sys.path.insert(0, tmp)
        model = importlib.import_module("solution").PredictionModel()
        busy, rows, cycles, io, preds = 0.0, 0, [0], 0.0, []
        t = time.perf_counter()
        for seq, feats, _y, need, _m in read_sequences(args.data, groups):
            io += time.perf_counter() - t
            seconds, pred = harness(model, seq, feats, need, cycles)
            preds.append(pred)
            busy += seconds
            rows += SEQUENCE_LENGTH
            t = time.perf_counter()
    result = {
        "zip": str(args.zip),
        "data": str(args.data),
        "rows": rows,
        "us_per_row": busy / rows * 1e6,
        "kcycles_per_row": cycles[0] / rows / 1e3,
        "ghz": cycles[0] / busy / 1e9,
        "read_ms_per_sequence": io / (rows // SEQUENCE_LENGTH) * 1e3,
    }
    if args.save:
        np.save(args.save, np.stack(preds))
    print(json.dumps(result))


def time_callback(make, sequences, repeats):
    """Best-of-`repeats` (us/row, kcycles/row) over the sequences, fresh callback per repeat."""
    best, best_cycles = float("inf"), float("inf")
    rows = len(sequences) * SEQUENCE_LENGTH
    for _ in range(repeats):
        callback = make()
        total, cycles = 0.0, [0]
        for seq, feats, _y, need, _m in sequences:
            total += harness(callback, seq, feats, need, cycles)[0]
        best = min(best, total / rows * 1e6)
        best_cycles = min(best_cycles, cycles[0] / rows / 1e3)
    return best, best_cycles


def cmd_profile(args):
    native = load_native(args.lib)
    h128 = [load_npz(p) for p in H128]
    h128.append(synth_member(h128[0], 128, 5))
    h256 = synth_member(h128[0], 256, 7)
    tiny = synth_member(h128[0], 16, 9)
    real = list(read_sequences(VALID, list(range(args.seqs))))
    synthetic = [synthetic_sequence()]
    h128 += [synth_member(h128[0], 128, 10 + i) for i in range(3)]  # 6 GRU128 in all
    lstm128 = synth_member(h128[0], 128, 8, cell="lstm")
    # Merged multi-head members: real since trunks with their own head plus the other member's
    # head (2H) and a synthetic third head (3H); a synthetic h256 trunk with two heads (2H).
    w2 = [merge_heads(h128[i], [h128[i], h128[1 - i]]) for i in (0, 1)]
    w3 = merge_heads(h128[0], [h128[0], h128[1], synth_member(h128[0], 128, 21)])
    h256w2 = merge_heads(h256, [h256, synth_member(h128[0], 256, 22)])
    sets = {
        "h128": h128[:1],
        "h128w2": w2[:1],
        "h128w3": [w3],
        "h256w2": [h256w2],
        "A_2xh128w2": w2,
        "B_2xh128w2+h256w2": w2 + [h256w2],
        "h256": [h256],
        "lstm128": [lstm128],
        "2xh128": h128[:2],
        "2xh128+h256": h128[:2] + [h256],
        "3xh128+h256": h128[:3] + [h256],
        "6xh128": h128[:6],
        "4xh128+h256": h128[:4] + [h256],
        "3xh128+lstm128+h256": h128[:3] + [lstm128, h256],
    }
    configs = {
        "python_constant": lambda: Constant(),
        "features_tiny_h16_py_callback": lambda: PyEnsemble(ensemble(native, [tiny], "f32")),
        "features_tiny_h16_native_callback": lambda: NativeCallback(ensemble(native, [tiny], "f32")),
        "3xh128+h256_i16_py_callback": lambda: PyEnsemble(ensemble(native, sets["3xh128+h256"], "i16")),
    }
    for name, members in sets.items():
        for prec in ("f32", "f16", "i16"):
            configs[f"{name}_{prec}"] = lambda m=members, p=prec: NativeCallback(ensemble(native, m, p))
    if args.only:
        configs = {k: v for k, v in configs.items() if any(s in k for s in args.only)}
    results = {}
    for name, make in configs.items():
        real_us, real_kc = time_callback(make, real, args.repeats)
        syn_us, syn_kc = time_callback(make, synthetic, args.repeats)
        results[name] = {
            "real_us_per_row": round(real_us, 2),
            "real_kcycles_per_row": round(real_kc, 1),
            "synthetic_us_per_row": round(syn_us, 2),
            "synthetic_kcycles_per_row": round(syn_kc, 1),
        }
        print(f"{name:34s} real {real_us:8.2f} us {real_kc:7.1f} kcyc | synthetic {syn_us:8.2f} us "
              f"{syn_kc:7.1f} kcyc", flush=True)
    if args.out:
        args.out.write_text(json.dumps({"lib": str(args.lib), "seqs": args.seqs, "results": results}, indent=1))


def run_direct(model, sequences):
    """Raw per-row outputs (warm-up rows included) of an EnsemblePredictor."""
    outs = []
    for seq, feats, _y, _need, _m in sequences:
        model.reset()
        out = np.empty((SEQUENCE_LENGTH, 2), dtype=np.float32)
        for t in range(SEQUENCE_LENGTH):
            out[t] = model.step(feats[t])
        outs.append(out)
    return np.stack(outs)


def global_wp(pred, sequences):
    acc = GlobalAccumulator()
    for p, (_seq, _f, y, _need, mask) in zip(pred, sequences):
        acc.add(y, p, mask)
    return acc.result()["wp"]


def cmd_parity(args):
    native = load_native(args.lib)
    h128 = [load_npz(p) for p in H128]
    extra = [load_npz(p) for p in args.members]
    h256 = synth_member(h128[0], 256, 7)
    step = max(1, 1873 // args.seqs)
    groups = [i * step for i in range(args.seqs)]
    sequences = list(read_sequences(VALID, groups))
    cases = {
        "merged_2head_h128": [merge_heads(h128[0], [h128[0], h128[1]])],
        "h128_member0": [h128[0]],
        "h128_member1": [h128[1]],
        "h256_synthetic": [h256],
        "2xh128": h128,
        "2xh128+h256": h128 + [h256],
    }
    for i, m in enumerate(extra):
        cases[f"extra_{i}_{Path(args.members[i]).parent.parent.name}"] = [m]
    if args.only:
        cases = {k: v for k, v in cases.items() if any(o in k for o in args.only)}
    report = {"lib": str(args.lib), "row_groups": groups, "cases": {}}
    arrays = {}
    need = np.stack([s[3] for s in sequences])
    for name, members in cases.items():
        f32 = run_direct(ensemble(native, members, "f32"), sequences)
        wp32 = global_wp(f32, sequences)
        arrays[f"{name}_f32"] = f32
        report["cases"][name] = {"wp_f32": wp32, "pred_abs_mean": float(np.abs(f32[need]).mean())}
        if name.startswith("merged_"):
            # Exactness of the merge: the wide head equals the mean of the two separate members.
            separate = [h128[0], with_head(h128[0], h128[1])]
            mean = run_direct(ensemble(native, separate, "f32"), sequences)
            report["cases"][name]["f32_vs_mean_of_separate_max_abs"] = float(np.abs(f32 - mean).max())
        for prec in args.precisions:
            model = ensemble(native, members, prec)
            low = run_direct(model, sequences)
            again = run_direct(ensemble(native, members, prec), sequences)
            diff = np.abs(low - f32)
            wp = global_wp(low, sequences)
            report["cases"][name][prec] = {
                "effective": model.precision,
                "max_abs_diff_all_rows": float(diff.max()),
                "max_abs_diff_need_rows": float(diff[need].max()),
                "mean_abs_diff_need_rows": float(diff[need].mean()),
                "p999_abs_diff_need_rows": float(np.quantile(diff[need], 0.999)),
                "wp": wp,
                "delta_wp": wp - wp32,
                "repeat_bitwise_equal": bool(np.array_equal(low.view(np.uint32), again.view(np.uint32))),
            }
            arrays[f"{name}_{prec}"] = low
        print(name, json.dumps(report["cases"][name]), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
        np.savez_compressed(args.out.with_suffix(".npz"), **arrays)


def cmd_torch_parity(args):
    """Model-level parity against torch CPU outputs from engine/torch_reference.py."""
    native = load_native(args.lib)
    report = {"lib": str(args.lib), "cases": {}}
    for path in args.refs:
        with np.load(path) as ref:
            x, y64, y32 = ref["X"], ref["y64"], ref["y32"]
            weights = {k[2:]: np.ascontiguousarray(ref[k]) for k in ref.files if k.startswith("w/")}
        case = {"rows": int(len(x)), "y_abs_mean": float(np.abs(y64).mean()),
                "torch_f32_vs_f64_max": float(np.abs(y32 - y64).max())}
        for prec in ("f32", "f16", "i16"):
            for fast in (False, True):
                model = native.ModelStepper(weights, fast_activations=fast, precision=prec)
                y = model.run(x)
                model.reset()
                again = model.run(x)
                d = np.abs(y - y64)
                case[f"{prec}_{'fast' if fast else 'exact'}"] = {
                    "cell": model.cell,
                    "effective": model.precision,
                    "max_abs_vs_torch64": float(d.max()),
                    "mean_abs_vs_torch64": float(d.mean()),
                    "reset_bitwise_repeat": bool(np.array_equal(y.view(np.uint32), again.view(np.uint32))),
                }
        report["cases"][Path(path).stem] = case
        print(Path(path).stem, json.dumps(case), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calib")
    p.add_argument("zip", type=Path)
    p.add_argument("--data", type=Path, default=ROOT / "cache" / "valid_subset20.parquet")
    p.add_argument("--seqs", type=int, default=3)
    p.add_argument("--save", type=Path, help="write the (seqs, 20000, 2) predictions here")
    p = sub.add_parser("profile")
    p.add_argument("--lib", type=Path, required=True)
    p.add_argument("--seqs", type=int, default=1)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--only", nargs="*")
    p.add_argument("--out", type=Path)
    p = sub.add_parser("parity")
    p.add_argument("--lib", type=Path, required=True)
    p.add_argument("--seqs", type=int, default=3)
    p.add_argument("--members", nargs="*", default=[], type=Path)
    p.add_argument("--precisions", nargs="+", default=["f16", "i16"])
    p.add_argument("--only", nargs="*", help="run only cases whose name contains one of these")
    p.add_argument("--out", type=Path)
    p = sub.add_parser("torch-parity")
    p.add_argument("--lib", type=Path, required=True)
    p.add_argument("refs", nargs="+", type=Path)
    p.add_argument("--out", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    {"calib": cmd_calib, "profile": cmd_profile, "parity": cmd_parity, "torch-parity": cmd_torch_parity}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
