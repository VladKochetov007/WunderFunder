"""Diagnostic-only solution.py: prints the scoring machine's CPU/Python identity to stderr, predicts zeros.

Never submit as a real entry; it exists to learn whether the scorer CPU has AVX-512 VNNI (q8vnni fast
path), AVX2 (q8vnni_avx2 path) or neither (portable). Output lines are prefixed "[wnn-diag]".
"""
import os
import platform
import sys

import numpy as np


def _report():
    flags, model, cores = set(), "?", 0
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags") and not flags:
                    flags = set(line.split(":", 1)[1].split())
                elif line.startswith("model name") and model == "?":
                    model = line.split(":", 1)[1].strip()
                elif line.startswith("processor"):
                    cores += 1
    except OSError as e:
        print(f"[wnn-diag] cpuinfo unreadable: {e}", file=sys.stderr)
    want = ("avx", "avx2", "fma", "f16c", "bmi1", "bmi2", "movbe", "abm", "xsave", "avx512f", "avx512bw",
            "avx512vl", "avx512dq", "avx512_vnni", "avx_vnni", "avx512_bf16", "amx_int8")
    print(f"[wnn-diag] python={sys.version.split()[0]} impl={platform.python_implementation()} "
          f"numpy={np.__version__} machine={platform.machine()} libc={platform.libc_ver()}", file=sys.stderr)
    print(f"[wnn-diag] cpu_model={model} logical_cpus={cores} affinity={len(os.sched_getaffinity(0))}",
          file=sys.stderr)
    print("[wnn-diag] isa " + " ".join(f"{k}={int(k in flags)}" for k in want), file=sys.stderr)
    print("[wnn-diag] flags " + " ".join(sorted(flags)), file=sys.stderr, flush=True)


_report()


class PredictionModel:
    def predict(self, data_point):
        if not data_point.need_prediction:
            return None
        return np.zeros(2, dtype=np.float32)
