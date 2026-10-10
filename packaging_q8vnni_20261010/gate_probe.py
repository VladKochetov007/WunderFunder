"""Offline Python 3.11 callback gate for the cpu_probe ZIP (zero predictions; a callback replay of 2000 rows)."""
import importlib.util
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace")
from utils import DataPoint  # noqa: E402

assert sys.version_info[:2] == (3, 11)
with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(sys.argv[1]) as z:
    assert z.testzip() is None
    z.extractall(tmp)
    spec = importlib.util.spec_from_file_location("sol", Path(tmp) / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.PredictionModel()
    row = np.zeros(112, np.float32)
    t0 = time.perf_counter()
    for i in range(2000):
        v = m.predict(DataPoint(0, i, i >= 99, row))
        assert (v is None) == (i < 99)
        if v is not None:
            assert v.shape == (2,) and v.dtype == np.float32 and np.isfinite(v).all()
    print(json.dumps({"passed": True, "python": sys.version.split()[0], "us_per_row": (time.perf_counter() - t0) / 2000 * 1e6,
                      "files": z.namelist()}))
