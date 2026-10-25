"""Run scripts/train_rnn.py unchanged with stack dumps and a shared portable-digest cache (Kaggle).

faulthandler writes every thread's stack to stderr every 240 s, so a silent stall becomes visible
in the worker log. With WNN_SHARED_DIGEST_CACHE set, wnn.portability's full-content SHA-256 of each
relocated source is computed once per kernel under a file lock and reused by concurrent workers,
instead of every worker re-hashing ~35 GB of parquet through the input mount. The digest is still
compared with the sealed registry by verify_portable_source. Arguments pass through to train_rnn.
"""

from __future__ import annotations

import faulthandler
import fcntl
import json
import os
import runpy
import sys
import time
from pathlib import Path

faulthandler.enable()
faulthandler.dump_traceback_later(240, repeat=True)
print(f"TRACED_WORKER_START {time.time():.0f}", flush=True)
script = Path(__file__).resolve().parent / "train_rnn.py"
sys.path.insert(0, str(script.parents[1] / "src"))


def install_shared_digest_cache(cache_path: Path) -> None:
    """Wrap wnn.portability._content_digest with a cross-process, lock-protected cache."""
    from wnn import portability

    original = portability._content_digest

    def cached(
        path: str, device: int, inode: int, size: int, modified_ns: int, changed_ns: int
    ) -> str:
        key = f"{path}|{size}|{modified_ns}"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            table = json.loads(cache_path.read_text()) if cache_path.exists() else {}
            if key in table:
                print(f"DIGEST_CACHE_HIT {path}", flush=True)
                return table[key]
            started = time.time()
            digest = original(path, device, inode, size, modified_ns, changed_ns)
            table[key] = digest
            cache_path.write_text(json.dumps(table, indent=1))
            print(f"DIGEST_COMPUTED {path} {size} B in {time.time() - started:.1f} s", flush=True)
            return digest

    portability._content_digest = cached


if os.environ.get("WNN_SHARED_DIGEST_CACHE"):
    install_shared_digest_cache(Path(os.environ["WNN_SHARED_DIGEST_CACHE"]))
sys.argv = [str(script), *sys.argv[1:]]
runpy.run_path(str(script), run_name="__main__")
