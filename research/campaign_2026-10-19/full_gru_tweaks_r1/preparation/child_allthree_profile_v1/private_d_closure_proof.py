"""Cold-import the staged profile closure from a private-D-shaped directory."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

EXPECTED = {
    "child_allthree_profile_probe_v1": "446e679ddae0f05225209950d928790264c8cd57fb8836b79f671d06d03be9e4",
    "child_allthree_profile_contract_v1": "ab68d868261106a7548b649243f4a645ece926b6138d683c731841cfb9f0b7c5",
    "propensity_terminal_support": "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409",
}


def main() -> int:
    private_dir = Path(__file__).resolve().parent
    modules = {}
    for name, expected_sha256 in EXPECTED.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.parent != private_dir or digest != expected_sha256:
            raise RuntimeError(f"private-D import origin or hash mismatch: {name}")
        modules[name] = {"path": str(path), "sha256": digest}
    if "torch" in sys.modules or "numpy" in sys.modules:
        raise RuntimeError("cold import unexpectedly loaded a model/data runtime")
    print(
        json.dumps(
            {
                "private_directory": str(private_dir),
                "modules": modules,
                "torch_imported": False,
                "numpy_imported": False,
                "payload_reader_called": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
