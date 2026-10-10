from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType

PACKET = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[6]
PREDECESSOR = PACKET.parent / "kaggle_parent_v6"


def load_runner() -> ModuleType:
    runner_path = PACKET / "payload_root/scripts/kaggle_gpu_runner.py"
    sys.path.insert(0, str(runner_path.parent))
    spec = importlib.util.spec_from_file_location("h230_v8_kaggle_gpu_runner", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Check V8's deferred-input predicate against real and adversarial paths."""
    runner = load_runner()
    cache_manifest = json.loads((PREDECESSOR / "bundle/bundle-manifest.json").read_text())
    train_name = "cache/bd26_h230_fold0_teacher_family_all_rows_v1/main_source_train.npy"
    train_descriptor = next(item for item in cache_manifest["files"] if item["path"] == train_name)

    # The actual source array has already passed the frozen bootstrap's full
    # streaming hash check; this checks the exact regular-copy state seen by the
    # Kaggle runner without rereading the 1.7 GB payload.
    actual = runner.deferred_mounted_input_observation(ROOT, train_name, train_descriptor["size"])
    assert actual["matches"] is True, actual
    assert actual["observed_size"] == train_descriptor["size"]
    assert actual["is_symlink"] is False
    assert actual["resolved_inside_root"] is True

    with tempfile.TemporaryDirectory(prefix="wnn-v8-deferred-input-") as temp:
        workspace = Path(temp)
        root = workspace / "experiment"
        root.mkdir()
        (root / "cache").mkdir()
        internal = root / "cache" / "good.npy"
        internal.write_bytes(b"123456")
        assert (
            runner.deferred_mounted_input_observation(root, "cache/good.npy", 6)["matches"] is True
        )

        short = root / "cache" / "short.npy"
        short.write_bytes(b"12345")
        assert (
            runner.deferred_mounted_input_observation(root, "cache/short.npy", 6)["matches"]
            is False
        )

        outside = workspace / "outside.npy"
        outside.write_bytes(b"123456")
        external_link = root / "cache" / "external.npy"
        external_link.symlink_to(outside)
        external = runner.deferred_mounted_input_observation(root, "cache/external.npy", 6)
        assert external["matches"] is True and external["is_symlink"] is True

        escaped_directory = root / "cache" / "escaped"
        escaped_directory.symlink_to(workspace, target_is_directory=True)
        escaped = runner.deferred_mounted_input_observation(root, "cache/escaped/outside.npy", 6)
        assert escaped["matches"] is False
        assert escaped["resolved_inside_root"] is False

        missing = runner.deferred_mounted_input_observation(root, "cache/missing.npy", 6)
        assert missing["matches"] is False

    print(
        json.dumps(
            {
                "status": "pass",
                "actual_materialized_source": {
                    "path": train_name,
                    "expected_size": train_descriptor["size"],
                    "observed_size": actual["observed_size"],
                    "regular_file_inside_root": True,
                    "full_hash_source": "frozen V7 bootstrap materialize_verified_input",
                },
                "wrong_size_rejected": True,
                "external_symlink_size_checked": True,
                "escaping_regular_path_rejected": True,
                "missing_path_rejected": True,
            }
        )
    )


if __name__ == "__main__":
    main()
