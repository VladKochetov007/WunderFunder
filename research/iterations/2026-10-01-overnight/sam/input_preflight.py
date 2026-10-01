"""Hash existing read-only training inputs and check propensity-cache provenance."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    output = HERE / "input-provenance.json"
    assert not output.exists()
    files = ["datasets/train.parquet", "datasets/valid.parquet", "cache/chain_folds.npz",
             "cache/train_targets.npz", "cache/valid_targets.npz",
             "src/wnn/data.py", "src/wnn/features.py", "src/wnn/nn.py", "src/wnn/propensity.py",
             "src/wnn/resources.py", "src/wnn/splits.py"]
    for name in ["prop_f0", "prop_f1"]:
        files.append(f"artifacts/runs/{name}/model.pt")
        for split in ["train", "valid"]:
            files.extend([f"cache/propcache_{name}_{split}.npy", f"cache/propcache_{name}_{split}.json"])
    receipt = {}
    for relative in files:
        path = ROOT / relative
        before = path.stat()
        sha = digest(path)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        receipt[relative] = {"sha256": sha, "bytes": before.st_size,
                             "mtime_ns": before.st_mtime_ns, "resolved_path": str(path.resolve())}
        print(relative, before.st_size, sha, flush=True)
    for name in ["prop_f0", "prop_f1"]:
        for split in ["train", "valid"]:
            meta = json.loads((ROOT / f"cache/propcache_{name}_{split}.json").read_text())
            assert meta["model_sha256"] == receipt[f"artifacts/runs/{name}/model.pt"]["sha256"]
    output.write_text(json.dumps({"status": "PASS", "utc": datetime.now(timezone.utc).isoformat(),
                                  "scope": "Immutable input identity preflight; no forecasting fit/WP query",
                                  "script_sha256": digest(Path(__file__)), "files": receipt}, indent=2) + "\n")


if __name__ == "__main__":
    main()
