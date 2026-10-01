"""SELRISK01 fixed sampling, role separation and immutable-input helpers."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PANELS = ("F", "A", "B")
SELECTORS = ("M", "X", "XY", "SHUFFLED")
BINS = np.linspace(99, 20000, 257, dtype=np.int64)
STUDENT_CPUS = {22, 23, 24, 25, 28, 32}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def key(*parts):
    return hashlib.sha256("|".join(map(str, ("SELRISK01", *parts))).encode()).hexdigest()


def repo(relative):
    p = Path(relative)
    assert not p.is_absolute() and ".." not in p.parts and str(p) == relative
    return ROOT / p  # Registered read-only cache/dataset symlinks are intentional.


def stat_identity(path):
    s = Path(path).stat()
    return {k: getattr(s, k) for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns")}


def verify_record(path, record):
    assert isinstance(record["sha256"], str) and len(record["sha256"]) == 64
    assert Path(path).stat().st_size == record["bytes"]
    if record.get("verification", "sha256") == "stat_guard":
        assert record["full_sha256_previously_verified"] is True
        assert stat_identity(path) == record["stat"], f"Stale full-hash identity: {path}"
    else:
        assert sha(path) == record["sha256"], f"Hash mismatch: {path}"


def load_allocation(path, expected, stage):
    assert sha(path) == expected
    allocation = json.loads(Path(path).read_text())
    assert allocation["status"] == "ROOT_ALLOCATED_SELECTION_RISK"
    bp = repo(allocation["bindings"])
    assert sha(bp) == allocation["bindings_sha256"]
    bindings = json.loads(bp.read_text())
    assert bindings["status"] == "FROZEN_SELECTION_RISK_BINDINGS"
    mandatory_inputs = {"datasets/valid.parquet", "cache/chain_folds.npz",
                        "research/artifacts/market-classes-v1.npz",
                        *(f"artifacts/runs/on_ctl128s3_f{f}/model_ep1.pt" for f in (0, 1))}
    assert set(bindings["inputs"]) == mandatory_inputs
    mandatory_sources = {str((HERE / name).relative_to(ROOT)) for name in
                         ("common.py", "gradient_core.py", "extract_selected.py", "fit_selectors.py", "spec.md")}
    mandatory_sources |= {"scripts/train_rnn.py", "src/wnn/nn.py", "src/wnn/data.py", "src/wnn/features.py"}
    assert mandatory_sources <= set(bindings["sources"])
    cpus = set(allocation[stage + "_cpus"])
    assert 1 <= len(cpus) <= 2 and not cpus & STUDENT_CPUS
    assert set(os.sched_getaffinity(0)) <= cpus
    for name, expected_sha in bindings["sources"].items():
        assert sha(repo(name)) == expected_sha, f"Source changed: {name}"
    for package, version in bindings["versions"].items():
        assert importlib.metadata.version(package) == version
    output = repo(allocation["output"]).resolve()
    assert output.is_relative_to(HERE.resolve()) and output != HERE.resolve()
    return allocation, bindings, output


def verify_inputs(bindings):
    for name, record in bindings["inputs"].items():
        verify_record(repo(name), record)
    for name, expected in bindings["sources"].items():
        assert sha(repo(name)) == expected


def panel_positions(fold, chain, row_group):
    result = {}
    for panel in PANELS:
        seed = int(key("rows", fold, chain, row_group, panel)[:32], 16)
        rng = np.random.default_rng(seed)
        result[panel] = {"seed": str(seed), "positions": [
            int(rng.integers(lo, hi)) for lo, hi in zip(BINS[:-1], BINS[1:])]}
    return result


def select_records(cf, classes):
    """Metadata only. Reject impure/mixed-class or cross-fold chains before sampling."""
    train = set(cf["chain"][cf["split"] == "train"].tolist())
    valid = cf["split"] == "valid"
    inventory = {(f, c): [] for f in (0, 1) for c in range(4)}
    for chain in np.unique(cf["chain"][valid]):
        at = valid & (cf["chain"] == chain)
        folds = np.unique(cf["fold"][at])
        assert len(folds) == 1
        fold = int(folds[0])
        if fold not in (0, 1) or not np.all(cf["pure"][at]):
            continue
        assert int(chain) not in train
        row_groups = cf["row_group"][at].astype(int).tolist()
        cl = np.unique(classes[row_groups])
        assert len(cl) == 1 and int(cl[0]) in range(4), "Mixed/unknown class in pure chain"
        c = int(cl[0])
        rg = min(row_groups, key=lambda g: key("sequence", fold, c, int(chain), g))
        inventory[(fold, c)].append((int(chain), rg))
    records = []
    for (fold, c), choices in inventory.items():
        assert len(choices) >= 16, f"Insufficient chain quota: fold{fold}/class{c}"
        chosen = sorted(choices, key=lambda x: key(fold, c, x[0]))[:16]
        for rank, (chain, rg) in enumerate(chosen):
            records.append({"fold": fold, "class": c, "chain": chain, "row_group": rg,
                            "group": rank % 4, "panels": panel_positions(fold, chain, rg)})
    assert len(records) == 128 and len({r["chain"] for r in records}) == 128
    return records


def role_indices(records, fold, audit_group):
    ids = [i for i, r in enumerate(records) if r["fold"] == fold]
    roles = {
        "audit": [i for i in ids if records[i]["group"] == audit_group],
        "calibration": [i for i in ids if records[i]["group"] == (audit_group + 1) % 4],
        "fit": [i for i in ids if records[i]["group"] not in (audit_group, (audit_group + 1) % 4)],
    }
    assert [len(roles[k]) for k in ("fit", "calibration", "audit")] == [32, 16, 16]
    assert not set(roles["fit"]) & set(roles["calibration"])
    assert not (set(roles["fit"]) | set(roles["calibration"])) & set(roles["audit"])
    return {k: np.asarray(v, dtype=np.int64) for k, v in roles.items()}


def derangement(records, indices, fold, rotation, role):
    """Donor whole sampled target streams stay inside the same role/class."""
    mapping = {}
    for c in range(4):
        ids = [int(i) for i in indices if records[int(i)]["class"] == c]
        ids.sort(key=lambda i: key("donor", fold, rotation, role, records[i]["chain"]))
        assert len(ids) >= 2
        mapping.update({i: ids[(j + 1) % len(ids)] for j, i in enumerate(ids)})
    assert set(mapping) == set(map(int, indices)) == set(mapping.values())
    assert all(i != j and records[i]["class"] == records[j]["class"] for i, j in mapping.items())
    return np.asarray([mapping[int(i)] for i in indices], dtype=np.int64)


def outcome_features(target):
    c = np.clip(np.asarray(target), -2, 2)
    return np.stack([c[..., 0], c[..., 1], abs(c[..., 0]), abs(c[..., 1]),
                     c[..., 0] * c[..., 1], abs(c[..., 0] - c[..., 1])], axis=-1)
