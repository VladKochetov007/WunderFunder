from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from train_rnn import checkpoint_selected, configure_checkpoint_policy

from wnn.splits import load_campaign_split


def main() -> None:
    """Independently reconstruct graph reachability and check frozen fitting pools."""
    started = time.perf_counter()
    checks: list[dict] = []
    with np.load(ROOT / "cache/chains.npz", allow_pickle=False) as data:
        successor = data["next"]
        keys = list(zip(data["split"].tolist(), data["row_group"].tolist(), strict=True))
    adjacency = [[] for _ in keys]
    for index, following in enumerate(successor):
        if following >= 0:
            adjacency[index].append(int(following))
            adjacency[int(following)].append(index)
    labels = np.full(len(keys), -1, dtype=np.int64)
    for start in range(len(keys)):
        if labels[start] >= 0:
            continue
        pending = [start]
        labels[start] = start
        while pending:
            node = pending.pop()
            for neighbor in adjacency[node]:
                if labels[neighbor] < 0:
                    labels[neighbor] = start
                    pending.append(neighbor)
    label_of = dict(zip(keys, labels.tolist(), strict=True))
    seen_evaluation: set[int] = set()
    for fold in range(4):
        path = ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
        split = load_campaign_split(path, ROOT)
        evaluated = {label_of[("valid", group)] for group in split.evaluation_groups}
        expected_train = {
            group
            for (name, group), label in label_of.items()
            if name == "train" and label not in evaluated
        }
        expected_valid = {
            group
            for (name, group), label in label_of.items()
            if name == "valid" and label not in evaluated
        }
        assert set(split.train_groups) == expected_train
        assert set(split.valid_groups) == expected_valid
        assert not seen_evaluation.intersection(split.evaluation_groups)
        seen_evaluation.update(split.evaluation_groups)
        split.assert_fit_groups(split.train_groups, split.valid_groups)
        checks.append(
            {
                "fold": fold,
                "sha256": split.sha256,
                "train": len(split.train_groups),
                "valid_fit": len(split.valid_groups),
                "evaluation": len(split.evaluation_groups),
                "status": "pass",
            }
        )
    assert seen_evaluation == {group for name, group in keys if name == "valid"}
    args = argparse.Namespace(
        epochs=4,
        checkpoint_policy="fixed",
        checkpoint_epoch=3,
        no_eval_during_fit=True,
        split_manifest="frozen.json",
    )
    configure_checkpoint_policy(args, argparse.ArgumentParser())
    for epoch in range(4):
        for observed in (None, -1.0, 0.999):
            assert checkpoint_selected(epoch, "fixed", 3, observed, -1.0) == (epoch == 3)
    hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "src/wnn/splits.py",
            "scripts/build_campaign_splits.py",
            "scripts/train_propensity.py",
            "scripts/train_rnn.py",
        )
    }
    report = {
        "status": "pass",
        "checks": checks,
        "evaluation_partition_rows": len(seen_evaluation),
        "independent_method": "undirected adjacency traversal, not production union-find",
        "fixed_policy_observed_score_independent": True,
        "hashes": hashes,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(__file__).with_name("independent_controls_check.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
