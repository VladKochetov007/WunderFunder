from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PACKET = Path(__file__).resolve().parent
SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load actual clean-F0 parent producer checkpoints through the frozen trainer consumer.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    os.environ["WNN_PROJECT_ROOT"] = str(root)
    source = PACKET / "source_snapshot"
    sys.path.insert(0, str(source / "src"))
    module_spec = importlib.util.spec_from_file_location("cleanfold_t3_actual_consumer", source / "scripts/train_rnn.py")
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot import frozen train_rnn consumer")
    trainer = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = trainer
    module_spec.loader.exec_module(trainer)
    from wnn.splits import load_campaign_split  # noqa: PLC0415

    split = load_campaign_split(PACKET / "reference_inputs/fold0.json", root)
    if split.sha256 != SPLIT_SHA:
        raise ValueError("actual source-qualified split differs from frozen F0 manifest")
    parents = [
        {
            "arm": "control",
            "run": "c20261003_combo_s3_f0",
            "path": root / "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt",
            "provenance": root / "artifacts/runs/c20261003_combo_s3_f0/provenance.json",
            "checkpoint_sha256": "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e",
            "provenance_sha256": "9a9d17eeb4b844d85bb5ca132e0085d0cf6dd1387d3d3de4c56d8e3bb8be7a6f",
            "prop_power": 2.0,
        },
        {
            "arm": "treatment",
            "run": "kg_h382r3_f0_muonall_prop0_e4_s3",
            "path": root / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/kaggle_h382_delivery_20261011_r4/provider_output/out/runs/kg_h382r3_f0_muonall_prop0_e4_s3/model_ep3.pt",
            "provenance": root / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/kaggle_h382_delivery_20261011_r4/provider_output/out/runs/kg_h382r3_f0_muonall_prop0_e4_s3/provenance.json",
            "checkpoint_sha256": "359d337db09f0f8e306a225b050884741f006190523b2e940d36ef87d35a25be",
            "provenance_sha256": "b220b6e822c9e4779d508655c56ecacb4c42399a001a817d30de8fd3bd2610b7",
            "prop_power": 0.0,
        },
    ]
    records: list[dict[str, Any]] = []
    for parent in parents:
        import torch  # noqa: PLC0415

        if sha256(parent["path"]) != parent["checkpoint_sha256"] or sha256(parent["provenance"]) != parent["provenance_sha256"]:
            raise ValueError(f"{parent['arm']} parent bytes differ from pinned producer receipt")
        checkpoint = torch.load(parent["path"], map_location="cpu", weights_only=False)
        provenance = json.loads(parent["provenance"].read_text())
        recipe = checkpoint["args"]
        expected = {
            "name": parent["run"],
            "seed": 3,
            "arch": "gru",
            "cell": "gru",
            "hidden": 256,
            "layers": 2,
            "epochs": 4,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 3,
            "no_eval_during_fit": True,
            "split_manifest_sha256": SPLIT_SHA,
            "propensity": "c20260929_prop_f0",
            "prop_power": parent["prop_power"],
            "prop_eps": 0.03,
        }
        if any(recipe.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{parent['arm']} actual producer recipe differs from its declared parent contract")
        if checkpoint.get("epoch") != 3 or checkpoint.get("n_in") != 211:
            raise ValueError(f"{parent['arm']} parent epoch/input width mismatch")
        fit = provenance.get("fit_row_groups", {})
        actual_train = set(map(int, fit.get("train", [])))
        actual_valid = set(map(int, fit.get("valid", [])))
        actual_eval = set(map(int, provenance.get("evaluation_row_groups", [])))
        if actual_train != set(split.train_groups) or actual_valid != set(split.valid_groups):
            raise ValueError(f"{parent['arm']} producer fitting membership differs from exact fold-0 fit groups")
        # Row-group integers are namespaced by source. Only valid-fit groups
        # can overlap valid evaluation groups; train-source IDs are disjoint
        # by construction even when their integer values happen to match.
        if actual_eval != set(split.evaluation_groups) or actual_valid & actual_eval:
            raise ValueError(f"{parent['arm']} producer ancestry includes held-out fold-0 evaluation groups")
        if provenance.get("checkpoint_sha256", {}).get("model_ep3.pt") != parent["checkpoint_sha256"]:
            raise ValueError(f"{parent['arm']} provenance checkpoint digest differs from actual bytes")
        trainer.validate_campaign_checkpoint(
            checkpoint,
            SimpleNamespace(split_manifest=True, split_manifest_sha256=SPLIT_SHA),
        )
        # Recreate the actual H390 student readout width (main targets plus
        # spread/next auxiliaries), then apply train_rnn's parent-load rule.
        student_args = {
            "arch": "gru", "cell": "gru", "hidden": 256, "layers": 2,
            "dropout": 0.0, "readout_activation": "gelu", "head": "mlp",
        }
        student_n_out = 2 + trainer.N_SP_AUX + 3 + 4 + 2
        model = trainer.make_model(student_args, checkpoint["n_in"], student_n_out)
        init_state = dict(checkpoint["model"])
        for key in ("head.2.weight", "head.2.bias"):
            if init_state[key].shape[0] > model.state_dict()[key].shape[0]:
                init_state[key] = init_state[key][: model.state_dict()[key].shape[0]].clone()
        model.load_state_dict(init_state)
        records.append({
            "arm": parent["arm"],
            "run": parent["run"],
            "checkpoint_path": str(parent["path"]),
            "checkpoint_sha256": parent["checkpoint_sha256"],
            "checkpoint_bytes": parent["path"].stat().st_size,
            "provenance_sha256": parent["provenance_sha256"],
            "split_manifest_sha256": SPLIT_SHA,
            "train_fit_groups": len(actual_train),
            "valid_fit_groups": len(actual_valid),
            "evaluation_groups_excluded": len(actual_eval),
            "source_qualified_evaluation_exclusion": "valid-fit row groups are disjoint from valid evaluation groups; train-source row-group integers are a separate namespace",
            "architecture": f"GRU256x2; n_in={checkpoint['n_in']}; student auxiliary output width={student_n_out}",
            "consumer_checks": ["frozen train_rnn.validate_campaign_checkpoint", "frozen train_rnn.make_model with H390 auxiliary output shape", "frozen parent head adaptation and model.load_state_dict"],
            "status": "actual_producer_fixture_consumed_pass",
        })
        del model, checkpoint
    receipt = {
        "schema": "wnn.cleanfold_t3_f0_actual_producer_preflight.r1",
        "status": "pass_static_actual_producer_fixtures_only",
        "scope": "CPU checkpoint deserialization, exact fold/source-provenance gates, and real trainer model consumer load; no forward, fit, prediction, provider or queue action",
        "source_snapshot_manifest_sha256": sha256(source / "SOURCE_MANIFEST_H390.json"),
        "train_rnn_consumer_sha256": sha256(source / "scripts/train_rnn.py"),
        "distillation_cache_consumer_sha256": sha256(source / "src/wnn/distillation_cache.py"),
        "split_manifest_sha256": SPLIT_SHA,
        "parents": records,
        "remaining_consumer_gates": ["profile actual T4 memory/time for H384 teachers and H256 resume boundaries", "validate producer-generated teacher prediction sidecars and arrays with cache builder", "validate new six-epoch training_state.pt outputs with exact resume consumer after each real producer stage", "grade final evaluation-only predictions against frozen fold0 evaluation membership"],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
