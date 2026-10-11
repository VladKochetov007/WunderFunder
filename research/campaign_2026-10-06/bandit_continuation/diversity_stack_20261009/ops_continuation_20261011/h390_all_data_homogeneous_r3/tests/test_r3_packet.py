from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PACKET_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKET_ROOT / "child_source_v1"
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from wnn.final_distillation_cache import (  # noqa: E402
    PARENTS,
    TEACHERS,
    file_sha256,
    validate_h390_final_cli_contract,
    verify_h390_r3_cache_reuse,
    verify_h390_source_snapshot,
)

R2_SOURCE = (
    PACKET_ROOT.parent
    / "h390_all_data_homogeneous_r2"
    / "child_source_v1"
    / "scripts"
    / "train_rnn.py"
)


def _parsed_recipe(
    fit_groups_manifest: str | None = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-distill-cache-root", default=None)
    parser.add_argument("--fit-groups-manifest", default=None)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--name", default="vt_all_h390_ds_homogeneous_s3_r3")
    parser.add_argument("--init", default=f"{PARENTS[3]['run']}@ep3")
    parser.set_defaults(
        train_source="train+valid",
        include_heldout_valid=True,
        valid_fold=-1,
        train_seqs=10607,
        valid_seqs=0,
        no_eval_during_fit=True,
        valid_train_fraction=1.0,
        valid_repeat=1,
        train_classes=False,
        epoch_samples=0,
        epochs=6,
        checkpoint_policy="fixed",
        checkpoint_epoch=5,
        distill_alpha=0.9,
        distill_alpha_far=None,
        distill_boot_h=0,
        distill_reanchor=0.0,
        distill_denominator="chunk",
        distill_family_weight=0.0,
        inherit_init_preprocessing=True,
        arch="gru",
        cell="gru",
        hidden=256,
        layers=2,
        dropout=0.0,
        optim="adamw",
        sched="onecycle",
        lr=0.0001,
        wd=0.01,
        batch=16,
        chunk=2000,
        loss="wmse",
        loss_norm="chunk",
        precision="fp16",
        propensity="c20260929_prop_f1",
        prop_power=2.0,
        prop_eps=0.03,
        valid_true_mask=0.03,
        aux_struct=0.3,
        aux_next=0.3,
        since_weight=32.0,
        prop_input=None,
        fast=True,
        save_training_state=True,
        profile_steps=0,
        max_batches=0,
        stop_after_epoch=None,
    )
    argv = ["--final-distill-cache-root", "cache/h390_all_data_homogeneous_r2"]
    if fit_groups_manifest is not None:
        argv.extend(["--fit-groups-manifest", fit_groups_manifest])
    return parser, parser.parse_args(argv)


def test_r3_snapshot_binds_original_producer_and_all_receipts() -> None:
    verified = verify_h390_source_snapshot(SOURCE_ROOT)
    assert verified["cache_producer_source_manifest_sha256"] == (
        "7c68c6de67f8498b92bfc0b462d76969444f2d8fc3289008c9caf060d9371d31"
    )
    assert verified["cache_producer_packet_manifest_sha256"] == (
        "70a8e01570a2d65de9bbd5ee413b59fe52764a4cacd61c1882c3c669da82e352"
    )
    assert verified["teacher_provenance_sha256"] == {
        name: receipt["provenance_sha256"] for name, receipt in TEACHERS.items()
    }
    lineage = json.loads((PACKET_ROOT / "H390_R2_CACHE_REUSE_LINEAGE_R3.json").read_text())
    assert lineage["cache"]["train_rows"] == 10606
    assert lineage["cache"]["valid_rows"] == 1873
    assert lineage["cache"]["evaluation_row_groups"] == []
    assert lineage["cache"]["heldout_claim"] is False


def test_r2_stale_namespace_reference_fails_and_r3_final_guard_accepts() -> None:
    parser, args = _parsed_recipe()
    assert isinstance(args, argparse.Namespace)
    r2_tree = ast.parse(R2_SOURCE.read_text())
    stale = next(
        node
        for node in ast.walk(r2_tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "fit_subset_manifest"
        and node.lineno == 1705
    )
    expression = ast.fix_missing_locations(ast.Expression(stale))
    with pytest.raises(AttributeError, match="fit_subset_manifest"):
        eval(compile(expression, str(R2_SOURCE), "eval"), {"args": args})
    validate_h390_final_cli_contract(args, parser, PARENTS)


def test_r3_final_guard_still_rejects_nonnull_fit_groups_manifest() -> None:
    parser, args = _parsed_recipe("groups.json")
    with pytest.raises(SystemExit):
        validate_h390_final_cli_contract(args, parser, PARENTS)


def test_r3_cache_reuse_rejects_nonproducer_metadata(tmp_path: Path) -> None:
    cache_root = tmp_path / "h390_all_data_homogeneous_r2"
    cache_root.mkdir()
    (cache_root / "metadata.json").write_text("{}\n")
    with pytest.raises(ValueError, match="metadata/array hashes"):
        verify_h390_r3_cache_reuse(
            SOURCE_ROOT,
            cache_root,
            {
                "metadata": {},
                "metadata_sha256": file_sha256(cache_root / "metadata.json"),
                "verified_files": {},
            },
        )


def test_r3_sequences_reads_homogeneous_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    import numpy as np

    np.save(
        cache_root / "main_homogeneous_train.npy",
        np.full((1, 20_000, 2), 0.25, dtype=np.float32),
    )
    consumer = textwrap.dedent(
        """
        import importlib.util
        import sys
        from pathlib import Path
        import numpy as np
        source_root, cache_root = map(Path, sys.argv[1:])
        spec = importlib.util.spec_from_file_location(
            "h390_r3_train_rnn_consumer", source_root / "scripts/train_rnn.py"
        )
        trainer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = trainer
        spec.loader.exec_module(trainer)
        trainer.HEURISTICS = False
        class FakeSequenceFile:
            def __init__(self, _path): pass
            def read(self, _row_group):
                return type("SequenceFixture", (), {
                    "features": np.zeros((20_000, trainer.N_FEATURES), dtype=np.float32),
                    "targets": np.zeros((20_000, 2), dtype=np.float32),
                    "scored": np.zeros(20_000, dtype=np.bool_),
                })()
        trainer.SequenceFile = FakeSequenceFile
        dataset = trainer.Sequences(
            "train", [17], distill_cache_root=cache_root,
            distill_cache_variant="homogeneous",
            distill_cache_metadata={"metadata": {"files": {
                "main_homogeneous_train": {"row_groups": [17]}
            }}},
        )
        features, targets, scored = dataset[0]
        assert tuple(targets.shape) == (20_000, 4)
        assert abs(targets[:, 2:].mean().item() - 0.25) < 1e-6
        assert features.shape[0] == 20_000 and not scored.any().item()
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SOURCE_ROOT / "src"), env.get("PYTHONPATH", "")) if part
    )
    env["WNN_PROJECT_ROOT"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-c", consumer, str(SOURCE_ROOT), str(cache_root)],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
