from __future__ import annotations

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
    TEACHERS,
    file_sha256,
    verify_h390_r3_cache_reuse,
    verify_h390_source_snapshot,
)


def _actual_trainer_parse_result(mode: str) -> dict:
    script = textwrap.dedent(
        """
        import argparse
        import ast
        import contextlib
        import importlib.util
        import io
        import json
        import sys
        from pathlib import Path

        mode, source_root = sys.argv[1], Path(sys.argv[2])
        sys.path.insert(0, str(source_root / "src"))
        spec = importlib.util.spec_from_file_location(
            "h390_r3_actual_trainer", source_root / "scripts/train_rnn.py"
        )
        trainer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = trainer
        spec.loader.exec_module(trainer)

        argv = [
            "--name", "vt_all_h390_ds_homogeneous_s3_r3", "--seed", "3",
            "--device", "cuda", "--arch", "gru", "--cell", "gru",
            "--hidden", "256", "--layers", "2", "--dropout", "0",
            "--train-source", "train+valid", "--train-seqs", "10607",
            "--valid-seqs", "0", "--valid-fold", "-1",
            "--include-heldout-valid", "--no-eval-during-fit",
            "--init", "vt_all_fin_mu256p0e4_s3@ep3", "--inherit-init-preprocessing",
            "--final-distill-cache-root", "cache/h390_all_data_homogeneous_r2",
            "--distill-alpha", "0.9", "--epochs", "6",
            "--checkpoint-policy", "fixed", "--checkpoint-epoch", "5",
            "--batch", "16", "--chunk", "2000", "--loss", "wmse",
            "--loss-norm", "chunk", "--lr", "0.0001", "--wd", "0.01",
            "--optim", "adamw", "--sched", "onecycle", "--precision", "fp16",
            "--propensity", "c20260929_prop_f1", "--prop-power", "2",
            "--prop-eps", "0.03", "--valid-true-mask", "0.03",
            "--aux-struct", "0.3", "--aux-next", "0.3", "--since-weight", "32",
            "--fast", "--save-training-state",
        ]
        if mode == "fit_groups":
            argv += ["--fit-groups-manifest", "groups.json"]

        class ParsedArgsCaptured(Exception):
            pass

        original_parse_args = argparse.ArgumentParser.parse_args
        captured = []
        def capture_parse_args(parser, args=None, namespace=None):
            parsed = original_parse_args(parser, args=args, namespace=namespace)
            captured.append((parser, parsed))
            raise ParsedArgsCaptured

        argparse.ArgumentParser.parse_args = capture_parse_args
        sys.argv = [str(source_root / "scripts/train_rnn.py"), *argv]
        try:
            trainer.main()
        except ParsedArgsCaptured:
            pass
        else:
            raise AssertionError("trainer.main did not reach the actual parser")
        parser, parsed = captured[0]
        assert isinstance(parsed, argparse.Namespace)
        assert not hasattr(parsed, "fit_subset_manifest")
        if mode == "accept":
            r2_source = source_root.parent.parent / "h390_all_data_homogeneous_r2" / "child_source_v1" / "scripts/train_rnn.py"
            tree = ast.parse(r2_source.read_text())
            stale = next(
                node for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and node.attr == "fit_subset_manifest" and node.lineno == 1705
            )
            expression = ast.fix_missing_locations(ast.Expression(stale))
            try:
                eval(compile(expression, str(r2_source), "eval"), {"args": parsed})
            except AttributeError as error:
                assert "fit_subset_manifest" in str(error)
            else:
                raise AssertionError("R2 stale parsed-Namespace access unexpectedly succeeded")
            trainer.validate_h390_final_cli_contract(parsed, parser, trainer.FINAL_DISTILL_PARENTS)
            result = {"mode": mode, "fit_groups_manifest": parsed.fit_groups_manifest, "guard": "accepted"}
        else:
            assert str(parsed.fit_groups_manifest) == "groups.json"
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    trainer.validate_h390_final_cli_contract(parsed, parser, trainer.FINAL_DISTILL_PARENTS)
            except SystemExit as error:
                assert error.code == 2
                result = {
                    "mode": mode,
                    "fit_groups_manifest": str(parsed.fit_groups_manifest),
                    "guard": "rejected",
                }
            else:
                raise AssertionError("final guard accepted a fit-groups manifest")
        print(json.dumps(result, sort_keys=True))
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SOURCE_ROOT / "src"), env.get("PYTHONPATH", "")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, mode, str(SOURCE_ROOT)],
        capture_output=True,
        check=False,
        text=True,
        timeout=90,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


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
    result = _actual_trainer_parse_result("accept")
    assert result == {"mode": "accept", "fit_groups_manifest": None, "guard": "accepted"}


def test_r3_final_guard_still_rejects_nonnull_fit_groups_manifest() -> None:
    result = _actual_trainer_parse_result("fit_groups")
    assert result == {
        "mode": "fit_groups",
        "fit_groups_manifest": "groups.json",
        "guard": "rejected",
    }


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
