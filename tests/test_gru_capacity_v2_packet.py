"""Offline invariants for the frozen GRU capacity packet builder and CLI."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKET = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
    / "gru512_768_muon_capacity_v2_20261008/packet"
)
sys.path.insert(0, str(PACKET))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import capacity_worker  # noqa: E402

_REAL_PARSE_ARGS = argparse.ArgumentParser.parse_args


def _capture_parser_exit(monkeypatch: pytest.MonkeyPatch, module, argv: list[str]):
    """Run the real CLI parser, then stop before any data/model operation."""

    class Parsed(Exception):
        def __init__(self, namespace):
            self.namespace = namespace

    def parse_then_stop(parser, args=None, namespace=None):
        parsed = _REAL_PARSE_ARGS(parser, args=args, namespace=namespace)
        raise Parsed(parsed)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    monkeypatch.setattr(sys, "argv", [module.__file__, *argv])
    with pytest.raises(Parsed) as result:
        module.main()
    return result.value.namespace


@pytest.mark.parametrize(
    ("arm_id", "hidden", "optimizer"),
    [("muon512", 512, "muon"), ("muon768", 768, "muon"), ("adamw512", 512, "adamw")],
)
def test_profile_and_fit_argv_parse_with_archived_trainer(
    monkeypatch: pytest.MonkeyPatch, arm_id: str, hidden: int, optimizer: str
) -> None:
    train_rnn = importlib.import_module("train_rnn")
    arm = {
        "id": arm_id,
        "hidden": hidden,
        "optimizer": optimizer,
        "run_name": f"{arm_id}_b8",
    }
    split = Path("/tmp/frozen-fold0.json")
    profile_argv = capacity_worker._profile_argv(arm, 8, split)
    fit_argv = capacity_worker._fit_argv(arm, 8, split)

    assert "--profile-warmup" in profile_argv
    assert "--profile-steps" in profile_argv
    assert "--profile-warmup" not in fit_argv
    assert "--profile-steps" not in fit_argv
    assert fit_argv[-1] != "20"

    profile_args = _capture_parser_exit(monkeypatch, train_rnn, profile_argv)
    fit_args = _capture_parser_exit(monkeypatch, train_rnn, fit_argv)
    assert profile_args.name == arm["run_name"]
    assert profile_args.hidden == hidden
    assert profile_args.optim == optimizer
    assert profile_args.profile_warmup == 3
    assert profile_args.profile_steps == 20
    assert fit_args.name == arm["run_name"]
    assert fit_args.hidden == hidden
    assert fit_args.optim == optimizer
    assert fit_args.profile_warmup == 5
    assert fit_args.profile_steps == 0
    assert fit_args.epochs == 4
    assert fit_args.checkpoint_epoch == 3
    if optimizer == "muon":
        assert fit_args.muon_ortho_dtype_policy == "native_bf16_if_supported_else_fp32_v1"


def test_feature_only_prediction_argv_parses_with_archived_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict_run = importlib.import_module("predict_run")
    argv = capacity_worker._predict_argv("muon512_b8", Path("/tmp/frozen-fold0.json"))
    args = _capture_parser_exit(monkeypatch, predict_run, argv)
    assert args.name == "muon512_b8"
    assert args.split == "valid"
    assert args.epoch == 3
    assert args.output_name == "muon512_b8"
    assert args.campaign_split_manifest == Path("/tmp/frozen-fold0.json")
    assert args.inference_precision == "fp32"
    assert args.fit_subset_manifest is None


def test_profile_gate_requires_all_attempts_finite_and_zero_skips() -> None:
    result = {
        "status": "complete",
        "exit_code": 0,
        "markers": {
            "profile": {"step_wall": 123.0},
            "profile_update_evidence": {
                "optimizer_attempts": 230,
                "optimizer_completed_updates": 230,
                "amp_skipped_updates": 0,
                "loss_checks": 230,
                "grad_scaler_enabled": True,
                "loss_finite": True,
                "gradient_finite": True,
                "valid_for_throughput": True,
            },
        },
        "gpu": {"gpu_peak_used_mb": 1234},
    }
    assert capacity_worker._parse_profile_result(result, "muon512")["step_wall_ms"] == 123.0
    result["markers"]["profile_update_evidence"]["optimizer_completed_updates"] = 229
    with pytest.raises(ValueError, match="profile update contract"):
        capacity_worker._parse_profile_result(result, "muon512")


def test_input_link_reuses_verified_stat_without_rehashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from verified_digest_handoff import FileIdentity

    source = tmp_path / "mounted-input.bin"
    source.write_bytes(b"already verified bytes")
    identity = FileIdentity.from_stat(source.stat())
    target = tmp_path / "project" / "datasets" / source.name

    def no_rehash(_path):
        raise AssertionError("linking reread the verified source bytes")

    monkeypatch.setattr(capacity_worker, "sha256_file", no_rehash)
    capacity_worker._link_source(source, target, identity)
    assert target.is_symlink()
    assert target.resolve() == source.resolve()


def test_asset_hashing_rejects_mutation_during_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "asset.bin"
    source.write_bytes(b"original")

    def mutate_while_hashing(path: Path) -> str:
        path.write_bytes(b"changed")
        return "0" * 64

    monkeypatch.setattr(capacity_worker, "sha256_file", mutate_while_hashing)
    with pytest.raises(ValueError, match="changed while hashing"):
        capacity_worker._sha_matches(source, "0" * 64, len(b"original"))


def test_built_archive_extracts_exact_manifest_and_imports_runtime(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "extracted-project"
    observed = capacity_worker._safe_extract_source(PACKET, project_root)
    manifest = json.loads(
        (PACKET / "source/source_archive_manifest.json").read_text(encoding="utf-8")
    )
    expected = {item["path"]: item["sha256"] for item in manifest["members"]}
    assert observed == expected
    assert expected["src/wnn/numeric_ple.py"] == (
        "229a7e0f032cd65bbf6700fa9372c08016b20e4719cb2f9eddd540ff8c468720"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(project_root / "scripts"), str(project_root / "src")))
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import train_rnn, predict_run, wnn.numeric_ple, wnn.optim, wnn.training_sampling; "
                "assert train_rnn.__file__.startswith("
                + repr(str(project_root / "scripts"))
                + "); "
                "assert predict_run.__file__.startswith("
                + repr(str(project_root / "scripts"))
                + "); "
                "assert wnn.numeric_ple.__file__.startswith("
                + repr(str(project_root / "src"))
                + "); "
                "assert wnn.optim.__file__.startswith(" + repr(str(project_root / "src")) + "); "
                "print('EXTRACTED_RUNTIME_IMPORTS_PASS')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=project_root,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "EXTRACTED_RUNTIME_IMPORTS_PASS" in result.stdout

    arms = [
        {"id": "muon512", "hidden": 512, "optimizer": "muon", "run_name": "muon512_b8"},
        {"id": "muon768", "hidden": 768, "optimizer": "muon", "run_name": "muon768_b8"},
        {"id": "adamw512", "hidden": 512, "optimizer": "adamw", "run_name": "adamw512_b8"},
    ]
    tasks = []
    for arm in arms:
        tasks.append(["train", capacity_worker._profile_argv(arm, 8, Path("/tmp/fold0.json"))])
        tasks.append(["train", capacity_worker._fit_argv(arm, 8, Path("/tmp/fold0.json"))])
    tasks.append(
        [
            "predict",
            capacity_worker._predict_argv("muon512_b8", Path("/tmp/fold0.json")),
        ]
    )
    parser_script = "\n".join(
        [
            "import argparse",
            "import json",
            "import sys",
            "import train_rnn",
            "import predict_run",
            "class_source = argparse.ArgumentParser.parse_args",
            "class Stop(Exception):",
            "    pass",
            "def parse_then_stop(parser, args=None, namespace=None):",
            "    parsed = class_source(parser, args=args, namespace=namespace)",
            "    raise Stop(vars(parsed))",
            "argparse.ArgumentParser.parse_args = parse_then_stop",
            "tasks = json.loads(sys.argv[1])",
            "captured = []",
            "for kind, argv in tasks:",
            "    module = train_rnn if kind == 'train' else predict_run",
            "    sys.argv = [module.__file__, *argv]",
            "    try:",
            "        module.main()",
            "    except Stop as result:",
            "        captured.append({'kind': kind, 'args': result.args[0], 'file': module.__file__})",
            "print(json.dumps(captured, default=str))",
        ]
    )
    parsed = subprocess.run(
        [sys.executable, "-c", parser_script, json.dumps(tasks)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=project_root,
        timeout=30,
    )
    assert parsed.returncode == 0, parsed.stderr
    cli_records = json.loads(parsed.stdout)
    assert len(cli_records) == 7
    assert all(str(project_root / "scripts") in row["file"] for row in cli_records)
    assert cli_records[0]["args"]["profile_steps"] == 20
    assert cli_records[1]["args"]["profile_steps"] == 0
    assert cli_records[-1]["args"]["inference_precision"] == "fp32"


def test_built_packet_verifies_but_pending_auth_blocks_execution() -> None:
    recipe, authorization = capacity_worker._verify_packet(PACKET)
    assert recipe["packet_id"] == capacity_worker.PACKET_ID
    dependency = recipe["source"]["runtime_dependency_additions"]["src/wnn/numeric_ple.py"]
    assert dependency["sha256"] == (
        "229a7e0f032cd65bbf6700fa9372c08016b20e4719cb2f9eddd540ff8c468720"
    )
    assert dependency["source_main_commit"] == "83b97650270c8bb8fdbbe5715f659b59bd20316c"
    assert {item["path"] for item in recipe["source"]["omitted_members"]} == {
        "src/wnn/conditional_readout.py",
        "src/wnn/online_dynamics.py",
    }
    assert authorization["authorized"] is False
    with pytest.raises(PermissionError, match="not authorized"):
        capacity_worker._check_authorized_review(PACKET, authorization)
