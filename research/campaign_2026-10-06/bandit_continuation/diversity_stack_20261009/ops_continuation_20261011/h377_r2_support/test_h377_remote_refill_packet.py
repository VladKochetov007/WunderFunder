"""Bounded local checks for the H377 R2 profile-only refill packet."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SUPPORT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[6]
PARENT_DAEMON = (
    PROJECT / "research/campaign_2026-10-06/bandit_continuation/vast_teachers_r1/remote_refill.py"
)
PACKET = SUPPORT / "p04_h377_b64_profile_s3_r2.json"
PREFLIGHT = SUPPORT / "source_preflight.py"
VALIDATOR = SUPPORT / "profile_validator.py"
PACKET_MANIFEST = SUPPORT / "H377_PROFILE_PACKET_MANIFEST_R2.json"


def _run_helper(
    script: Path, args: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _daemon_script_expression() -> ast.expr:
    tree = ast.parse(PARENT_DAEMON.read_text())
    launch = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "launch"
    )
    assignment = next(
        node
        for node in launch.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "script" for target in node.targets)
    )
    return assignment.value


def _render_like_daemon(job: dict[str, object], log_dir: Path) -> str:
    expression = _daemon_script_expression()
    rendered = eval(compile(ast.Expression(expression), str(PARENT_DAEMON), "eval"), {"job": job})
    log_dir.mkdir(parents=True, exist_ok=True)
    return rendered.replace("/root/run/logs/", f"{log_dir}/")


def test_source_preflight_checks_manifest_inventory_aliases_and_inputs(tmp_path: Path) -> None:
    project = tmp_path / "wnn"
    snapshot = project / "lr377_source_r1"
    (snapshot / "scripts").mkdir(parents=True)
    (snapshot / "src/wnn").mkdir(parents=True)
    for directory in ("artifacts", "cache", "datasets", "research"):
        (project / directory).mkdir()
        (snapshot / directory).symlink_to(project / directory, target_is_directory=True)

    source_files = {
        "scripts/train_rnn.py": "print('trainer fixture')\n",
        "src/wnn/optim.py": "VALUE = 1\n",
    }
    source_map = {}
    for relative, content in source_files.items():
        path = snapshot / relative
        path.write_text(content)
        source_map[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_manifest = {
        "schema": "wnn.frozen-source-snapshot.v1",
        "remote_root": str(snapshot),
        "sources": source_map,
    }
    source_manifest_path = snapshot / "SOURCE_MANIFEST.json"
    source_manifest_path.write_text(json.dumps(source_manifest, sort_keys=True))
    registry = project / "research/portable_registry.json"
    registry.write_text(json.dumps({"sources": {"fixture": {"sha256": "abc"}}}))
    split = project / "research/fold1.json"
    split.write_text(json.dumps({"fold": 1}))

    expected = {
        "source_manifest_sha256": hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
        "portable_manifest_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
    }
    env = os.environ | {
        "WNN_PROJECT_ROOT": str(project),
        "PYTHONPATH": str(snapshot / "src"),
        "WNN_PORTABLE_DATA_MANIFEST": str(registry),
    }
    args = [
        "--project-root",
        str(project),
        "--snapshot-root",
        str(snapshot),
        "--source-manifest-sha256",
        expected["source_manifest_sha256"],
        "--portable-manifest",
        str(registry),
        "--portable-manifest-sha256",
        expected["portable_manifest_sha256"],
        "--split-manifest",
        str(split),
        "--split-manifest-sha256",
        expected["split_manifest_sha256"],
    ]
    valid = _run_helper(PREFLIGHT, args, env)
    assert valid.returncode == 0, valid.stderr
    assert "H377_SOURCE_INPUT_PREFLIGHT_OK" in valid.stdout

    (snapshot / "src/wnn/optim.py").write_text("VALUE = 2\n")
    changed = _run_helper(PREFLIGHT, args, env)
    assert changed.returncode == 1
    assert "source src/wnn/optim.py sha256 differs" in changed.stderr


def test_profile_validator_accepts_profile_only_output_and_rejects_checkpoint(
    tmp_path: Path,
) -> None:
    name = "fixture_profile"
    project = tmp_path / "wnn"
    run_dir = project / "artifacts/runs" / name
    run_dir.mkdir(parents=True)
    preflight_log = tmp_path / "preflight.log"
    preflight_log.write_text("H377_SOURCE_INPUT_PREFLIGHT_OK {}\n")
    profile_log = tmp_path / "profile.log"
    profile = {"gpu_total": 1.2, "step_wall": 0.4, "seq_per_s": 42.0, "loader_wait": 0.01}
    profile_log.write_text(
        json.dumps({"name": name, "profile_steps": 3, "profile_warmup": 5})
        + "\nPROFILE "
        + json.dumps(profile)
        + "\n"
    )
    args = [
        "--preflight-log",
        str(preflight_log),
        "--profile-log",
        str(profile_log),
        "--run-dir",
        str(run_dir),
        "--project-root",
        str(project),
        "--run-name",
        name,
        "--profile-steps",
        "3",
        "--profile-warmup",
        "5",
    ]
    valid = _run_helper(VALIDATOR, args, os.environ.copy())
    assert valid.returncode == 0, valid.stderr
    assert "H377_PROFILE_VALIDATION_OK" in valid.stdout

    (run_dir / "training_state.pt").write_bytes(b"unexpected")
    checkpoint = _run_helper(VALIDATOR, args, os.environ.copy())
    assert checkpoint.returncode == 1
    assert "unexpectedly produced fit artifacts" in checkpoint.stderr


def test_r2_packet_is_profile_only_and_shell_parses_with_current_daemon_renderer(
    tmp_path: Path,
) -> None:
    job = json.loads(PACKET.read_text())
    assert job["packet_role"] == "profile_only_no_fit_checkpoint"
    assert job["expected_full_fit_artifact"] is False
    assert [step["label"] for step in job["steps"]] == ["preflight", "profile", "validate"]
    for step in job["steps"]:
        argv = step["argv"]
        assert "-c" not in argv
        assert all("'" not in value for value in argv)
    profile_argv = job["steps"][1]["argv"]
    assert profile_argv[profile_argv.index("--profile-steps") + 1] == "3"
    assert profile_argv[profile_argv.index("--profile-warmup") + 1] == "5"
    assert profile_argv[profile_argv.index("--name") + 1] == job["name"]
    assert profile_argv[profile_argv.index("--name") + 1].endswith("_r2")

    manifest = json.loads(PACKET_MANIFEST.read_text())
    assert manifest["status"] == "prepared_not_queued_not_run"
    assert manifest["job_packet"]["sha256"] == hashlib.sha256(PACKET.read_bytes()).hexdigest()
    for helper in manifest["external_helpers"]:
        assert (
            helper["sha256"]
            == hashlib.sha256((PROJECT / helper["local_path"]).read_bytes()).hexdigest()
        )
        assert helper["path"].startswith("/root/wnn/")
    assert (
        manifest["daemon_renderer"]["sha256"]
        == hashlib.sha256(PARENT_DAEMON.read_bytes()).hexdigest()
    )

    shell = _render_like_daemon(job, tmp_path / "packet-logs")
    checked = subprocess.run(
        ["bash", "-n", "-c", shell], capture_output=True, text=True, check=False
    )
    assert checked.returncode == 0, checked.stderr


def test_daemon_rendered_argv_round_trip_preserves_packet_cli_tokens(tmp_path: Path) -> None:
    packet = json.loads(PACKET.read_text())
    preflight_argv = packet["steps"][0]["argv"]
    cli_start = preflight_argv.index("--project-root")
    cli_args = preflight_argv[cli_start:]
    fixture = tmp_path / "argv_fixture.py"
    fixture.write_text("import json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    job = {
        "name": "argv_roundtrip",
        "steps": [
            {
                "label": "fixture",
                "argv": [
                    sys.executable,
                    str(fixture),
                    *cli_args,
                    "--fixture-extra",
                    "path with spaces",
                ],
            }
        ],
    }
    shell = _render_like_daemon(job, tmp_path / "roundtrip-logs")
    checked = subprocess.run(
        ["bash", "-n", "-c", shell], capture_output=True, text=True, check=False
    )
    assert checked.returncode == 0, checked.stderr
    run = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    observed = json.loads((tmp_path / "roundtrip-logs/argv_roundtrip.fixture.log").read_text())
    assert observed == [*cli_args, "--fixture-extra", "path with spaces"]


@pytest.mark.parametrize(
    "bad_profile",
    [
        {"gpu_total": float("nan"), "step_wall": 1, "seq_per_s": 1},
        {"gpu_total": 1, "step_wall": 0, "seq_per_s": 1},
    ],
)
def test_profile_validator_rejects_nonfinite_or_missing_throughput_metrics(
    tmp_path: Path, bad_profile: dict[str, float]
) -> None:
    name = "invalid_profile"
    project = tmp_path / "wnn"
    run_dir = project / "artifacts/runs" / name
    run_dir.mkdir(parents=True)
    preflight_log = tmp_path / "preflight.log"
    preflight_log.write_text("H377_SOURCE_INPUT_PREFLIGHT_OK {}\n")
    profile_log = tmp_path / "profile.log"
    profile_log.write_text(
        json.dumps({"name": name, "profile_steps": 3, "profile_warmup": 5})
        + "\nPROFILE "
        + json.dumps(bad_profile)
        + "\n"
    )
    args = [
        "--preflight-log",
        str(preflight_log),
        "--profile-log",
        str(profile_log),
        "--run-dir",
        str(run_dir),
        "--project-root",
        str(project),
        "--run-name",
        name,
        "--profile-steps",
        "3",
        "--profile-warmup",
        "5",
    ]
    result = _run_helper(VALIDATOR, args, os.environ.copy())
    assert result.returncode == 1
