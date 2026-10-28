from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHARED = REPO / "research/experiment_drafts/source-holdout-transformer-kd-20261028"
DEFAULT = "source-holdout-transformer-kd-20261028"
QUALITY = "transformer-width-context-20261028"


def _inspect(study: str, source: str) -> subprocess.CompletedProcess[str]:
    """Isolate import-time study selection without touching already imported gates."""
    environment = os.environ | {"PYTHONPATH": str(SHARED), "WNN_SEQUENCE_STUDY": study}
    return subprocess.run(
        [sys.executable, "-c", source],
        env=environment,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def test_quality_cells_change_only_the_declared_architecture() -> None:
    """Hold all numerical recipes fixed while widening or lengthening attention."""
    result = _inspect(
        QUALITY,
        "import json; from packet_contract import CELLS, normalized_config; print(json.dumps({c: normalized_config(c) for c in CELLS}))",
    )
    assert result.returncode == 0, result.stderr
    configs = json.loads(result.stdout)
    assert len(configs) == 6
    for seed in (1001, 1002):
        baseline = configs[f"baseline_s{seed}"]
        wide, context = configs[f"wide_s{seed}"], configs[f"context_s{seed}"]
        assert wide == baseline | {"model": baseline["model"] | {"hidden": 1024, "heads": 16}}
        assert context == baseline | {"model": baseline["model"] | {"context_rows": 4096}}
        assert baseline["model"]["architecture"] == "transformer"
        assert baseline["model"]["hidden"] == 768 and baseline["model"]["heads"] == 12
        assert baseline["model"]["context_rows"] == 1024
        assert baseline["model"]["layers"] == 16
        assert baseline["kd_weight"] == baseline["hint_weight"] == 0
        assert baseline["epochs"] == 5 and baseline["checkpoint_retention"] == "final_only"
        assert baseline["precision_mode"] == "bf16_bf16x9"
        assert baseline["batch_size"] == 128 and baseline["required_world_size"] == 2
        assert baseline["chunk_rows"] == 512 and baseline["accumulate_chunks"] == 8


def test_default_study_stays_the_original_teacher_student_panel() -> None:
    """The new wrapper must not silently replace the prior study's six cells."""
    result = _inspect(
        DEFAULT, "import json; from packet_contract import CELLS; print(json.dumps(CELLS))"
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        f"{role}_s{seed}" for seed in (1001, 1002) for role in ("teacher", "labels", "kd")
    ]


@pytest.mark.parametrize("study", ["", "../outside", "/tmp/foreign", "unknown-study"])
def test_unregistered_study_is_rejected_before_config_loading(study: str) -> None:
    """An environment variable cannot redirect execution to arbitrary configuration."""
    result = _inspect(study, "import packet_contract")
    assert result.returncode != 0
    assert "Unregistered sequence study" in result.stderr


def test_quality_packet_binds_shared_runtime_and_own_configs() -> None:
    """A shared-code modification must invalidate an already prepared quality root."""
    result = _inspect(
        QUALITY,
        "import json; from packet_contract import packet_files; print(json.dumps(packet_files()))",
    )
    assert result.returncode == 0, result.stderr
    names = json.loads(result.stdout)
    prefix = "research/experiment_drafts/"
    assert prefix + DEFAULT + "/packet_runtime.py" in names
    assert prefix + DEFAULT + "/packet_study.py" in names
    assert prefix + QUALITY + "/configs/context_s1002.json" in names
    assert prefix + QUALITY + "/run.sh" in names


def test_quality_panel_refuses_cache_and_foreign_cells() -> None:
    """A pure Transformer study cannot launch its dependency's KD or GRU commands."""
    source = """
from types import SimpleNamespace
from packet_runtime import command_for
for cell, action in [('baseline_s1001', 'cache'), ('kd_s1001', 'profile')]:
    try:
        command_for(SimpleNamespace(cell=cell, action=action))
    except ValueError:
        continue
    raise RuntimeError('foreign work admitted')
"""
    result = _inspect(QUALITY, source)
    assert result.returncode == 0, result.stderr


def test_prepared_root_cannot_cross_studies(tmp_path: Path) -> None:
    """A valid schema and caller-selected root cannot change the frozen panel."""
    (tmp_path / "packet.json").write_text(
        json.dumps({"schema": "wnn.source-kd-packet.v1", "study": DEFAULT})
    )
    source = f"""
from pathlib import Path
import packet_contract
packet_contract.guard = lambda root, commit: None
try:
    packet_contract.binding(Path({str(tmp_path)!r}), Path('/data'), 'a' * 40)
except ValueError as error:
    assert 'another study' in str(error)
else:
    raise RuntimeError('cross-study root admitted')
"""
    result = _inspect(QUALITY, source)
    assert result.returncode == 0, result.stderr
