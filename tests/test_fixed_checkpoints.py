"""Fixed checkpoint decisions cannot depend on evaluation scores or raw EMA weights."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_rnn as training  # noqa: E402
from train_rnn import (  # noqa: E402
    checkpoint_model,
    checkpoint_selected,
    configure_checkpoint_policy,
    inherit_init_arguments,
    validate_campaign_checkpoint,
)


def test_fixed_checkpoint_is_independent_of_evaluation() -> None:
    for score in (None, -1.0, 1.0):
        assert checkpoint_selected(3, "fixed", 3, score, 0.9)
        assert not checkpoint_selected(2, "fixed", 3, score, 0.9)
    assert checkpoint_selected(0, "best", None, 0.5, 0.4)
    with pytest.raises(ValueError, match="requires evaluation"):
        checkpoint_selected(0, "best", None, None, -1.0)


def test_global_evaluation_survives_without_eligible_legacy_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = np.tile(np.linspace(0.1, 1.9, 128, dtype=np.float32)[None, :, None], (2, 1, 2))
    mask = np.ones((2, 128), dtype=bool)
    mask[:, :99] = False
    monkeypatch.setattr(training, "predict", lambda *arguments: targets.copy())
    monkeypatch.setattr(training, "market_class", lambda values: 0)
    loader = [(torch.zeros((2, 128, 1)), torch.from_numpy(targets), torch.from_numpy(mask))]
    result = training.evaluate(None, loader, 4000, torch.device("cpu"))
    assert result["weighted_pearson"] == pytest.approx(1.0)
    assert result["by_class"][0] == pytest.approx(1.0)
    assert result["per_seq_wp"] is None
    assert result["block_by_class"][0] is None


def test_no_eval_and_campaign_fits_require_fixed_policy() -> None:
    args = argparse.Namespace(
        epochs=4,
        checkpoint_policy="best",
        checkpoint_epoch=None,
        no_eval_during_fit=True,
        split_manifest=None,
    )
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        configure_checkpoint_policy(args, parser)
    args.checkpoint_policy = "fixed"
    configure_checkpoint_policy(args, parser)
    assert args.checkpoint_epoch == 3
    args.split_manifest = "fold0.json"
    args.no_eval_during_fit = False
    with pytest.raises(SystemExit):
        configure_checkpoint_policy(args, parser)


def test_ema_snapshot_saves_teacher_weights_without_evaluation(tmp_path: Path) -> None:
    raw = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        raw.weight.fill_(1.0)
    ema = torch.optim.swa_utils.AveragedModel(raw)
    with torch.no_grad():
        ema.module.weight.fill_(2.0)
    chosen = checkpoint_model(raw, ema, None, True)
    path = tmp_path / "model_ep3.pt"
    torch.save({"model": chosen.state_dict(), "epoch": 3}, path)
    snapshot = torch.load(path, weights_only=True)
    assert snapshot["epoch"] == 3
    assert snapshot["model"]["weight"].item() == 2.0
    assert raw.weight.item() == 1.0


def test_campaign_initialization_requires_same_fold_and_withheld_evaluation() -> None:
    args = argparse.Namespace(split_manifest="fold0.json", split_manifest_sha256="fold0")
    source = {
        "epoch": 3,
        "args": {
            "split_manifest_sha256": "fold0",
            "no_eval_during_fit": True,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 3,
        },
    }
    validate_campaign_checkpoint(source, args)
    source["args"]["split_manifest_sha256"] = "fold1"
    with pytest.raises(ValueError, match="provenance"):
        validate_campaign_checkpoint(source, args)


def test_fine_tuning_preserves_parent_preprocessing_and_rejects_conflicts() -> None:
    args = argparse.Namespace(hidden=256, layers=2, online_norm=False, raw_only=False)
    source = {"args": {"hidden": 128, "layers": 2, "online_norm": True, "raw_only": False}}
    inherit_init_arguments(args, source, [])
    assert args.hidden == 128 and args.online_norm
    args.online_norm = False
    with pytest.raises(ValueError, match="disagrees"):
        inherit_init_arguments(args, source, ["--online-norm"])


def test_ft_wrapper_keeps_freezing_in_head_stage_and_inherits_resource_environment(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (tmp_path / "artifacts/logs").mkdir(parents=True)
    shutil.copy(Path(__file__).resolve().parents[1] / "scripts/ft_gclp.sh", scripts / "ft_gclp.sh")
    trace = tmp_path / "trace.jsonl"
    limited = scripts / "limited"
    limited.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['WNN_FT_TRACE'], 'a') as stream:\n"
        "    stream.write(json.dumps({'args': sys.argv[1:], 'workers': os.environ['WNN_LOADER_WORKERS'], "
        "'cpu': os.environ['WNN_CPU_PCT'], 'ram': os.environ['WNN_MEM_MAX']}) + '\\n')\n"
    )
    limited.chmod(0o755)
    environment = dict(
        os.environ,
        WNN_FT_TRACE=str(trace),
        WNN_LOADER_WORKERS="4",
        WNN_CPU_PCT="75",
        WNN_MEM_MAX="36G",
    )
    subprocess.run(
        [
            "bash",
            str(scripts / "ft_gclp.sh"),
            "parent@ep3",
            "output",
            "0",
            "3",
            "--split-manifest",
            "fold0.json",
            "--checkpoint-policy",
            "fixed",
            "--checkpoint-epoch",
            "0",
            "--no-eval-during-fit",
            "--freeze",
            "inp,gru,head.0",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(calls) == 3
    for call in calls:
        assert (call["workers"], call["cpu"], call["ram"]) == ("4", "75", "36G")
    head, full = calls[0]["args"], calls[1]["args"]
    assert head[head.index("--freeze") + 1] == "inp,gru,head.0"
    assert full[full.index("--freeze") + 1] == ""
    assert full[full.index("--init") + 1] == "outputh@ep0"
    for arguments in (head, full):
        assert "--no-eval-during-fit" in arguments
        assert "--inherit-init-preprocessing" in arguments
        assert arguments[arguments.index("--split-manifest") + 1] == "fold0.json"
