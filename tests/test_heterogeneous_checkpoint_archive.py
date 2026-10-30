from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.archive_heterogeneous_checkpoints import write_zip
from scripts.heterogeneous_checkpoint_archive_contract import (
    canonical,
    capture,
    encoded,
    fit_receipt,
    sha256,
    validate_run,
)


def test_zip_retains_original_payload_bytes_and_rejects_later_mutation(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    payload = bytes(range(256)) * 17
    path.write_bytes(payload)
    entry = capture(path, "run/checkpoints/model.pt")
    output = tmp_path / "weights.zip"
    receipt = write_zip(output, [entry], {"scope": "synthetic byte-format fixture"}, "fixture only")
    with zipfile.ZipFile(output) as archive:
        assert archive.read(entry.destination) == payload
    assert receipt["payload_roundtrip_verified"] is True
    assert receipt["sha256"] == sha256(output)
    path.write_bytes(b"mutated checkpoint")
    with pytest.raises(ValueError, match="changed during archive"):
        write_zip(tmp_path / "mutated.zip", [entry], {}, "fixture")
    assert not (tmp_path / "mutated.zip").exists()
    assert (tmp_path / "mutated.zip.partial").is_file()


def test_metadata_is_captured_once_and_archive_names_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    original = b'{"producer":"captured once"}\n'
    path.write_bytes(original)
    entry = capture(path, "run/metadata/provenance.json", text=True)
    path.write_text('{"producer":"new observation"}')
    write_zip(tmp_path / "metadata.zip", [entry], {}, "fixture")
    with zipfile.ZipFile(tmp_path / "metadata.zip") as archive:
        assert archive.read(entry.destination) == original
    with pytest.raises(ValueError, match="unsafe archive"):
        capture(path, "../foreign.json", text=True)
    with pytest.raises(ValueError, match="apparent credential"):
        encoded({"token": "hf_" + "a" * 30})


@pytest.mark.parametrize("phase", ["fit", "scientific_fit"])
def test_only_successful_owned_fit_receipts_are_checkpoint_sources(
    tmp_path: Path, phase: str
) -> None:
    requests, statuses = tmp_path / "requests", tmp_path / "status"
    requests.mkdir()
    statuses.mkdir()
    command = ["python", "/source/scripts/train_rnn.py", "--name", "het6_control_s1301"]
    request = requests / "fit.json"
    request.write_text(json.dumps({"phase": phase, "command": command}))
    status_path = statuses / request.name
    status = {
        "schema": "wnn.owned-command.v1",
        "status": "running",
        "return_code": None,
        "request_sha256": sha256(request),
        "command_sha256": canonical(command),
    }
    status_path.write_text(json.dumps(status))
    assert fit_receipt(tmp_path, "het6_control_s1301") is None
    status.update(status="complete", return_code=0)
    status_path.write_text(json.dumps(status))
    assert Path(fit_receipt(tmp_path, "het6_control_s1301")[0]["source_root"]) == Path("/source")
    status["command_sha256"] = "0" * 64
    status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="receipt identity mismatch"):
        fit_receipt(tmp_path, "het6_control_s1301")


def test_source_digest_and_membership_accept_permutations_but_not_leakage(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    script = source / "model.py"
    script.write_text("# immutable source fixture\n")
    files = {"model.py": sha256(script)}
    split = {"fit": {"train": [1, 2], "valid": []}, "evaluation": {"valid": [0, 1]}}
    provenance = {
        "args": {
            "name": "het6_control_s1301",
            "epochs": 4,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 3,
            "precision": "fp32",
        },
        "selected_epoch": 3,
        "split_manifest_sha256": "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c",
        "fit_row_groups": {"train": [2, 1], "valid": []},
        "evaluation_row_groups": [1, 0],
        "trainer_implementation": {"sha256": canonical(files), "files": files},
    }
    history = [
        {
            "epoch": epoch,
            "selected_rows": 3528273,
            "split": "vfold1873",
            "weighted_pearson": 0.6 + epoch * 0.01,
        }
        for epoch in range(4)
    ]
    validate_run("het6_control_s1301", 4, provenance, history, split, source)
    provenance["fit_row_groups"]["valid"] = [0]
    with pytest.raises(ValueError, match="official membership"):
        validate_run("het6_control_s1301", 4, provenance, history, split, source)
    provenance["fit_row_groups"]["valid"] = []
    script.write_text("# changed source\n")
    with pytest.raises(ValueError, match="identity changed"):
        validate_run("het6_control_s1301", 4, provenance, history, split, source)
