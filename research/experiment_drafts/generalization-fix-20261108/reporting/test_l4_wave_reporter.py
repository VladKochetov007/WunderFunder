from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from collect_l4_readouts import collect
from l4_readout_mirror import ReadoutMirror, apply_readouts
from l4_wave_evidence import (
    CELLS,
    ROWS,
    SOURCE,
    SPLIT,
    UPDATES,
    fit_complete,
    full_valid,
    inventory,
)
from l4_wave_publish import publish
from l4_wave_readouts import REFERENCE, grade
from l4_wave_report import render, snapshot

from scripts.sequence_progress_report import canonical_sha


def terminal_fixture() -> tuple[dict, dict, list[dict]]:
    ids = {"source": SOURCE, "config": "c" * 64, "data": "d" * 64, "split": SPLIT}
    identity = {f"{key}_sha256": ids[key] for key in ("source", "config", "data")} | {
        "world_size": 1
    }
    terminal = {
        "status": "completed",
        "completion_kind": "max_epochs",
        "completed_epochs": 6,
        "requested_epochs": 6,
        "global_updates": UPDATES,
        "model_sha256": "a" * 64,
        "identity": identity,
    }
    events = [
        {
            "kind": "evaluation",
            "epoch": epoch,
            "step": epoch * 6460,
            "context": {
                "valid": {"split_sha256": SPLIT},
                "run_identity_sha256": canonical_sha(identity),
            },
            "metrics": {
                "valid/sequence_count": 1873,
                "valid/selected_rows": ROWS,
                "valid/wp": 0.6,
                "valid/wp/t0": 0.5,
                "valid/wp/t1": 0.7,
            },
        }
        for epoch in range(1, 7)
    ]
    return terminal, ids, events


def test_exact_budget_full_validation_and_profile_refusal(tmp_path: Path) -> None:
    terminal, ids, events = terminal_fixture()
    assert all(full_valid(event) for event in events)
    assert fit_complete(terminal, ids, events)
    for key, value in (
        ("completed_epochs", 12),
        ("global_updates", UPDATES - 1),
        ("model_sha256", "bad"),
        ("status", "profile_complete"),
    ):
        wrong = terminal | {key: value}
        assert not fit_complete(wrong, ids, events)
    assert not fit_complete(terminal, ids, events[:-1])
    assert not fit_complete(terminal, ids, events[:-1] + [events[-2]])
    assert not full_valid(events[0] | {"step": 20})
    assert not fit_complete(terminal, ids | {"source": "e" * 64}, events)
    (tmp_path / f"l4-{CELLS[0]}-profile").mkdir()
    value = snapshot(tmp_path, None, {}, 100)
    assert len(value["cells"]) == 6 and not any(row["fit_complete"] for row in value["cells"])
    assert set(inventory()) == set(CELLS)


def grade_fixture() -> tuple[dict, dict, dict, dict]:
    terminal, ids, _ = terminal_fixture()
    config = {"epochs": 6}
    ids["config"] = canonical_sha(config)
    terminal["identity"]["config_sha256"] = ids["config"]
    row = {"fit_complete": True, "terminal": terminal, "ids": ids}
    result = {
        "metric_id": "global_wp_clip2_v1",
        "selected_rows": ROWS,
        "wp": 0.6,
        "t0": 0.5,
        "t1": 0.7,
        "wmse": 0.2,
    }
    entries = [
        {
            "source": "valid",
            "row_group": i,
            "sha256": "f" * 64,
            "selected_rows": ROWS if i == 0 else 0,
        }
        for i in range(1873)
    ]
    index = {
        "completed_epochs": 6,
        "identity": terminal["identity"],
        "index_sha256": "e" * 64,
        "entries": [{"checkpoint_sha256": f"{i:064x}"} for i in range(1, 7)],
    }
    index["index_sha256"] = canonical_sha({k: v for k, v in index.items() if k != "index_sha256"})
    approved = {"source_sha256": "b" * 64}
    parents = [
        {
            "completed_epochs": epoch,
            "checkpoint_sha256": index["entries"][epoch - 1]["checkpoint_sha256"],
            "index_sha256": index["index_sha256"],
            "identity_sha256": canonical_sha(terminal["identity"]),
        }
        for epoch in (3, 4, 5, 6)
    ]
    receipt = {
        "schema": "wnn.fixed-swa-e6-readout.v1",
        "complete": True,
        "readout": "swa3456",
        "split_sha256": SPLIT,
        "sequence_count": 1873,
        "selected_rows": ROWS,
        "reference": {"manifest_sha256": REFERENCE},
        "mixture_weights": {"reference": 0.9, "candidate": 0.1},
        "readout_implementation": {"sha256": approved["source_sha256"]},
        "config": config,
        "checkpoint_sha256": "9" * 64,
        "producer": {
            "identity": terminal["identity"],
            "swa": {
                "epochs": [3, 4, 5, 6],
                "weights": [0.25] * 4,
                "inference_only": True,
                "implementation_source": {"sha256": approved["source_sha256"]},
                "parents": parents,
            },
        },
        "metrics": {
            "candidate": result.copy(),
            "mixture": result.copy(),
            "reference": result | {"wp": 0.6923774248169172},
        },
        "sequences": entries,
    }
    return receipt, row, index, approved


def test_fixed_readout_identity_and_parent_hash_gates() -> None:
    receipt, row, index, approved = grade_fixture()
    assert grade(receipt, "swa3456", row, index, approved)
    assert not grade(receipt, "swa3456", row | {"fit_complete": False}, index, approved)
    assert not grade(receipt, "swa3456", row, index, {})
    for field, value in (
        ("selected_rows", ROWS - 1),
        ("split_sha256", "0" * 64),
        ("readout_implementation", {"sha256": "0" * 64}),
    ):
        assert not grade(receipt | {field: value}, "swa3456", row, index, approved)
    mutated = copy.deepcopy(receipt)
    mutated["producer"]["swa"]["parents"][0]["checkpoint_sha256"] = "0" * 64
    assert not grade(mutated, "swa3456", row, index, approved)
    e6 = receipt | {"readout": "epoch6", "checkpoint_sha256": row["terminal"]["model_sha256"]}
    assert grade(e6, "epoch6", row, index, approved)
    assert not grade(e6 | {"checkpoint_sha256": "8" * 64}, "epoch6", row, index, approved)


def test_actual_readout_bytes_and_digest_refusals(tmp_path: Path) -> None:
    native, local = tmp_path / "runs", tmp_path / "mirror"
    source = native / CELLS[0] / "readouts/epoch6/readout.json"
    source.parent.mkdir(parents=True)
    raw = b'{"complete":true,"note":"actual bytes"}\n'
    source.write_bytes(raw)
    value = collect(str(native), {})
    digest = hashlib.sha256(raw).hexdigest()
    name = f"{CELLS[0]}/epoch6/readout.json"
    assert apply_readouts(value, local) == {name: digest}
    assert (local / name).read_bytes() == raw
    assert collect(str(native), {name: digest})["files"] == []
    wrong = copy.deepcopy(value)
    wrong["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        apply_readouts(wrong, local)
    wrong["files"][0]["name"] = "../foreign/readout.json"
    with pytest.raises(ValueError, match="exact"):
        apply_readouts(wrong, local)
    conflicting = copy.deepcopy(value)
    conflicting["files"][0].update(
        data_base64=base64.b64encode(b"{}").decode(),
        bytes=2,
        sha256=hashlib.sha256(b"{}").hexdigest(),
    )
    with pytest.raises(ValueError, match="different source bytes"):
        apply_readouts(conflicting, local)
    assert (local / name).read_bytes() == raw


def test_rpc_atomic_request_pending_and_partial_result(tmp_path: Path) -> None:
    rpc = tmp_path / "rpc"
    (rpc / "requests").mkdir(parents=True)
    (rpc / "results").mkdir()
    path = tmp_path / "mirror-state.json"
    native = tmp_path / "native-runs"
    native.mkdir()
    mirror = ReadoutMirror(rpc, str(native), tmp_path / "out", path)
    mirror.tick(100)
    pending = mirror.state["pending"]
    assert len(list((rpc / "requests").glob("*.json"))) == 1
    request = json.loads((rpc / "requests" / f"{pending}.json").read_bytes())
    with redirect_stdout(io.StringIO()) as captured:
        exec(compile(request["code"], "fixture-rpc", "exec"), {})
    assert json.loads(captured.getvalue())["schema"] == "wnn.l4-fixed-readout-bytes.v1"
    mirror = ReadoutMirror(rpc, str(native), tmp_path / "out", path)
    mirror.tick(101)
    assert mirror.state["pending"] == pending
    (rpc / "requests" / f"{pending}.json").unlink()
    mirror.tick(101.5)
    assert (
        json.loads((rpc / "requests" / f"{pending}.json").read_bytes())
        == mirror.state["pending_request"]
    )
    result = rpc / "results" / f"{pending}.json"
    result.write_text('{"id":')
    mirror.tick(102)
    assert mirror.state["pending"] == pending
    value = {
        "schema": "wnn.l4-fixed-readout-bytes.v1",
        "collected_at_unix": 103,
        "files": [],
        "errors": [],
    }
    output = f"WNN_RPC_BEGIN {pending}\n{json.dumps(value)}\nWNN_RPC_END {pending} ok\n"
    result.write_text(json.dumps({"id": pending, "output": output}))
    mirror.tick(103)
    assert "pending" not in mirror.state and mirror.state["last_collected_unix"] == 103


class FakeGitHub:
    def __init__(self, uncertain: bool = False) -> None:
        self.posts, self.uncertain, self.found = 0, uncertain, None

    def find_report(self, number: int, marker: str, prepared: float) -> dict | None:
        return self.found

    def post_report(self, number: int, body: str, *, before_post: object) -> dict:
        before_post()
        self.posts += 1
        if self.uncertain:
            raise RuntimeError("uncertain request")
        return {"id": 123}


def test_publication_restart_and_uncertain_post_are_not_duplicated(tmp_path: Path) -> None:
    value = snapshot(tmp_path, None, {}, 100)
    path, state, client = tmp_path / "state.json", {}, FakeGitHub(uncertain=True)
    publish(value, 71, state, path, client, 100)
    assert client.posts == 1
    state = json.loads(path.read_bytes())
    publish(value, 71, state, path, client, 401)
    assert client.posts == 1
    client.found = {"id": 123}
    publish(value, 71, state, path, client, 702)
    assert client.posts == 1 and "pending" not in state["pull_requests"]["71"]
    publish(value, 71, state, path, client, 703)
    assert client.posts == 1
    assert "0/6 fits complete" in render(value) and "0/24" in render(value)
