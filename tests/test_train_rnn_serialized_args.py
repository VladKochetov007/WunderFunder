from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.train_rnn import serialized_args


def test_argument_paths_are_stored_as_strings_in_json() -> None:
    args = argparse.Namespace(
        student_frontier_contract=Path("/tmp/consumer-contract.json"),
        split_manifest="/tmp/fold0.json",
    )

    serialized = serialized_args(args)

    assert serialized == {
        "student_frontier_contract": "/tmp/consumer-contract.json",
        "split_manifest": "/tmp/fold0.json",
    }
    assert json.loads(json.dumps(serialized)) == serialized
