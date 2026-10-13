from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.build_cross_team_submission import _write_deterministic_zip, parse_recipe


def _recipe(path: Path, members: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "name": "kotiki_xteam_s6_z4_h390s3_q8_20261013",
                "members": members,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_parse_recipe_accepts_frozen_four_member_mix(tmp_path: Path) -> None:
    path = _recipe(
        tmp_path / "recipe.json",
        [
            {"candidate": "pair_ftt2", "weights": "weights_s6.npz", "coef": 0.25},
            {"candidate": "pair_ftt2", "weights": "weights_ftt6.npz", "coef": 0.25},
            {"candidate": "E", "weights": "weights_2.npz", "coef": 0.25},
            {"candidate": "E", "weights": "weights_0.npz", "coef": 0.25},
        ],
    )
    name, members = parse_recipe(path)
    assert name == "kotiki_xteam_s6_z4_h390s3_q8_20261013"
    assert [(member["candidate"], member["weights"], member["coef"]) for member in members] == [
        ("pair_ftt2", "weights_s6.npz", 0.25),
        ("pair_ftt2", "weights_ftt6.npz", 0.25),
        ("E", "weights_2.npz", 0.25),
        ("E", "weights_0.npz", 0.25),
    ]


@pytest.mark.parametrize(
    "members, message",
    [
        (
            [
                {"candidate": "pair_ftt2", "weights": "weights_s6.npz", "coef": 0.5},
                {"candidate": "pair_ftt2", "weights": "weights_s6.npz", "coef": 0.5},
            ],
            "duplicate recipe weight",
        ),
        (
            [{"candidate": "pair_ftt2", "weights": "weights_s6.npz", "coef": 0.9}],
            "expected 1",
        ),
        (
            [{"candidate": "pair_ftt2", "weights": "weights_s6.npz", "coef": float("inf")}],
            "finite and non-negative",
        ),
        (
            [{"candidate": "pair_ftt2", "weights": "det_a.npz", "coef": 1.0}],
            "predictor NPZ basename",
        ),
    ],
)
def test_parse_recipe_rejects_invalid_member_lists(
    tmp_path: Path, members: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_recipe(_recipe(tmp_path / "bad.json", members))


def test_deterministic_zip_has_stable_sorted_entries_and_timestamps(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    (package / "v2").mkdir(parents=True)
    (package / "solution.py").write_text("# fixed callback\n", encoding="utf-8")
    (package / "v2" / "wnn_native.abi3.so").write_bytes(b"native fixture")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_record = _write_deterministic_zip(package, first, 100_000)
    second_record = _write_deterministic_zip(package, second, 100_000)

    assert first_record["sha256"] == second_record["sha256"]
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["solution.py", "v2/wnn_native.abi3.so"]
        assert {entry.date_time for entry in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
        assert archive.testzip() is None


def test_deterministic_zip_rejects_size_over_limit(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "solution.py").write_text("callback fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="limit is 1"):
        _write_deterministic_zip(package, tmp_path / "too-large.zip", 1)
