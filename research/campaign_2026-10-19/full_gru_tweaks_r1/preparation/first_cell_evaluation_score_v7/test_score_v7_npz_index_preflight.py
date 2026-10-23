"""Synthetic metadata fixtures for the bounded selected-index ZIP gate."""

from __future__ import annotations

import io
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import first_cell_score_core as core  # noqa: E402

CAP = core.MAX_MEMBERSHIP_ARRAY_BYTES


class FakeArchive:
    def __init__(self, infos: list[zipfile.ZipInfo], members: dict[str, bytes]) -> None:
        self.infos = infos
        self.members = members
        self.opened: list[str] = []

    def __enter__(self) -> FakeArchive:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def infolist(self) -> list[zipfile.ZipInfo]:
        return self.infos

    def open(self, member: zipfile.ZipInfo, mode: str = "r") -> io.BytesIO:
        if mode != "r":
            raise AssertionError("the preflight must open selected members read-only")
        self.opened.append(member.filename)
        return io.BytesIO(self.members[member.filename])


def npy_member(
    *, dtype: str = "<i8", shape: tuple[object, ...] = (3,), payload: bytes | None = None
) -> bytes:
    header = repr({"descr": dtype, "fortran_order": False, "shape": shape}).encode("latin1")
    padding = 16 - ((10 + len(header) + 1) % 16)
    header += b" " * padding + b"\n"
    prefix = b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header
    if payload is None:
        count = shape[0] if len(shape) == 1 and type(shape[0]) is int else 0
        payload = b"\x00" * (count * 8)
    return prefix + payload


def info(name: str, size: int, *, flags: int = 0) -> zipfile.ZipInfo:
    result = zipfile.ZipInfo(name)
    result.file_size = size
    result.flag_bits = flags
    return result


class NpzIndexPreflightTests(unittest.TestCase):
    def run_gate(
        self,
        infos: list[zipfile.ZipInfo],
        members: dict[str, bytes],
        *,
        selected_only: bool,
        cap: int = CAP,
        required: set[str] | None = None,
    ) -> FakeArchive:
        archive = FakeArchive(infos, members)
        with patch.object(core.zipfile, "ZipFile", return_value=archive):
            core._preflight_npz_members(
                Path("synthetic-only.npz"),
                {"row_group", "seq_ix"} if required is None else required,
                cap,
                selected_only=selected_only,
            )
        return archive

    def test_actual_diagnostic_shapes_pass_without_opening_unselected_members(self) -> None:
        train_row = npy_member(shape=(10_606,))
        train_seq = npy_member(shape=(10_606,))
        infos = [
            info("row_group.npy", len(train_row)),
            info("seq_ix.npy", len(train_seq)),
            info("y.npy", 1_696_960_128),
            info("mask.npy", 212_120_128),
            info("market_class.npy", 84_976),
        ]
        archive = self.run_gate(
            infos,
            {"row_group.npy": train_row, "seq_ix.npy": train_seq},
            selected_only=True,
        )
        self.assertEqual(set(archive.opened), {"row_group.npy", "seq_ix.npy"})

        valid_row = npy_member(shape=(1_873,))
        valid_seq = npy_member(shape=(1_873,))
        valid = self.run_gate(
            [
                info("row_group.npy", len(valid_row)),
                info("seq_ix.npy", len(valid_seq)),
                info("y.npy", 299_680_128),
                info("mask.npy", 37_460_128),
                info("market_class.npy", 15_112),
            ],
            {"row_group.npy": valid_row, "seq_ix.npy": valid_seq},
            selected_only=True,
        )
        self.assertEqual(set(valid.opened), {"row_group.npy", "seq_ix.npy"})

    def test_default_cache_path_still_checks_every_member_without_opening(self) -> None:
        archive = self.run_gate(
            [info("row_group.npy", 64), info("next.npy", 64)],
            {},
            selected_only=False,
            required=set(),
        )
        self.assertEqual(archive.opened, [])
        with self.assertRaisesRegex(ValueError, "unsafe expanded member"):
            self.run_gate(
                [info("row_group.npy", 64), info("large.npy", CAP + 1)],
                {},
                selected_only=False,
                required=set(),
            )

    def test_only_exact_unique_index_names_are_accepted(self) -> None:
        row = npy_member()
        seq = npy_member()
        with self.assertRaisesRegex(ValueError, "lacks unique"):
            self.run_gate(
                [
                    info("row_group.npy", len(row)),
                    info("seq_ix.npy", len(seq)),
                    info("seq_ix.npy", len(seq)),
                ],
                {"row_group.npy": row, "seq_ix.npy": seq},
                selected_only=True,
            )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.run_gate(
                [
                    info("row_group.npy", len(row)),
                    info("seq_ix.npy", len(seq)),
                    info("nested/row_group.npy", len(row)),
                ],
                {"row_group.npy": row, "seq_ix.npy": seq},
                selected_only=True,
            )

    def test_unsafe_or_malformed_selected_headers_are_rejected(self) -> None:
        for label, payload in (
            ("object dtype", npy_member(dtype="|O")),
            ("multidimensional shape", npy_member(shape=(3, 1))),
            ("boolean dimension", npy_member(shape=(True,))),
            ("payload mismatch", npy_member(shape=(3,), payload=b"\x00" * 8)),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.run_gate(
                    [info("row_group.npy", len(payload)), info("seq_ix.npy", len(payload))],
                    {"row_group.npy": payload, "seq_ix.npy": payload},
                    selected_only=True,
                )

        with self.assertRaisesRegex(ValueError, "unsafe expanded member"):
            self.run_gate(
                [info("row_group.npy", CAP + 1), info("seq_ix.npy", CAP + 1)],
                {"row_group.npy": b"not-opened", "seq_ix.npy": b"not-opened"},
                selected_only=True,
            )
        header_over_limit = b"\x93NUMPY\x01\x00" + (core.MAX_NPY_HEADER_BYTES + 1).to_bytes(
            2, "little"
        )
        with self.assertRaisesRegex(ValueError, "header exceeds its bound"):
            self.run_gate(
                [
                    info("row_group.npy", len(header_over_limit)),
                    info("seq_ix.npy", len(header_over_limit)),
                ],
                {"row_group.npy": header_over_limit, "seq_ix.npy": header_over_limit},
                selected_only=True,
            )
        with self.assertRaisesRegex(ValueError, "unsafe expanded member"):
            self.run_gate(
                [info("row_group.npy", 64, flags=1), info("seq_ix.npy", 64)],
                {"row_group.npy": npy_member(), "seq_ix.npy": npy_member()},
                selected_only=True,
            )


if __name__ == "__main__":
    unittest.main()
