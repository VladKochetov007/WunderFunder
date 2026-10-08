from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SOURCE_DATA = Path(__file__).resolve().parents[1] / "source" / "src" / "wnn" / "data.py"


class _Column:
    def __init__(self, offset: int, size: int) -> None:
        self.file_offset = offset
        self.data_page_offset = offset
        self.dictionary_page_offset = -1
        self.total_compressed_size = size


class _RowGroup:
    num_columns = 2

    def column(self, index: int) -> _Column:
        return _Column(8_192 + index * 8_192, 4_096)


class _Metadata:
    num_row_groups = 8

    def row_group(self, index: int) -> _RowGroup:
        return _RowGroup()


class _ParquetFile:
    metadata = _Metadata()


def _load_data_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("h232_v6_data_trace", SOURCE_DATA)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned H232 data module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_trace_uses_existing_parquet_reader_range_and_exact_input_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_module = _load_data_module()
    raw_path = tmp_path / "train.parquet"
    raw_path.write_bytes(bytes(40_000))
    trace_path = tmp_path / "rowgroups.tsv"
    monkeypatch.setenv("WNN_CACHE_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("WNN_CACHE_TRAIN_PATH", str(raw_path.resolve()))

    data_module._record_cache_recovery_read(raw_path, 7, _ParquetFile())

    assert trace_path.read_text(encoding="ascii") == "train\t7\t8192\t20480\n"

    other_path = tmp_path / "other.parquet"
    other_path.write_bytes(bytes(40_000))
    data_module._record_cache_recovery_read(other_path, 3, _ParquetFile())
    assert trace_path.read_text(encoding="ascii") == "train\t7\t8192\t20480\n"


def test_trace_rejects_rowgroup_ranges_outside_verified_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_module = _load_data_module()
    raw_path = tmp_path / "train.parquet"
    raw_path.write_bytes(bytes(10_000))
    trace_path = tmp_path / "rowgroups.tsv"
    monkeypatch.setenv("WNN_CACHE_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("WNN_CACHE_TRAIN_PATH", str(raw_path.resolve()))

    with pytest.raises(ValueError, match="outside the parquet input"):
        data_module._record_cache_recovery_read(raw_path, 1, _ParquetFile())

    assert not trace_path.exists()
