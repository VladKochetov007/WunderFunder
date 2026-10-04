from __future__ import annotations

import importlib.util
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "submission_abi", Path(__file__).resolve().parents[1] / "scripts/check_submission_abi.py"
)
assert SPEC is not None and SPEC.loader is not None
abi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(abi)


def test_abi3_filename_does_not_hide_python313_dependency(tmp_path: Path) -> None:
    """Reproduce the actual failure mode with a real linked ELF, not a mock."""
    if shutil.which("cc") is None or shutil.which("readelf") is None:
        pytest.skip("ELF regression needs a C compiler and readelf")
    dependency = tmp_path / "python.c"
    dependency.write_text("int PyFake(void) { return 13; }\n")
    library = tmp_path / "libpython3.13.so.1.0"
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-Wl,-soname,libpython3.13.so.1.0",
            str(dependency),
            "-o",
            str(library),
        ],
        check=True,
    )
    source = tmp_path / "extension.c"
    source.write_text("extern int PyFake(void); int extension(void) { return PyFake(); }\n")
    extension = tmp_path / "wnn_native.abi3.so"
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(source), str(library), "-o", str(extension)], check=True
    )
    package = tmp_path / "submission.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.write(extension, "v2/wnn_native.abi3.so")
    report = abi.inspect_artifact(package)
    assert report["passed"] is False
    assert "libpython3.13.so.1.0" in report["libraries"]["v2/wnn_native.abi3.so"]["needed"]


def test_empty_native_build_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no native libraries"):
        abi.inspect_artifact(tmp_path)
