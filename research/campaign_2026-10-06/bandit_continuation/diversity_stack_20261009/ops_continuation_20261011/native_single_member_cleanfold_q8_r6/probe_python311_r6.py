#!/usr/bin/env python3
"""Inspect the frozen Python 3.11 runtime without importing/installing packages."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REQUIRED = ("numpy", "pyarrow")
resolved = str(Path(sys.executable).resolve())
available = {name: importlib.util.find_spec(name) is not None for name in REQUIRED}
passed = (
    resolved == "/usr/bin/python3.11"
    and tuple(sys.version_info[:2]) == (3, 11)
    and all(available.values())
)
print(
    json.dumps(
        {
            "schema": "wnn.native.single-member-cleanfold-q8-python311-probe.r6",
            "status": "PASS_PY311_DEPENDENCY_SPECS_PRESENT"
            if passed
            else "FAIL_PY311_DEPENDENCY_PROBE",
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "resolved_executable": resolved,
            "prefix": sys.prefix,
            "required_dependencies": available,
            "packages_imported": [],
            "package_install_or_environment_creation": False,
        },
        sort_keys=True,
    )
)
