"""Verify installed upstream source bytes instead of trusting its version string."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files

from . import UPSTREAM_REVISION


def verify_installation() -> dict[str, str]:
    """Require the exact reviewed upstream Python sources before any refinement."""
    manifest = json.loads(files(__package__).joinpath("upstream.json").read_text(encoding="utf-8"))
    if manifest["revision"] != UPSTREAM_REVISION:
        raise ValueError("Upstream manifest revision does not match the integration")
    package = files("harness2")
    actual = {}
    for name, expected in manifest["files"].items():
        digest = hashlib.sha256(
            package.joinpath(name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        if digest != expected:
            raise ValueError(f"Installed Harness² differs from reviewed source: {name}")
        actual[name] = digest
    return actual
