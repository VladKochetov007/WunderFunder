from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import resource
import signal
import stat
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

PACKET_ID = "wnn-e463-combo-reduction-tpu-v5e8-20261007-v5"
OUTPUT = Path("/kaggle/working/cloud_preflight_tpu_v5")
INPUT_ROOT = Path("/kaggle/input")
SOURCE_DATASET_OWNER = "vladkochetov07"
SOURCE_DATASET_SLUG = "wnn-jax-realdata-e463-tpu-20261004-v1"
SOURCE_MOUNT_ROOT = INPUT_ROOT / "datasets" / SOURCE_DATASET_OWNER / SOURCE_DATASET_SLUG
SOURCE_MANIFEST_SHA256 = "851898475dfa701f2089ec76cfe9592538aa2e7f9b70ebcb68c0a614e332d6fb"
SOURCE_ARCHIVE_SHA256 = "a29bdf21a910a6d7b403d9ed8c52cf0761618f5ef6576b5f06d70d0a8fd5ec9e"
EXPECTED_CHECKPOINT_SHA256 = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"
EXPECTED_NPZ_SHA256 = "677f604fd86d993c3d4a66a10638755253bfe52f1f578009dbd781c73fe6d04a"
EXPECTED_SIDECAR_SHA256 = "3e65d2a88352d22bc90d88b829f05beee3faa7a66b9de41c7507f521aab12668"
EXPECTED_MODEL_SOURCE_SHA256 = "294eff11d906fe434dba2c4d0ed4741ad2191f756c4a35204474cb229ff939dc"
DIMENSIONS = (211, 256, 2, 23)
GLOBAL_BATCH = 16
LOCAL_BATCH = 2
CHUNK_ROWS = 2000
PARENT_TIMEOUT_SECONDS = 1650
WORKER_TIMEOUT_SECONDS = 1500
STARTED = time.monotonic()
PHASE = "python_entry"
CHILD: subprocess.Popen[bytes] | None = None
TERMINAL: dict[str, Any] = {"status": "starting"}

COMBO_LOSS_REFERENCE_SHA256 = "4e3bbaaea722ba9d36771521fcf3ac495d3e9e747ba8b1f8bb21be3158252d74"
COMBO_LOSS_REFERENCE_B64 = (
    "ZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKZnJvbSBjb2xsZWN0aW9ucy5hYmMg"
    "aW1wb3J0IE1hcHBpbmcsIFNlcXVlbmNlCmZyb20gbWF0aCBpbXBvcnQgZnN1bQoKaW1wb3J0IG51"
    "bXB5IGFzIG5wCgoKZGVmIGNvbXBvbmVudF9zdW1zKAogICAgcHJlZGljdGlvbjogbnAubmRhcnJh"
    "eSwKICAgIGJhdGNoOiBNYXBwaW5nW3N0ciwgbnAubmRhcnJheV0sCiAgICBlbWFfcHJlZGljdGlv"
    "bjogbnAubmRhcnJheSwKKSAtPiBkaWN0W3N0ciwgZmxvYXRdOgogICAgIiIiQ29tcHV0ZSBjb21i"
    "by1sb3NzIG51bWVyYXRvcnMgYW5kIGRlbm9taW5hdG9ycyBpbmRlcGVuZGVudGx5IGluIE51bVB5"
    "LgoKICAgIEFyZ3M6CiAgICAgICAgcHJlZGljdGlvbjogTW9kZWwgb3V0cHV0cyB3aXRoIHNoYXBl"
    "IGBgW2JhdGNoLCB0aW1lLCAyM11gYC4KICAgICAgICBiYXRjaDogU3ludGhldGljIG9yIGhlbGQt"
    "b3V0IHN1cGVydmlzaW9uIGFycmF5cyB3aXRoIHRoZSBmcm96ZW4gbG9zcyBrZXlzLgogICAgICAg"
    "IGVtYV9wcmVkaWN0aW9uOiBGaXhlZCBpbmNvbWluZyBFTUEgdGFyZ2V0cyB3aXRoIHNoYXBlIGBg"
    "W2JhdGNoLCB0aW1lLCAyXWBgLgoKICAgIFJldHVybnM6CiAgICAgICAgVGhlIHJhdyBjb21wb25l"
    "bnQgbnVtZXJhdG9ycyBhbmQgdGhlaXIgZXhhY3Qgbm9ybWFsaXphdGlvbiBtYXNzZXMuCgogICAg"
    "UmFpc2VzOgogICAgICAgIFZhbHVlRXJyb3I6IElmIG91dHB1dCBvciBzdXBlcnZpc2lvbiBzaGFw"
    "ZXMgZG8gbm90IG1hdGNoIHRoZSBmcm96ZW4gb2JqZWN0aXZlLgogICAgIiIiCiAgICBpZiBwcmVk"
    "aWN0aW9uLm5kaW0gIT0gMyBvciBwcmVkaWN0aW9uLnNoYXBlWy0xXSAhPSAyMzoKICAgICAgICBy"
    "YWlzZSBWYWx1ZUVycm9yKCJwcmVkaWN0aW9uIG11c3QgaGF2ZSBzaGFwZSBbYmF0Y2gsIHRpbWUs"
    "IDIzXSIpCiAgICBleHBlY3RlZF9zaGFwZSA9IHByZWRpY3Rpb24uc2hhcGVbOjJdCiAgICBpZiBl"
    "bWFfcHJlZGljdGlvbi5zaGFwZSAhPSAoKmV4cGVjdGVkX3NoYXBlLCAyKToKICAgICAgICByYWlz"
    "ZSBWYWx1ZUVycm9yKCJFTUEgdGFyZ2V0cyBtdXN0IGhhdmUgc2hhcGUgW2JhdGNoLCB0aW1lLCAy"
    "XSIpCgogICAgbGFiZWxzID0gbnAuY2xpcChucC5hc2FycmF5KGJhdGNoWyJsYWJlbHMiXSwgZHR5"
    "cGU9bnAuZmxvYXQ2NCksIC0yLjAsIDIuMCkKICAgIHJvd193ZWlnaHRzID0gbnAuYXNhcnJheShi"
    "YXRjaFsibWFpbl93ZWlnaHRzIl0sIGR0eXBlPW5wLmZsb2F0NjQpCiAgICBmaXhlZF90YXJnZXQg"
    "PSBucC5jbGlwKG5wLmFzYXJyYXkoYmF0Y2hbImZpeGVkX3RhcmdldCJdLCBkdHlwZT1ucC5mbG9h"
    "dDY0KSwgLTIuMCwgMi4wKQogICAgc3RydWN0dXJhbF9sYWJlbHMgPSBucC5hc2FycmF5KGJhdGNo"
    "WyJzdHJ1Y3R1cmFsX2xhYmVscyJdLCBkdHlwZT1ucC5pbnQ2NCkKICAgIG5leHRfd2VpZ2h0cyA9"
    "IG5wLmFzYXJyYXkoYmF0Y2hbIm5leHRfd2VpZ2h0cyJdLCBkdHlwZT1ucC5mbG9hdDY0KQogICAg"
    "bmV4dF9yb3dzID0gbnAuYXNhcnJheShiYXRjaFsibmV4dF9yb3dzIl0sIGR0eXBlPW5wLmZsb2F0"
    "NjQpCiAgICBuZXh0X3VwID0gbnAuYXNhcnJheShiYXRjaFsibmV4dF91cCJdLCBkdHlwZT1ucC5m"
    "bG9hdDY0KQogICAgcHJlZGljdGlvbjY0ID0gbnAuYXNhcnJheShwcmVkaWN0aW9uLCBkdHlwZT1u"
    "cC5mbG9hdDY0KQogICAgZW1hNjQgPSBucC5hc2FycmF5KGVtYV9wcmVkaWN0aW9uLCBkdHlwZT1u"
    "cC5mbG9hdDY0KQogICAgaWYgbGFiZWxzLnNoYXBlICE9ICgqZXhwZWN0ZWRfc2hhcGUsIDIpIG9y"
    "IGZpeGVkX3RhcmdldC5zaGFwZSAhPSBsYWJlbHMuc2hhcGU6CiAgICAgICAgcmFpc2UgVmFsdWVF"
    "cnJvcigibGFiZWxzIGFuZCBmaXhlZCB0YXJnZXRzIG11c3QgaGF2ZSBzaGFwZSBbYmF0Y2gsIHRp"
    "bWUsIDJdIikKICAgIGlmIHN0cnVjdHVyYWxfbGFiZWxzLnNoYXBlICE9ICgqZXhwZWN0ZWRfc2hh"
    "cGUsIDMpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoInN0cnVjdHVyYWwgbGFiZWxzIG11c3Qg"
    "aGF2ZSBzaGFwZSBbYmF0Y2gsIHRpbWUsIDNdIikKICAgIGlmIGFueSgKICAgICAgICB2YWx1ZS5z"
    "aGFwZSAhPSBleHBlY3RlZF9zaGFwZSBmb3IgdmFsdWUgaW4gKHJvd193ZWlnaHRzLCBuZXh0X3dl"
    "aWdodHMsIG5leHRfcm93cywgbmV4dF91cCkKICAgICk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJv"
    "cigicm93LXZhbHVlZCBzdXBlcnZpc2lvbiBtdXN0IG1hdGNoIHByZWRpY3Rpb24gYmF0Y2gvdGlt"
    "ZSBkaW1lbnNpb25zIikKCiAgICBwcmltYXJ5X3dlaWdodHMgPSBucC5hYnMobGFiZWxzKSAqIHJv"
    "d193ZWlnaHRzWy4uLiwgTm9uZV0KICAgIHByaW1hcnlfbnVtID0gbnAuc3VtKAogICAgICAgIHBy"
    "aW1hcnlfd2VpZ2h0cyAqIChwcmVkaWN0aW9uNjRbLi4uLCA6Ml0gLSBmaXhlZF90YXJnZXQpICoq"
    "IDIsCiAgICAgICAgZHR5cGU9bnAuZmxvYXQ2NCwKICAgICkKICAgIGNvbnNpc3RlbmN5X251bSA9"
    "IG5wLnN1bSgKICAgICAgICByb3dfd2VpZ2h0c1suLi4sIE5vbmVdICogKHByZWRpY3Rpb242NFsu"
    "Li4sIDoyXSAtIGVtYTY0KSAqKiAyLAogICAgICAgIGR0eXBlPW5wLmZsb2F0NjQsCiAgICApCgog"
    "ICAgc3RydWN0dXJhbF9udW0gPSAwLjAKICAgIGZvciBzdGFydCwgd2lkdGgsIHRhcmdldF9pbmRl"
    "eCBpbiAoKDIsIDEyLCAwKSwgKDE0LCAzLCAxKSwgKDE3LCA0LCAyKSk6CiAgICAgICAgbG9naXRz"
    "ID0gcHJlZGljdGlvbjY0Wy4uLiwgc3RhcnQgOiBzdGFydCArIHdpZHRoXQogICAgICAgIG1heF9s"
    "b2dpdHMgPSBucC5tYXgobG9naXRzLCBheGlzPS0xLCBrZWVwZGltcz1UcnVlKQogICAgICAgIGxv"
    "Z19ub3JtYWxpemVyID0gbWF4X2xvZ2l0c1suLi4sIDBdICsgbnAubG9nKG5wLnN1bShucC5leHAo"
    "bG9naXRzIC0gbWF4X2xvZ2l0cyksIGF4aXM9LTEpKQogICAgICAgIGNsYXNzX2luZGljZXMgPSBz"
    "dHJ1Y3R1cmFsX2xhYmVsc1suLi4sIHRhcmdldF9pbmRleF0KICAgICAgICBzZWxlY3RlZF9sb2dp"
    "dHMgPSBucC50YWtlX2Fsb25nX2F4aXMobG9naXRzLCBjbGFzc19pbmRpY2VzWy4uLiwgTm9uZV0s"
    "IGF4aXM9LTEpWy4uLiwgMF0KICAgICAgICBzdHJ1Y3R1cmFsX251bSArPSBmbG9hdChucC5zdW0o"
    "KGxvZ19ub3JtYWxpemVyIC0gc2VsZWN0ZWRfbG9naXRzKSAqIHJvd193ZWlnaHRzKSkKCiAgICBu"
    "ZXh0X2Rpc3RhbmNlID0gKG5leHRfcm93cyAtIDMuMCkgLyAyLjAKICAgIG5leHRfbXNlID0gKHBy"
    "ZWRpY3Rpb242NFsuLi4sIDIxXSAtIG5leHRfZGlzdGFuY2UpICoqIDIKICAgIG5leHRfbG9naXQg"
    "PSBwcmVkaWN0aW9uNjRbLi4uLCAyMl0KICAgIG5leHRfYmNlID0gbnAubG9nYWRkZXhwKDAuMCwg"
    "bmV4dF9sb2dpdCkgLSBuZXh0X3VwICogbmV4dF9sb2dpdAogICAgbmV4dF9udW0gPSBucC5zdW0o"
    "KG5leHRfbXNlICsgbmV4dF9iY2UpICogbmV4dF93ZWlnaHRzLCBkdHlwZT1ucC5mbG9hdDY0KQog"
    "ICAgcmV0dXJuIHsKICAgICAgICAicHJpbWFyeV9udW0iOiBmbG9hdChwcmltYXJ5X251bSksCiAg"
    "ICAgICAgInByaW1hcnlfZGVuIjogZmxvYXQobnAuc3VtKHByaW1hcnlfd2VpZ2h0cywgZHR5cGU9"
    "bnAuZmxvYXQ2NCkpLAogICAgICAgICJjb25zaXN0ZW5jeV9udW0iOiBmbG9hdChjb25zaXN0ZW5j"
    "eV9udW0pLAogICAgICAgICJyb3dfZGVuIjogZmxvYXQobnAuc3VtKHJvd193ZWlnaHRzLCBkdHlw"
    "ZT1ucC5mbG9hdDY0KSksCiAgICAgICAgInN0cnVjdHVyYWxfbnVtIjogc3RydWN0dXJhbF9udW0s"
    "CiAgICAgICAgIm5leHRfbnVtIjogZmxvYXQobmV4dF9udW0pLAogICAgICAgICJuZXh0X2RlbiI6"
    "IGZsb2F0KG5wLnN1bShuZXh0X3dlaWdodHMsIGR0eXBlPW5wLmZsb2F0NjQpKSwKICAgIH0KCgpk"
    "ZWYgbG9zc19mcm9tX3N1bXMoc3VtczogTWFwcGluZ1tzdHIsIGZsb2F0XSkgLT4gZmxvYXQ6CiAg"
    "ICAiIiJDb21iaW5lIHJhdyBzdW1zIHVzaW5nIHRoZSBmcm96ZW4gZm91ci10ZXJtIGNvbWJvIG9i"
    "amVjdGl2ZS4KCiAgICBBcmdzOgogICAgICAgIHN1bXM6IFJhdyBudW1lcmF0b3JzIGFuZCBkZW5v"
    "bWluYXRvcnMgZnJvbSA6ZnVuYzpgY29tcG9uZW50X3N1bXNgLgoKICAgIFJldHVybnM6CiAgICAg"
    "ICAgVGhlIHdlaWdodGVkIGNvbWJvIG9iamVjdGl2ZSB2YWx1ZS4KICAgICIiIgogICAgcHJpbWFy"
    "eSA9IHN1bXNbInByaW1hcnlfbnVtIl0gLyBtYXgoc3Vtc1sicHJpbWFyeV9kZW4iXSwgMWUtOCkK"
    "ICAgIGNvbnNpc3RlbmN5ID0gc3Vtc1siY29uc2lzdGVuY3lfbnVtIl0gLyBtYXgoc3Vtc1sicm93"
    "X2RlbiJdLCAxZS04KQogICAgc3RydWN0dXJhbCA9IHN1bXNbInN0cnVjdHVyYWxfbnVtIl0gLyBt"
    "YXgoc3Vtc1sicm93X2RlbiJdLCAxZS04KQogICAgbmV4dF9hdXggPSBzdW1zWyJuZXh0X251bSJd"
    "IC8gbWF4KHN1bXNbIm5leHRfZGVuIl0sIDFlLTgpCiAgICByZXR1cm4gcHJpbWFyeSArIGNvbnNp"
    "c3RlbmN5ICsgMC4zICogc3RydWN0dXJhbCArIDAuMyAqIG5leHRfYXV4CgoKZGVmIGFnZ3JlZ2F0"
    "ZV9zaGFyZHMoc2hhcmRzOiBTZXF1ZW5jZVtNYXBwaW5nW3N0ciwgZmxvYXRdXSkgLT4gZGljdFtz"
    "dHIsIGZsb2F0XToKICAgICIiIlN1bSBjb21wb25lbnQgbnVtZXJhdG9ycyBhbmQgZGVub21pbmF0"
    "b3JzIGJlZm9yZSBnbG9iYWwgbm9ybWFsaXphdGlvbi4KCiAgICBBcmdzOgogICAgICAgIHNoYXJk"
    "czogT25lIHJhdyBjb21wb25lbnQtc3VtIG1hcHBpbmcgcGVyIGRhdGEgc2hhcmQuCgogICAgUmV0"
    "dXJuczoKICAgICAgICBBIHNpbmdsZSBtYXBwaW5nIHJlcHJlc2VudGluZyB0aGUgY29uY2F0ZW5h"
    "dGVkIGdsb2JhbCBiYXRjaC4KCiAgICBSYWlzZXM6CiAgICAgICAgVmFsdWVFcnJvcjogSWYgbm8g"
    "c2hhcmQgdmFsdWVzIGFyZSBzdXBwbGllZC4KICAgICIiIgogICAgaWYgbm90IHNoYXJkczoKICAg"
    "ICAgICByYWlzZSBWYWx1ZUVycm9yKCJhdCBsZWFzdCBvbmUgc2hhcmQgaXMgcmVxdWlyZWQiKQog"
    "ICAga2V5cyA9IHR1cGxlKHNoYXJkc1swXSkKICAgIGlmIGFueSh0dXBsZShzaGFyZCkgIT0ga2V5"
    "cyBmb3Igc2hhcmQgaW4gc2hhcmRzWzE6XSk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiYWxs"
    "IHNoYXJkIG1hcHBpbmdzIG11c3QgaGF2ZSB0aGUgc2FtZSBvcmRlcmVkIGtleXMiKQogICAgcmV0"
    "dXJuIHtrZXk6IGZzdW0oc2hhcmRba2V5XSBmb3Igc2hhcmQgaW4gc2hhcmRzKSBmb3Iga2V5IGlu"
    "IGtleXN9Cg=="
)


def write_json(path: Path, value: object) -> None:
    """Atomically persist a phase or final status for Kaggle's output collector."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_embedded_support() -> Path:
    """Materialize the pinned helper before the isolated TPU child imports it."""
    module_bytes = base64.b64decode(COMBO_LOSS_REFERENCE_B64, validate=True)
    observed_hash = hashlib.sha256(module_bytes).hexdigest()
    if observed_hash != COMBO_LOSS_REFERENCE_SHA256:
        raise RuntimeError("embedded combo-loss helper digest mismatch")
    support_root = OUTPUT / "support"
    support_root.mkdir(parents=True, exist_ok=True)
    module_path = support_root / "combo_loss_reference.py"
    if module_path.exists():
        if module_path.read_bytes() != module_bytes:
            raise RuntimeError("materialized combo-loss helper differs from pinned bytes")
    else:
        temporary = module_path.with_suffix(".py.tmp")
        temporary.write_bytes(module_bytes)
        temporary.replace(module_path)
    support_path = str(support_root)
    if support_path not in sys.path:
        sys.path.insert(0, support_path)
    return support_root


def record_phase(phase: str, scope: str, **details: Any) -> None:
    """Persist and print bounded parent/worker progress for live monitoring."""
    global PHASE
    PHASE = phase
    record = {
        "scope": scope,
        "phase": phase,
        "time_epoch_seconds": time.time(),
        "elapsed_seconds": time.monotonic() - STARTED,
        **details,
    }
    if OUTPUT.is_dir():
        write_json(OUTPUT / f"{scope}_phase.json", record)
    print(json.dumps({"event": "tpu_diagnostic_phase", **record}, sort_keys=True), flush=True)


def sha256_file(path: Path) -> str:
    """Hash data and source incrementally to keep host memory bounded."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_mount_inventory() -> dict[str, object]:
    """Record the provider's real mount layout before resolving private inputs."""
    root_entries = []
    dataset_mounts = []
    if INPUT_ROOT.is_dir():
        for path in sorted(INPUT_ROOT.iterdir()):
            root_entries.append(
                {
                    "name": path.name,
                    "is_dir": path.is_dir(),
                    "first_children": sorted(child.name for child in path.iterdir())[:20]
                    if path.is_dir()
                    else [],
                }
            )
            if path.name != "datasets" or not path.is_dir():
                continue
            for owner_path in sorted(path.iterdir()):
                if not owner_path.is_dir():
                    continue
                for dataset_path in sorted(owner_path.iterdir()):
                    if dataset_path.is_dir():
                        dataset_mounts.append(
                            {
                                "owner": owner_path.name,
                                "dataset": dataset_path.name,
                                "path": str(dataset_path),
                                "first_children": sorted(
                                    child.name for child in dataset_path.iterdir()
                                )[:20],
                            }
                        )
    return {
        "root": str(INPUT_ROOT),
        "exists": INPUT_ROOT.is_dir(),
        "entries": root_entries,
        "dataset_mounts": dataset_mounts,
        "expected_source_mount_path": str(SOURCE_MOUNT_ROOT),
        "source_mount_exists": SOURCE_MOUNT_ROOT.is_dir(),
        "expected_dataset_slug": SOURCE_DATASET_SLUG,
    }


def verify_direct_source_tree() -> dict[str, object]:
    """Verify every read-only file in the manifest-bound direct source mount."""
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Kaggle input root is absent: {INPUT_ROOT}")
    manifest_path = SOURCE_MOUNT_ROOT / "source-manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"E463 manifest is absent from the expected mount: {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != SOURCE_MANIFEST_SHA256:
        raise ValueError(f"E463 source manifest hash changed: {manifest_sha256}")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("dataset") != f"{SOURCE_DATASET_OWNER}/{SOURCE_DATASET_SLUG}"
        or manifest.get("version") != 2
        or manifest.get("format") != "wnn.jax-realdata-preflight-inputs"
        or manifest.get("private") is not True
        or manifest.get("internet") is not False
        or manifest.get("gpu") is not False
        or manifest.get("tpu") is not True
        or manifest.get("archive_sha256") != SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("E463 source manifest privacy/resource/archive contract changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("E463 source manifest has no file inventory")
    bundle_root = manifest_path.parent / "source_bundle"
    if not bundle_root.is_dir() or bundle_root.is_symlink():
        raise FileNotFoundError(f"E463 direct source bundle is absent: {bundle_root}")
    verified: set[str] = set()
    verified_files: dict[str, dict[str, object]] = {}
    for name, identity in sorted(files.items()):
        if not isinstance(name, str) or not isinstance(identity, dict):
            raise ValueError("E463 source manifest contains an invalid file entry")
        relative = PurePosixPath(name)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"unsafe E463 manifest path: {name}")
        path = bundle_root.joinpath(*relative.parts)
        if path.is_symlink():
            raise ValueError(f"E463 manifest-listed source file is a symlink: {name}")
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise FileNotFoundError(f"E463 manifest-listed regular file is absent: {name}")
        digest = sha256_file(path)
        size = path.stat().st_size
        if size != identity.get("size_bytes") or digest != identity.get("sha256"):
            raise ValueError(f"E463 direct source identity mismatch: {name}")
        verified.add(relative.as_posix())
        verified_files[relative.as_posix()] = {"size_bytes": size, "sha256": digest}
    actual_files: set[str] = set()
    for path in bundle_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"E463 direct source tree contains a symlink: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(bundle_root).as_posix())
    if actual_files != verified:
        raise ValueError(
            "E463 direct source tree/manifest member sets differ: "
            f"missing={sorted(verified - actual_files)}, extra={sorted(actual_files - verified)}"
        )
    return {
        "status": "verified_direct_read_only_tree",
        "dataset": f"{SOURCE_DATASET_OWNER}/{SOURCE_DATASET_SLUG}",
        "mount_path": str(manifest_path.parent),
        "source_manifest_sha256": manifest_sha256,
        "declared_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_archive_present": (manifest_path.parent / "source_bundle.zip").is_file(),
        "archive_digest_used_as_gate": False,
        "verification_representation": "frozen manifest SHA plus every direct file size/SHA and exact member set",
        "source_root": str(bundle_root),
        "verified_file_count": len(verified),
        "verified_bytes": sum(int(entry["size_bytes"]) for entry in files.values()),
        "files": verified_files,
    }


def _read_limit(path: Path) -> int | None:
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def host_snapshot() -> dict[str, Any]:
    """Measure host and cgroup RAM before importing JAX or TPU runtime libraries."""
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, raw = line.split(":", 1)
        memory[name] = int(raw.strip().split()[0]) * 1024
    page_size = os.sysconf("SC_PAGE_SIZE")
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    group = Path("/sys/fs/cgroup")
    limit = _read_limit(group / "memory.max")
    current = _read_limit(group / "memory.current")
    if limit is not None and current is not None:
        effective_total = min(memory["MemTotal"], limit)
        cgroup_available = max(0, limit - current)
        available = min(memory["MemAvailable"], cgroup_available)
    else:
        effective_total = memory["MemTotal"]
        available = memory["MemAvailable"]
    reserve = max(2 * 1024**3, int(effective_total * 0.20))
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "host_mem_total_bytes": memory["MemTotal"],
        "host_mem_available_bytes": memory["MemAvailable"],
        "cgroup_memory_limit_bytes": limit,
        "cgroup_memory_current_bytes": current,
        "effective_memory_total_bytes": effective_total,
        "effective_memory_available_bytes": available,
        "required_headroom_bytes": reserve,
        "process_rss_bytes": resident_pages * page_size,
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "headroom_passed": available >= reserve,
    }


def safe_stats(device: Any) -> dict[str, Any]:
    """Convert backend-specific TPU memory stats into a JSON-safe mapping."""
    try:
        values = device.memory_stats()
    except Exception as error:
        return {"memory_stats_error": f"{type(error).__name__}: {error}"}
    if not isinstance(values, dict):
        return {}
    result: dict[str, Any] = {}
    for key, value in values.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            result[str(key)] = value
        else:
            try:
                result[str(key)] = int(value)
            except (TypeError, ValueError, OverflowError):
                result[str(key)] = str(value)
    return result


def hbm_fields(stats: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    """Read current, peak and total HBM capacity across PJRT naming variants."""
    current = stats.get("bytes_in_use")
    if current is None:
        current = stats.get("bytes_reserved")
    peak = stats.get("peak_bytes_in_use")
    if peak is None:
        peak = stats.get("peak_bytes_reserved")
    limit = stats.get("bytes_limit")
    if limit is None:
        limit = stats.get("bytes_reservable_limit")
    if limit is None:
        limit = stats.get("bytes_reserved_limit")
    values = tuple(
        int(value) if isinstance(value, (int, float)) and value >= 0 else None
        for value in (current, peak, limit)
    )
    return values  # type: ignore[return-value]


def tpu_hbm_capacity(jax: Any) -> tuple[int | None, dict[str, Any]]:
    """Use PJRT's capacity stat or the official JAX TPU hardware descriptor."""
    try:
        from jax.experimental.pallas.tpu import get_tpu_info

        info = get_tpu_info()
        per_core = int(info.hbm_capacity_bytes)
        core_count = int(getattr(info, "num_cores", 1))
        return per_core * core_count, {
            "source": "jax.experimental.pallas.tpu.get_tpu_info",
            "hbm_capacity_bytes_per_tensorcore": per_core,
            "tensorcores_per_chip": core_count,
            "chip_version": str(info.chip_version),
        }
    except Exception as error:
        return None, {"source": "unavailable", "error": f"{type(error).__name__}: {error}"}


def device_records(devices: list[Any], fallback_hbm_capacity: int | None) -> list[dict[str, Any]]:
    """Inventory every visible local TPU chip and retain raw HBM telemetry."""
    records = []
    for device in devices:
        stats = safe_stats(device)
        current, peak, limit = hbm_fields(stats)
        capacity_source = "device.memory_stats" if limit is not None else None
        if limit is None and fallback_hbm_capacity is not None:
            limit = fallback_hbm_capacity
            capacity_source = "jax_tpu_hardware_info_fallback"
        records.append(
            {
                "id": int(getattr(device, "id", -1)),
                "platform": str(getattr(device, "platform", "unknown")),
                "device_kind": str(getattr(device, "device_kind", "unknown")),
                "process_index": int(getattr(device, "process_index", -1)),
                "repr": str(device),
                "hbm_current_bytes": current,
                "hbm_peak_bytes": peak,
                "hbm_capacity_bytes": limit,
                "hbm_capacity_source": capacity_source,
                "hbm_required_free_bytes": int(limit * 0.20) if limit else None,
                "memory_stats": stats,
            }
        )
    return records


def verify_inputs(source_root: Path) -> dict[str, Any]:
    """Validate the exact source and epoch-3 JAX export in the verified mount."""
    run = source_root / "artifacts/runs/c20261003_combo_s3_f0"
    checkpoint = run / "model_ep3.pt"
    npz = run / "model_ep3.jax.npz"
    sidecar_path = run / "model_ep3.jax.npz.json"
    model_source = source_root / "src/wnn_jax/rnn.py"
    observed = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "npz_sha256": sha256_file(npz),
        "sidecar_sha256": sha256_file(sidecar_path),
        "model_source_sha256": sha256_file(model_source),
    }
    expected = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "npz_sha256": EXPECTED_NPZ_SHA256,
        "sidecar_sha256": EXPECTED_SIDECAR_SHA256,
        "model_source_sha256": EXPECTED_MODEL_SOURCE_SHA256,
    }
    if observed != expected:
        raise ValueError(f"E463 immutable input hash mismatch: {observed}")
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("npz_sha256") != EXPECTED_NPZ_SHA256:
        raise ValueError("NPZ sidecar does not bind the expected NPZ bytes")
    if sidecar.get("source_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("NPZ sidecar does not bind the frozen epoch-3 Torch checkpoint")
    if sidecar.get("dimensions") != {
        "n_in": 211,
        "hidden": 256,
        "layers": 2,
        "n_out": 23,
    }:
        raise ValueError("NPZ sidecar model dimensions do not match the E463 model")
    return {**observed, "dimensions": sidecar["dimensions"], "sidecar": sidecar}


def check_finite(jnp: Any, value: Any, name: str) -> bool:
    """Return a synchronized finite predicate or raise with the affected tensor."""
    finite = bool(jnp.all(jnp.isfinite(value)))
    if not finite:
        raise FloatingPointError(f"nonfinite {name}")
    return finite


def _combo_sufficient_statistics(
    jax: Any,
    jnp: Any,
    prediction: Any,
    batch: dict[str, Any],
    ema_prediction: Any,
) -> tuple[Any, Any]:
    """Return the frozen combo-loss component numerators and denominators.

    JAX arrays and parameter pytrees are backend-dependent runtime values, so
    ``Any`` is unavoidable at this device boundary.
    """
    labels = jnp.clip(batch["labels"].astype(jnp.float32), -2.0, 2.0)
    row_weights = batch["main_weights"].astype(jnp.float32)
    primary_weights = jnp.abs(labels) * row_weights[..., None]
    primary_num = jnp.sum(
        primary_weights
        * (
            prediction[..., :2].astype(jnp.float32)
            - jnp.clip(batch["fixed_target"].astype(jnp.float32), -2.0, 2.0)
        )
        ** 2
    )
    consistency_num = jnp.sum(
        row_weights[..., None]
        * (prediction[..., :2].astype(jnp.float32) - ema_prediction[..., :2].astype(jnp.float32))
        ** 2
    )
    structural_labels = batch["structural_labels"].astype(jnp.int32)
    structural_num = jnp.asarray(0.0, dtype=jnp.float32)
    for start, width, target_index in ((2, 12, 0), (14, 3, 1), (17, 4, 2)):
        log_probs = jax.nn.log_softmax(prediction[..., start : start + width], axis=-1)
        selected = jnp.take_along_axis(
            log_probs,
            structural_labels[..., target_index, None],
            axis=-1,
        )[..., 0]
        structural_num -= jnp.sum(selected * row_weights)

    next_weights = batch["next_weights"].astype(jnp.float32)
    next_distance = (batch["next_rows"].astype(jnp.float32) - 3.0) / 2.0
    next_mse = (prediction[..., 21].astype(jnp.float32) - next_distance) ** 2
    next_logit = prediction[..., 22].astype(jnp.float32)
    next_bce = jax.nn.softplus(next_logit) - batch["next_up"].astype(jnp.float32) * next_logit
    next_num = jnp.sum((next_mse + next_bce) * next_weights)
    numerators = jnp.stack((primary_num, consistency_num, structural_num, next_num))
    denominators = jnp.stack(
        (
            jnp.sum(primary_weights),
            jnp.sum(row_weights),
            jnp.sum(next_weights),
        )
    )
    return numerators, denominators


def _combo_denominators(jnp: Any, batch: dict[str, Any]) -> Any:
    """Compute only the three global normalization masses for one shard."""
    labels = jnp.clip(batch["labels"].astype(jnp.float32), -2.0, 2.0)
    row_weights = batch["main_weights"].astype(jnp.float32)
    primary_weights = jnp.abs(labels) * row_weights[..., None]
    return jnp.stack(
        (
            jnp.sum(primary_weights),
            jnp.sum(row_weights),
            jnp.sum(batch["next_weights"].astype(jnp.float32)),
        )
    )


def _combo_loss_from_global_sums(jnp: Any, numerators: Any, denominators: Any) -> Any:
    """Combine raw sums only after their masses have been reduced globally."""
    primary = numerators[0] / jnp.maximum(denominators[0], 1e-8)
    consistency = numerators[1] / jnp.maximum(denominators[1], 1e-8)
    structural = numerators[2] / jnp.maximum(denominators[1], 1e-8)
    next_aux = numerators[3] / jnp.maximum(denominators[2], 1e-8)
    return primary + consistency + 0.3 * structural + 0.3 * next_aux


def _synthetic_sharded_batch(np: Any, rows: int) -> tuple[Any, dict[str, Any], Any]:
    """Make distinct replica inputs with strongly unequal objective masses."""
    rng = np.random.default_rng(20261007)
    prefix = (8, LOCAL_BATCH, rows)
    shard_scales = np.asarray([0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 3.0, 8.0], dtype=np.float32)
    labels = rng.normal(0.0, 0.2, (*prefix, 2)).astype(np.float32)
    fixed_target = np.clip(labels * 0.7 + rng.normal(0.0, 0.05, (*prefix, 2)), -2.0, 2.0).astype(
        np.float32
    )
    row_weights = rng.uniform(0.1, 1.0, prefix).astype(np.float32)
    row_weights *= shard_scales[:, None, None]
    row_weights[:, :, ::31] = 0.0
    next_weights = rng.uniform(0.1, 1.0, prefix).astype(np.float32)
    next_weights *= shard_scales[:, None, None]
    next_weights[0] = 0.0
    inputs = rng.normal(0.0, 0.1, (*prefix, DIMENSIONS[0])).astype(np.float32)
    ema_prediction = rng.normal(0.0, 0.15, (*prefix, 2)).astype(np.float32)
    structural_labels = np.stack(
        (
            rng.integers(0, 12, size=prefix),
            rng.integers(0, 3, size=prefix),
            rng.integers(0, 4, size=prefix),
        ),
        axis=-1,
    ).astype(np.int32)
    batch = {
        "labels": labels,
        "fixed_target": fixed_target,
        "main_weights": row_weights,
        "structural_labels": structural_labels,
        "next_weights": next_weights,
        "next_rows": rng.integers(0, 8, size=prefix).astype(np.float32),
        "next_up": rng.integers(0, 2, size=prefix).astype(np.float32),
    }
    return inputs, batch, ema_prediction


def _flatten_replica_axis(np: Any, value: Any) -> Any:
    """Concatenate eight local batches into the matching global batch."""
    return value.reshape((GLOBAL_BATCH, *value.shape[2:]))


def _gradient_error(jax: Any, np: Any, reference: Any, observed: Any) -> dict[str, float]:
    """Summarize a pytree difference after synchronizing values to the host."""
    squared_error = 0.0
    squared_reference = 0.0
    max_absolute = 0.0
    for reference_leaf, observed_leaf in zip(
        jax.tree_util.tree_leaves(reference),
        jax.tree_util.tree_leaves(observed),
        strict=True,
    ):
        reference_array = np.asarray(jax.device_get(reference_leaf), dtype=np.float64)
        observed_array = np.asarray(jax.device_get(observed_leaf), dtype=np.float64)
        difference = observed_array - reference_array
        squared_error += float(np.sum(difference**2))
        squared_reference += float(np.sum(reference_array**2))
        max_absolute = max(max_absolute, float(np.max(np.abs(difference))))
    return {
        "relative_l2": float(np.sqrt(squared_error) / max(np.sqrt(squared_reference), 1e-12)),
        "max_absolute": max_absolute,
    }


def run_device_worker() -> dict[str, Any]:
    """Check full-objective sharded gradients and profile synthetic TPU updates."""
    global PHASE
    record_phase("framework_import", "device_worker")
    import jax
    import jax.numpy as jnp
    import numpy as np

    prepare_embedded_support()
    from combo_loss_reference import component_sums, loss_from_sums
    from jax import lax

    source_root = Path(os.environ["WNN_E463_SOURCE_ROOT"])
    run = source_root / "artifacts/runs/c20261003_combo_s3_f0"
    npz = run / "model_ep3.jax.npz"
    sys.path.insert(0, str(source_root / "src"))
    from wnn_jax import bridge, rnn

    record_phase("actual_tpu_inventory", "device_worker")
    local_devices = list(jax.local_devices())
    all_devices = list(jax.devices())
    fallback_hbm_capacity, tpu_info = tpu_hbm_capacity(jax)
    records = device_records(all_devices, fallback_hbm_capacity)
    write_json(
        OUTPUT / "tpu_inventory.json",
        {
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "local_device_count": len(local_devices),
            "global_device_count": len(all_devices),
            "tpu_hardware_info": tpu_info,
            "devices": records,
        },
    )
    if jax.default_backend() != "tpu" or len(local_devices) != 8 or len(all_devices) != 8:
        raise RuntimeError(
            "requested TpuV5E8 must expose exactly eight local/global TPU devices; "
            f"backend={jax.default_backend()}, local={len(local_devices)}, global={len(all_devices)}"
        )
    if any(device.platform != "tpu" for device in all_devices):
        raise RuntimeError("JAX exposed a non-TPU device; CPU fallback is not accepted")
    for device_record in records:
        current = device_record["hbm_current_bytes"]
        limit = device_record["hbm_capacity_bytes"]
        if current is None or limit is None:
            raise RuntimeError(
                f"TPU {device_record['id']} did not report measurable HBM capacity/current use"
            )
        if limit - current < int(limit * 0.20):
            raise MemoryError(f"TPU {device_record['id']} lacks 20% free HBM before the test")
    write_json(OUTPUT / "tpu_hbm_before.json", records)

    record_phase("checkpoint_hash_and_load", "device_worker")
    input_identity = verify_inputs(source_root)
    params = rnn.load_npz_state(npz, *DIMENSIONS)
    optimizer = rnn.adamw_init(params)

    def make_fullbatch_shard(value: Any) -> Any:
        return jnp.asarray(_flatten_replica_axis(np, value))

    def make_replicated(value: Any) -> Any:
        host_value = np.asarray(jax.device_get(value))
        return jnp.asarray(np.repeat(host_value[None, ...], len(local_devices), axis=0))

    def mapped_update_one_replica(
        model_params: Any,
        opt_state: Any,
        sequence: Any,
        batch: dict[str, Any],
        ema_targets: Any,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        local_denominators = _combo_denominators(jnp, batch)
        global_denominators = lax.psum(local_denominators, "tpu_replicas")

        def objective(current: Any) -> tuple[Any, tuple[Any, Any, Any]]:
            prediction, next_state = rnn.forward_chunk(current, sequence, precision="fp32")
            numerators, _ = _combo_sufficient_statistics(jax, jnp, prediction, batch, ema_targets)
            local_loss = _combo_loss_from_global_sums(jnp, numerators, global_denominators)
            finite = jnp.all(jnp.isfinite(prediction)) & jnp.all(jnp.isfinite(next_state))
            return local_loss, (finite, next_state, numerators)

        (local_loss, (finite_forward, next_state, local_numerators)), local_gradients = (
            jax.value_and_grad(objective, has_aux=True)(model_params)
        )
        global_gradients = jax.tree_util.tree_map(
            lambda gradient: lax.psum(gradient, "tpu_replicas"), local_gradients
        )
        global_loss = lax.psum(local_loss, "tpu_replicas")
        global_numerators = lax.psum(local_numerators, "tpu_replicas")
        updated, next_optimizer = rnn.adamw_update(
            model_params,
            global_gradients,
            opt_state,
            learning_rate=1e-4,
            weight_decay=0.01,
        )
        finite_gradients = jnp.asarray(True)
        for gradient in jax.tree_util.tree_leaves(global_gradients):
            finite_gradients = finite_gradients & jnp.all(jnp.isfinite(gradient))
        finite_updated = jnp.asarray(True)
        for parameter in jax.tree_util.tree_leaves(updated):
            finite_updated = finite_updated & jnp.all(jnp.isfinite(parameter))
        return (
            updated,
            next_optimizer,
            global_loss,
            global_gradients,
            global_numerators,
            global_denominators,
            finite_forward & finite_updated,
            finite_gradients & jnp.all(jnp.isfinite(next_state)),
        )

    record_phase("synthetic_unequal_mass_fixture", "device_worker")
    full_inputs, full_batch_np, full_ema_np = _synthetic_sharded_batch(np, CHUNK_ROWS)
    if float(np.sum(full_batch_np["next_weights"][0])) != 0.0:
        raise AssertionError("synthetic negative-control replica must have zero next mass")
    if float(np.sum(full_batch_np["next_weights"][1:])) <= 0.0:
        raise AssertionError("synthetic fixture must have positive global next mass")
    if np.array_equal(full_inputs[0], full_inputs[1]):
        raise AssertionError("synthetic replica inputs must be distinct")
    per_replica_denominators = np.stack(
        [
            _combo_denominators(
                np,
                {name: value[replica_index] for name, value in full_batch_np.items()},
            )
            for replica_index in range(8)
        ]
    )
    row_masses = per_replica_denominators[:, 1]
    if np.any(row_masses <= 0.0) or float(np.max(row_masses) / np.min(row_masses)) < 10.0:
        raise AssertionError(
            f"synthetic replica main-weight masses are not sufficiently unequal: {row_masses}"
        )
    fixture_denominators = _combo_denominators(
        np,
        {
            name: value.reshape((GLOBAL_BATCH, *value.shape[2:]))
            for name, value in full_batch_np.items()
        },
    )
    if not np.isfinite(fixture_denominators).all() or np.any(fixture_denominators <= 1e-8):
        raise AssertionError(
            f"synthetic fixture has invalid global normalization masses: {fixture_denominators}"
        )
    shard_params = jax.tree_util.tree_map(make_replicated, params)
    shard_optimizer = jax.tree_util.tree_map(make_replicated, optimizer)
    mapped_update = jax.pmap(
        mapped_update_one_replica,
        axis_name="tpu_replicas",
        in_axes=(0, 0, 0, 0, 0),
        devices=local_devices,
    )

    record_phase("global_batch_reference_gradient", "device_worker")
    parity_rows = 128
    parity_inputs = full_inputs[:, :, :parity_rows]
    parity_batch_np = {name: value[:, :, :parity_rows] for name, value in full_batch_np.items()}
    parity_ema_np = full_ema_np[:, :, :parity_rows]
    full_inputs_ref = make_fullbatch_shard(parity_inputs)
    full_batch_ref = {name: make_fullbatch_shard(value) for name, value in parity_batch_np.items()}
    full_ema_ref = make_fullbatch_shard(parity_ema_np)

    def full_batch_objective(current: Any) -> tuple[Any, tuple[Any, Any]]:
        prediction, next_state = rnn.forward_chunk(current, full_inputs_ref, precision="fp32")
        loss = bridge.combo_sbd_loss(prediction, full_batch_ref, full_ema_ref)
        return loss, (prediction, next_state)

    reference_value_and_grad = jax.jit(jax.value_and_grad(full_batch_objective, has_aux=True))
    (reference_loss, (reference_prediction, reference_state)), reference_gradients = (
        reference_value_and_grad(params)
    )
    jax.block_until_ready(
        (reference_loss, reference_prediction, reference_state, reference_gradients)
    )
    check_finite(jnp, reference_prediction, "full-batch reference prediction")
    check_finite(jnp, reference_state, "full-batch reference state")
    for leaf in jax.tree_util.tree_leaves(reference_gradients):
        check_finite(jnp, leaf, "full-batch reference gradient")

    flat_batch_np = {
        name: _flatten_replica_axis(np, value) for name, value in parity_batch_np.items()
    }
    flat_ema_np = _flatten_replica_axis(np, parity_ema_np)
    independent_sums = component_sums(
        np.asarray(jax.device_get(reference_prediction)),
        flat_batch_np,
        flat_ema_np,
    )
    independent_loss = loss_from_sums(independent_sums)
    source_loss = float(np.asarray(jax.device_get(reference_loss)))
    if not np.isclose(source_loss, independent_loss, rtol=1e-4, atol=1e-5):
        raise AssertionError(
            f"frozen JAX combo loss differs from independent NumPy equation: "
            f"source={source_loss}, numpy={independent_loss}"
        )

    record_phase("unequal_mass_distributed_gradient_parity", "device_worker")
    parity_batch = {name: jnp.asarray(value) for name, value in parity_batch_np.items()}
    parity_ema = jnp.asarray(parity_ema_np)
    parity_result = mapped_update(
        shard_params,
        shard_optimizer,
        jnp.asarray(parity_inputs),
        parity_batch,
        parity_ema,
    )
    jax.block_until_ready(parity_result)
    (
        _,
        _,
        distributed_losses,
        distributed_gradients,
        distributed_numerators,
        distributed_denominators,
        finite_forward,
        finite_gradients,
    ) = parity_result
    distributed_loss = float(np.asarray(jax.device_get(distributed_losses[0])))
    if not np.isclose(distributed_loss, source_loss, rtol=1e-4, atol=1e-5):
        raise AssertionError(
            f"global distributed loss differs from full batch: "
            f"distributed={distributed_loss}, full={source_loss}"
        )
    if not bool(jnp.all(finite_forward)) or not bool(jnp.all(finite_gradients)):
        raise FloatingPointError("unequal-mass distributed update produced nonfinite tensors")
    observed_gradients = jax.tree_util.tree_map(lambda value: value[0], distributed_gradients)
    gradient_metrics = _gradient_error(jax, np, reference_gradients, observed_gradients)
    for reference_leaf, observed_leaf in zip(
        jax.tree_util.tree_leaves(reference_gradients),
        jax.tree_util.tree_leaves(observed_gradients),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(jax.device_get(observed_leaf)),
            np.asarray(jax.device_get(reference_leaf)),
            rtol=3e-4,
            atol=3e-5,
        )
    observed_sums = np.asarray(jax.device_get(distributed_numerators[0]), dtype=np.float64)
    expected_sums = np.asarray(
        [
            independent_sums["primary_num"],
            independent_sums["consistency_num"],
            independent_sums["structural_num"],
            independent_sums["next_num"],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(observed_sums, expected_sums, rtol=1e-4, atol=1e-5)
    observed_denominators = np.asarray(
        jax.device_get(distributed_denominators[0]), dtype=np.float64
    )
    expected_denominators = np.asarray(
        [
            independent_sums["primary_den"],
            independent_sums["row_den"],
            independent_sums["next_den"],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(observed_denominators, expected_denominators, rtol=1e-5, atol=1e-5)

    record_phase("locally_normalized_pmean_negative_control", "device_worker")

    def negative_control_one_replica(
        model_params: Any,
        sequence: Any,
        batch: dict[str, Any],
        ema_targets: Any,
    ) -> tuple[Any, Any]:
        def local_objective(current: Any) -> Any:
            prediction, _ = rnn.forward_chunk(current, sequence, precision="fp32")
            return bridge.combo_sbd_loss(prediction, batch, ema_targets)

        local_loss, local_gradients = jax.value_and_grad(local_objective)(model_params)
        mean_gradients = jax.tree_util.tree_map(
            lambda gradient: lax.pmean(gradient, "tpu_replicas"), local_gradients
        )
        return lax.pmean(local_loss, "tpu_replicas"), mean_gradients

    negative_control = jax.pmap(
        negative_control_one_replica,
        axis_name="tpu_replicas",
        in_axes=(0, 0, 0, 0),
        devices=local_devices,
    )
    bad_losses, bad_gradients = negative_control(
        shard_params,
        jnp.asarray(parity_inputs),
        parity_batch,
        parity_ema,
    )
    jax.block_until_ready((bad_losses, bad_gradients))
    bad_observed = jax.tree_util.tree_map(lambda value: value[0], bad_gradients)
    bad_gradient_metrics = _gradient_error(jax, np, reference_gradients, bad_observed)
    if bad_gradient_metrics["relative_l2"] < 1e-3:
        raise AssertionError(
            "unequal-mass negative control failed to distinguish local pmean "
            f"from the global-batch gradient: {bad_gradient_metrics}"
        )
    write_json(
        OUTPUT / "tpu_objective_reduction_parity.json",
        {
            "status": "passed",
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "parity_global_batch": GLOBAL_BATCH,
            "parity_local_batch_per_tpu": LOCAL_BATCH,
            "parity_chunk_rows": parity_rows,
            "synthetic_seed": 20261007,
            "fixed_ema_predictions_synthetic": True,
            "raw_training_data_consumed": False,
            "source_jax_loss": source_loss,
            "independent_numpy_loss": independent_loss,
            "distributed_global_loss": distributed_loss,
            "independent_component_sums": independent_sums,
            "per_replica_component_denominators": per_replica_denominators.tolist(),
            "distributed_component_numerators": observed_sums.tolist(),
            "distributed_component_denominators": observed_denominators.tolist(),
            "distributed_gradient_vs_full_batch": gradient_metrics,
            "invalid_pmean_gradient_vs_full_batch": bad_gradient_metrics,
            "negative_control_losses": np.asarray(jax.device_get(bad_losses)).tolist(),
            "pass_tolerances": {
                "loss_rtol": 1e-4,
                "loss_atol": 1e-5,
                "gradient_rtol": 3e-4,
                "gradient_atol": 3e-5,
            },
        },
    )

    record_phase("fullshape_synthetic_update_profile", "device_worker")
    fullshape_inputs = jnp.asarray(full_inputs)
    fullshape_batch = {name: jnp.asarray(value) for name, value in full_batch_np.items()}
    fullshape_ema = jnp.asarray(full_ema_np)
    profile_params = jax.tree_util.tree_map(make_replicated, params)
    profile_optimizer = jax.tree_util.tree_map(make_replicated, optimizer)
    warmup_started = time.monotonic()
    profile_state = mapped_update(
        profile_params,
        profile_optimizer,
        fullshape_inputs,
        fullshape_batch,
        fullshape_ema,
    )
    jax.block_until_ready(profile_state)
    warmup_seconds = time.monotonic() - warmup_started
    profile_params, profile_optimizer = profile_state[0], profile_state[1]
    fullshape_times = []
    for _ in range(3):
        update_started = time.monotonic()
        profile_state = mapped_update(
            profile_params,
            profile_optimizer,
            fullshape_inputs,
            fullshape_batch,
            fullshape_ema,
        )
        jax.block_until_ready(profile_state)
        fullshape_times.append(time.monotonic() - update_started)
        profile_params, profile_optimizer = profile_state[0], profile_state[1]
    step_counts = np.asarray(jax.device_get(profile_optimizer.count)).tolist()
    if step_counts != [4] * len(local_devices):
        raise AssertionError(f"synthetic profile should perform warmup+3 updates: {step_counts}")
    final_profile = profile_state
    if not bool(jnp.all(final_profile[6])) or not bool(jnp.all(final_profile[7])):
        raise FloatingPointError("full-shape synthetic profile produced nonfinite tensors")
    check_finite(jnp, final_profile[2], "full-shape loss")
    check_finite(jnp, final_profile[4], "full-shape component numerators")
    check_finite(jnp, final_profile[5], "full-shape component denominators")
    for leaf in jax.tree_util.tree_leaves(final_profile[0]):
        check_finite(jnp, leaf, "full-shape updated parameter")
    for leaf in jax.tree_util.tree_leaves(final_profile[3]):
        check_finite(jnp, leaf, "full-shape global gradient")
    for leaf in jax.tree_util.tree_leaves(profile_optimizer):
        check_finite(jnp, leaf, "full-shape optimizer state")
    profile_records = {
        "status": "passed",
        "global_batch": GLOBAL_BATCH,
        "local_batch_per_tpu": LOCAL_BATCH,
        "chunk_rows": CHUNK_ROWS,
        "input_width": DIMENSIONS[0],
        "hidden_width": DIMENSIONS[1],
        "layers": DIMENSIONS[2],
        "output_width": DIMENSIONS[3],
        "synthetic_inputs_and_targets": True,
        "same_seeded_fixture_reused_for_profile_steps": True,
        "per_replica_component_denominators": per_replica_denominators.tolist(),
        "compile_warmup_update_seconds": warmup_seconds,
        "steady_state_update_seconds": fullshape_times,
        "optimizer_steps_per_replica_including_warmup": step_counts,
        "finite_loss": np.asarray(jax.device_get(final_profile[2])).tolist(),
        "finite_outputs_states_gradients_parameters": True,
        "checkpoint_write_or_export": False,
        "scientific_score_claim": None,
    }
    write_json(OUTPUT / "tpu_fullshape_profile.json", profile_records)

    record_phase("hbm_headroom_validation", "device_worker")
    final_records = device_records(all_devices, fallback_hbm_capacity)
    hbm_failures = []
    for device_record in final_records:
        peak = device_record["hbm_peak_bytes"]
        capacity = device_record["hbm_capacity_bytes"]
        if peak is None or capacity is None or peak > int(capacity * 0.80):
            hbm_failures.append(device_record["id"])
    write_json(OUTPUT / "tpu_hbm_after.json", final_records)
    if hbm_failures:
        raise MemoryError(
            f"TPU devices exceeded the 80% peak HBM cap or lacked stats: {hbm_failures}"
        )
    result = {
        "status": "passed",
        "classification": "actual TPUv5e eight-device unequal-mass combo-objective gradient diagnostic and full-shape synthetic profile",
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "device_count": len(local_devices),
        "input_identity": input_identity,
        "synthetic_seed": 20261007,
        "no_training_data_used": True,
        "no_checkpoint_saved_or_exported": True,
        "objective_reduction_parity": "tpu_objective_reduction_parity.json",
        "fullshape_profile": profile_records,
        "hbm_before": records,
        "hbm_after": final_records,
        "scientific_fit_or_score": None,
    }
    write_json(OUTPUT / "tpu_diagnostic_complete.json", result)
    return result


def worker_main() -> int:
    """Run device checks in a child so import/runtime crashes retain parent logs."""
    global PHASE, TERMINAL
    record_phase("device_worker_start", "device_worker")
    write_json(
        OUTPUT / "device_worker_start.json",
        {"time_epoch_seconds": time.time(), "python": sys.version, "packet": PACKET_ID},
    )
    try:
        result = run_device_worker()
    except BaseException as error:
        TERMINAL = {
            "status": "failed",
            "phase": PHASE,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps({"event": "tpu_probe_failed", **TERMINAL}), flush=True)
        write_json(OUTPUT / "device_worker_terminal.json", TERMINAL)
        return 1
    TERMINAL = {"status": "passed", "phase": "complete", "result": result}
    write_json(OUTPUT / "device_worker_result.json", result)
    write_json(OUTPUT / "device_worker_terminal.json", TERMINAL)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def stop_child_process_group(child: subprocess.Popen[bytes]) -> None:
    """Terminate the bounded worker and any XLA helper processes it started."""
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


def parent_main() -> int:
    """Run the TPU payload under an internal bound and always preserve terminal status."""
    global CHILD, PHASE, TERMINAL
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(
        OUTPUT / "startup.json",
        {
            "phase": "python_started_before_jax_import",
            "time_epoch_seconds": time.time(),
            "python": sys.version,
            "platform": platform.platform(),
            "packet": PACKET_ID,
            "requested_hardware": "TpuV5E8",
        },
    )
    record_phase("python_started_before_jax_import", "parent")
    signal.signal(
        signal.SIGALRM,
        lambda _signum, _frame: (_ for _ in ()).throw(
            TimeoutError(f"TPU payload exceeded {PARENT_TIMEOUT_SECONDS} seconds")
        ),
    )
    signal.alarm(PARENT_TIMEOUT_SECONDS)
    try:
        support_root = prepare_embedded_support()
        record_phase("host_admission", "parent")
        host = host_snapshot()
        write_json(OUTPUT / "host_start.json", host)
        if not host["headroom_passed"]:
            raise MemoryError("host RAM is below the 20%/2GiB preflight headroom requirement")
        write_json(OUTPUT / "input_mounts.json", input_mount_inventory())
        source_materialization = verify_direct_source_tree()
        write_json(OUTPUT / "source_materialization.json", source_materialization)
        source_root = Path(str(source_materialization["source_root"]))
        run = source_root / "artifacts/runs/c20261003_combo_s3_f0"
        if not run.is_dir():
            raise FileNotFoundError(f"verified E463 checkpoint directory is absent: {run}")
        input_identity = verify_inputs(source_root)
        write_json(OUTPUT / "e463_checkpoint_identity.json", input_identity)
        record_phase("launch_bounded_tpu_worker", "parent")
        environment = os.environ.copy()
        environment.update(
            {
                "JAX_PLATFORMS": "tpu",
                "PJRT_DEVICE": "TPU",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
                "WNN_E463_SOURCE_ROOT": str(source_root),
                "PYTHONPATH": os.pathsep.join(
                    path for path in (str(support_root), environment.get("PYTHONPATH", "")) if path
                ),
            }
        )
        log_path = OUTPUT / "tpu_worker.log"
        with log_path.open("wb") as log:
            CHILD = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--device-worker"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            write_json(
                OUTPUT / "worker_launch.json",
                {
                    "pid": CHILD.pid,
                    "started_epoch_seconds": time.time(),
                    "timeout_seconds": WORKER_TIMEOUT_SECONDS,
                },
            )
            try:
                return_code = CHILD.wait(timeout=WORKER_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as error:
                record_phase("device_worker_timeout", "parent")
                stop_child_process_group(CHILD)
                return_code = CHILD.returncode
                write_json(
                    OUTPUT / "worker_process.json",
                    {
                        "status": "timeout",
                        "timeout_seconds": WORKER_TIMEOUT_SECONDS,
                        "returncode": return_code,
                    },
                )
                raise TimeoutError(
                    f"TPU device worker exceeded {WORKER_TIMEOUT_SECONDS} seconds"
                ) from error
        write_json(OUTPUT / "worker_process.json", {"status": "exited", "returncode": return_code})
        if return_code != 0:
            child_terminal = OUTPUT / "device_worker_terminal.json"
            failure = json.loads(child_terminal.read_text()) if child_terminal.exists() else None
            raise RuntimeError(f"TPU device worker failed (exit={return_code}): {failure}")
        result = json.loads((OUTPUT / "device_worker_result.json").read_text())
        if result.get("status") != "passed":
            raise RuntimeError("TPU worker completed without a passing device result")
        record_phase("complete", "parent", worker_returncode=return_code)
        TERMINAL = {"status": "passed", "phase": PHASE, "worker_returncode": return_code}
        write_json(OUTPUT / "result.json", result)
        return 0
    except BaseException as error:
        TERMINAL = {
            "status": "failed",
            "phase": PHASE,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps({"event": "tpu_parent_failed", **TERMINAL}), flush=True)
        return 1
    finally:
        signal.alarm(0)
        if CHILD is not None and CHILD.poll() is None:
            stop_child_process_group(CHILD)
        TERMINAL["elapsed_seconds"] = time.monotonic() - STARTED
        TERMINAL["last_phase"] = PHASE
        TERMINAL["terminal_host_memory"] = host_snapshot()
        write_json(OUTPUT / "terminal.json", TERMINAL)
        write_json(OUTPUT / "complete.json", {"completed_epoch_seconds": time.time(), **TERMINAL})


if __name__ == "__main__":
    if sys.argv[1:] == ["--device-worker"]:
        raise SystemExit(worker_main())
    raise SystemExit(parent_main())
