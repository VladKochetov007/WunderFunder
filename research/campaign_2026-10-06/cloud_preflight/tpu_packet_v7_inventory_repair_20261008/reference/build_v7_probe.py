from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference"
PACKET = ROOT / "packet"
PACKET_ID = "vladkochetov07/e463-tpu-realdata-reload-v7-20261008"
PACKET_TITLE = "E463 TPU Real Data Reload V7 20261008"
BASE_PROBE_SHA256 = "3392ed77b9000c7bab9b236531c9a16734325152b95a60c955408df1b9bb02f2"
BASE_PACKET_MANIFEST_SHA256 = "a609e728b212065a53459b1c2630bced018cb80614d7eea55ebc5ecce3401448"
BASE_TERMINAL_RECEIPT_SHA256 = "96bbb2ab216973018e7e50e5a2c315ce845447d7cd1bcc4e715f5433f1445596"
ASSET_MANIFEST_SHA256 = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one {label} marker, found {count}")
    return text.replace(old, new, 1)


def replace_exact(text: str, old: str, new: str, label: str, count: int) -> str:
    actual = text.count(old)
    if actual != count:
        raise ValueError(f"expected {count} {label} markers, found {actual}")
    return text.replace(old, new)


INVENTORY_VALIDATOR = '''def validate_asset_inventory(
    asset_root: Path, entries: Any
) -> dict[str, dict[str, Any]]:
    """Validate the 23 payload files and the separate manifest in the flat mount."""
    if not isinstance(entries, list) or len(entries) != 23:
        raise ValueError(
            "fold-0 assets manifest must contain exactly 23 payload entries"
        )

    by_path: dict[str, dict[str, Any]] = {}
    filenames: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("fold-0 assets manifest entries must be objects")
        relative_value = item.get("path")
        filename = item.get("filename")
        digest = item.get("sha256")
        size = item.get("size")
        if not isinstance(relative_value, str):
            raise ValueError("fold-0 assets manifest path must be text")
        relative = _safe_relative_path(relative_value)
        if relative.as_posix() != relative_value:
            raise ValueError(f"non-canonical fold-0 asset path: {relative_value}")
        if (
            not isinstance(filename, str)
            or not filename
            or filename in {".", "..", "assets-manifest.json"}
            or PurePosixPath(filename).name != filename
            or "\\\\" in filename
        ):
            raise ValueError(f"unsafe fold-0 asset filename: {filename!r}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid fold-0 asset SHA-256 for {relative_value}")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"invalid fold-0 asset size for {relative_value}")
        if relative_value in by_path or filename in filenames:
            raise ValueError("fold-0 assets manifest has duplicate paths or filenames")
        by_path[relative_value] = item
        filenames.add(filename)

    missing_required = set(REQUIRED_ASSET_PATHS) - set(by_path)
    if missing_required:
        raise ValueError(
            "fold-0 assets inventory is missing required members: "
            + ", ".join(sorted(missing_required))
        )

    expected_names = filenames | {"assets-manifest.json"}
    actual_names: set[str] = set()
    for path in asset_root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("fold-0 assets mount must contain direct regular files only")
        actual_names.add(path.name)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"fold-0 assets mount names differ (missing={missing}, extra={extra})"
        )
    return by_path


'''


def build_probe() -> bytes:
    source_path = REFERENCE / "v6_probe.py"
    source_bytes = source_path.read_bytes()
    if sha256_bytes(source_bytes) != BASE_PROBE_SHA256:
        raise ValueError("frozen V6 probe bytes changed")

    text = source_bytes.decode("utf-8")
    text = replace_exact(
        text,
        'PACKET_ID = "e463-tpu-realdata-reload-v6-20261008"',
        f'PACKET_ID = "{PACKET_ID.rsplit("/", 1)[1]}"',
        "packet id",
        2,
    )
    text = replace_exact(
        text,
        'OUTPUT = Path("/kaggle/working/cloud_preflight_tpu_v6")',
        'OUTPUT = Path("/kaggle/working/cloud_preflight_tpu_v7")',
        "output directory",
        2,
    )
    text = replace_once(
        text,
        "from contextlib import suppress\nfrom pathlib import Path, PurePosixPath\n"
        "from collections.abc import Sequence\n",
        "from collections.abc import Sequence\nfrom contextlib import suppress\n"
        "from pathlib import Path, PurePosixPath\n",
        "stdlib import order",
    )
    for seed in ("s3", "s4"):
        text = replace_once(
            text,
            f'"artifacts/runs/c20260929_since_{seed}_f0/model_ep3.pt"',
            f'"artifacts/runs/c20260929_since_{seed}_f0/model.pt"',
            f"manifested {seed} checkpoint path",
        )
    text = replace_once(
        text,
        "def materialize_realdata_project(\n",
        INVENTORY_VALIDATOR + "def materialize_realdata_project(\n",
        "asset inventory validator insertion",
    )
    old_inventory = '''    entries = asset_manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 24:
        raise ValueError("fold-0 assets file inventory differs from the reviewed 24-file bundle")
    by_path = {str(item["path"]): item for item in entries}
    if len(by_path) != len(entries) or set(REQUIRED_ASSET_PATHS) - set(by_path):
        raise ValueError("fold-0 assets inventory has duplicate or missing required members")
'''
    text = replace_once(
        text,
        old_inventory,
        '    entries = asset_manifest.get("files")\n'
        "    by_path = validate_asset_inventory(asset_root, entries)\n",
        "fold-0 asset inventory gate",
    )
    return text.encode("utf-8")


def build() -> dict[str, object]:
    v6_manifest_path = REFERENCE / "v6_packet_manifest.json"
    v6_manifest_bytes = v6_manifest_path.read_bytes()
    if sha256_bytes(v6_manifest_bytes) != BASE_PACKET_MANIFEST_SHA256:
        raise ValueError("frozen V6 packet manifest bytes changed")
    v6_manifest = json.loads(v6_manifest_bytes)
    if v6_manifest.get("code_sha256") != BASE_PROBE_SHA256:
        raise ValueError("V6 manifest does not bind the frozen probe")

    terminal_path = REFERENCE / "v6_terminal_receipt.json"
    if sha256_file(terminal_path) != BASE_TERMINAL_RECEIPT_SHA256:
        raise ValueError("frozen V6 terminal receipt bytes changed")
    asset_manifest_path = REFERENCE / "fold0_assets_manifest.json"
    if sha256_file(asset_manifest_path) != ASSET_MANIFEST_SHA256:
        raise ValueError("frozen fold-0 assets manifest bytes changed")
    asset_manifest = json.loads(asset_manifest_path.read_bytes())
    if (
        asset_manifest.get("dataset") != "vladkochetov07/wnn-c20261004-fold0-assets"
        or len(asset_manifest.get("files", [])) != 23
    ):
        raise ValueError("reviewed fold-0 manifest is not the expected 23-payload bundle")

    code = build_probe()
    metadata = json.loads((REFERENCE / "v6_kernel_metadata.json").read_text(encoding="utf-8"))
    metadata.update({"id": PACKET_ID, "title": PACKET_TITLE, "code_file": "probe.py"})
    if metadata.get("enable_gpu") is not False or metadata.get("enable_tpu") is not True:
        raise ValueError("V6 provider hardware settings changed")
    metadata_bytes = json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n"

    PACKET.mkdir(parents=True, exist_ok=True)
    (PACKET / "probe.py").write_bytes(code)
    (PACKET / "kernel-metadata.json").write_bytes(metadata_bytes)
    bundle_path = ROOT / "packet.bundle.zip"
    with zipfile.ZipFile(
        bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, content in (("kernel-metadata.json", metadata_bytes), ("probe.py", code)):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            archive.writestr(member, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    manifest: dict[str, object] = dict(v6_manifest)
    manifest.update(
        {
            "packet_id": PACKET_ID,
            "packet_title": PACKET_TITLE,
            "packet_code_file": "probe.py",
            "code_bytes": len(code),
            "code_sha256": sha256_bytes(code),
            "metadata_sha256": sha256_bytes(metadata_bytes),
            "bundle_sha256": sha256_file(bundle_path),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "test_sha256": sha256_file(ROOT / "tests/test_v7_inventory_packet.py"),
            "readme_sha256": sha256_file(ROOT / "README.md"),
            "base_v6_packet": "vladkochetov07/e463-tpu-realdata-reload-v6-20261008/1",
            "base_v6_probe_sha256": BASE_PROBE_SHA256,
            "base_v6_packet_manifest_sha256": BASE_PACKET_MANIFEST_SHA256,
            "base_v6_terminal_receipt_sha256": BASE_TERMINAL_RECEIPT_SHA256,
            "assets_manifest_sha256": ASSET_MANIFEST_SHA256,
            "assets_manifest_payload_entries": 23,
            "assets_mount_expected_regular_files": 24,
            "inventory_repair": {
                "classification": "provider host-admission inventory correction",
            "change": "count the 23 manifest payload entries and verify the separate assets-manifest.json in the 24-file mount",
            "additional_consumer_path_correction": "the V6 required-path list used model_ep3.pt names absent from the pinned manifest; V7 stages the exact manifested model.pt paths without asserting an epoch label",
                "source_only": True,
                "no_scientific_recipe_change": True,
                "no_full_asset_hash_scan_added": True,
            },
            "provider_push_authorized": False,
            "source_only": True,
        }
    )
    (ROOT / "packet_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    result = build()
    print(
        json.dumps(
            {
                "packet_id": result["packet_id"],
                "code_bytes": result["code_bytes"],
                "code_sha256": result["code_sha256"],
                "metadata_sha256": result["metadata_sha256"],
                "bundle_sha256": result["bundle_sha256"],
                "provider_push_authorized": result["provider_push_authorized"],
            },
            indent=2,
            sort_keys=True,
        )
    )
