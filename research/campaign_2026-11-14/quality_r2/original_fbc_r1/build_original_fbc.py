"""Preserve original TL3C bytes and isolate its event211 member for a zero-fit audit."""

import hashlib
import json
import zipfile
from pathlib import Path

root = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
base = root / "submissions/tl3c_pr49_20261029/blend-dynamic_int8"
out = root / "research/campaign_2026-11-14/quality_r2/original_fbc_r1"
out.mkdir(exist_ok=False)
receipt = json.loads((base / "receipt.json").read_text())
archive = base / "submission.zip"
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt["artifact"]["sha256"]
member = receipt["members"][0]
assert member["run"] == "tl3c_fbc1k_s5" and member["coef"] == 0.34
with zipfile.ZipFile(archive) as src:
    contract = json.loads(src.read("members.json"))
    contract["members"] = [dict(contract["members"][0], coef=1.0)]
    with zipfile.ZipFile(out / "original_fbc_only.zip", "x", zipfile.ZIP_DEFLATED) as dst:
        for name in sorted(src.namelist()):
            if name == "members.json":
                data = (json.dumps(contract, indent=2) + "\n").encode()
            elif name.endswith(".npz") and name != member["weight_file"]:
                continue
            else:
                data = src.read(name)
            dst.writestr(zipfile.ZipInfo(name, (2026, 11, 14, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)
receipt["parent_artifact"] = receipt["artifact"]
receipt["members"] = [dict(member, coef=1.0)]
receipt["candidate"] = "original_fbc_only_zero_fit_diagnostic"
receipt["created_at"] = None
receipt["creation_scope"] = (
    "2026-11-14 quality_r2; same original weights/native runtime; only member inventory and coefficient changed"
)
receipt["artifact"] = {
    "path": "original_fbc_only.zip",
    "bytes": (out / "original_fbc_only.zip").stat().st_size,
    "sha256": hashlib.sha256((out / "original_fbc_only.zip").read_bytes()).hexdigest(),
}
receipt.pop("files", None)
(out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
(out / "build_original_fbc.py").write_text(Path(__file__).read_text())
print(json.dumps(receipt["artifact"]))
