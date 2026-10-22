"""Create one fixed, source-bound two-member TL3 ablation without model fitting."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from tl3_source_archive import SOURCE_ARCHIVE, file_sha256, verify_source_archive  # noqa: E402

OUT = Path("/home/vlad/development/wunder/research_tmp/tl3-retained-pair-quota-20261022-r1")


def main() -> None:
    """Verify every source entry and change only the fixed roster and its normalization."""
    source_sha, receipt_sha, entries = verify_source_archive(SOURCE_ARCHIVE)
    OUT.mkdir(exist_ok=False)
    output = OUT / "submission.zip"
    with ZipFile(SOURCE_ARCHIVE) as original, ZipFile(output, "x") as target:
        assert original.testzip() is None
        assert set(original.namelist()) == set(entries)
        roster = json.loads(original.read("members.json"))
        assert [m["weights"] for m in roster] == [
            "weights_s5.npz",
            "weights_tl5.npz",
            "weights_tl6.npz",
        ]
        assert [m["coef"] for m in roster] == [0.34, 0.33, 0.33]
        revised = [dict(m, coef=0.5) for m in roster[1:]]
        for item in original.infolist():
            payload = original.read(item.filename)
            assert hashlib.sha256(payload).hexdigest() == entries[item.filename]["sha256"]
            if item.filename == "weights_s5.npz":
                continue
            if item.filename == "members.json":
                payload = (json.dumps(revised, indent=2) + "\n").encode()
            target.writestr(item, payload)
    assert output.stat().st_size < 20_000_000
    with ZipFile(output) as result:
        assert result.testzip() is None
        assert set(result.namelist()) == set(entries) - {"weights_s5.npz"}
        hashes = {name: hashlib.sha256(result.read(name)).hexdigest() for name in result.namelist()}
        for name, digest in hashes.items():
            if name != "members.json":
                assert digest == entries[name]["sha256"], name
    receipt = {
        "status": "BUILT_NOT_YET_VALIDATED_OR_SUBMITTED",
        "zip": str(output),
        "zip_sha256": file_sha256(output),
        "bytes": output.stat().st_size,
        "source_zip_sha256": source_sha,
        "source_receipt_sha256": receipt_sha,
        "members": revised,
        "entry_sha256": hashes,
        "new_fits": 0,
        "build_script_sha256": file_sha256(Path(__file__)),
    }
    (OUT / "BUILD_RECEIPT_R1.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
