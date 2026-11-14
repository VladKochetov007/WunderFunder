"""Build the frozen three-member replacement from two verified, preserved ZIPs."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    """Hash the small immutable submission artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Replace exactly the .34 fbc1k slot; preserve the .33/.33 tiedlead bytes."""
    base = ROOT / "submissions/tl3c_pr49_20261029/blend-dynamic_int8"
    receipt = json.loads((base / "receipt.json").read_text())
    student = ROOT / "submissions/local_20261114/kd90_s3_single.zip"
    if digest(base / "submission.zip") != receipt["artifact"]["sha256"]:
        raise ValueError("original TL3C artifact changed")
    if digest(student) != "a63ce606d7f7c040721950d553f74ed4499cec5ee359d0aa832260581a057541":
        raise ValueError("checked KD90 artifact changed")
    provenance_path = Path(
        "/home/vlad/development/wunder/research_tmp/chrono_root/artifacts/runs/l14_kd90_s3/provenance.json"
    )
    provenance = json.loads(provenance_path.read_text())
    with zipfile.ZipFile(student) as archive:
        spec = json.loads(archive.read("members.json"))
        if len(spec) != 1 or spec[0]["coef"] != 1:
            raise ValueError("expected checked single-member student")
        weights = archive.read(spec[0]["weights"])
    weight_file = "l14_kd90_s3.npz"
    replacement = {"run": "l14_kd90_s3", "coef": 0.34, "family": "event211", "weights": weight_file}
    target = OUT / "kd90_tiedlead_pair.zip"
    with (
        zipfile.ZipFile(base / "submission.zip") as src,
        zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as dst,
    ):
        contract = json.loads(src.read("members.json"))
        old = contract["members"][0]
        if old["run"] != "tl3c_fbc1k_s5" or old["coef"] != 0.34:
            raise ValueError("original slot changed")
        contract["members"][0] = replacement
        for name in sorted(src.namelist()):
            if name == old["weights"]:
                continue
            data = (
                (json.dumps(contract, indent=2) + "\n").encode()
                if name == "members.json"
                else src.read(name)
            )
            dst.writestr(zipfile.ZipInfo(name, (2026, 11, 14, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)
        dst.writestr(
            zipfile.ZipInfo(weight_file, (2026, 11, 14, 0, 0, 0)), weights, zipfile.ZIP_DEFLATED
        )
    if target.stat().st_size > 20_000_000:
        raise ValueError("oversized package")
    new_member = dict(replacement)
    new_member.pop("weights")
    new_member.update(
        weight_file=weight_file,
        weight_sha256=hashlib.sha256(weights).hexdigest(),
        checkpoint={"sha256": provenance["checkpoint_sha256"]["model_ep3.pt"]},
        provenance={"path": str(provenance_path), "sha256": digest(provenance_path)},
        teacher_ancestry=provenance["args"]["distill_provenance"],
        training_precision="FP32 model kernels, legacy BF16 Newton-Schulz",
    )
    result = {
        "schema": "tl3c-submission-candidate-v1",
        "candidate": "fixed_kd90_tiedlead_pair",
        "mode": "dynamic_int8",
        "artifact": {"path": target.name, "bytes": target.stat().st_size, "sha256": digest(target)},
        "members": [new_member, *receipt["members"][1:]],
        "source_artifacts": {
            "tl3c": receipt["artifact"],
            "student": {"path": str(student), "sha256": digest(student)},
        },
        "training_reference_commit": None,
        "scope": "diagnostic candidate, exposed full-valid; no release approval or public gain claim",
    }
    (OUT / "replacement_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["artifact"]))


if __name__ == "__main__":
    main()
