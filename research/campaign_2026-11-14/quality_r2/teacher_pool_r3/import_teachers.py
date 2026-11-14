"""Create explicit portable views of unchanged team teacher tensors and clean ancestry."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SCRATCH = Path("/home/vlad/development/wunder/research_tmp")
RUNS = SCRATCH / "chrono_root/artifacts/runs"
SOURCE = SCRATCH / "q14_strict_source"
ARCHIVE = SCRATCH / "q14_team_artifacts_r3/tabdata-pr49-checkpoints-20261029-r1.zip"
ARCHIVE_SHA = "02899081e9576bd86a0197cb3eae5bd1ccc8a4b1c26e1669cd1e10a62efd31fa"
SPLIT = ROOT / "research/campaign_2026-10-29/teacher_track_r1/chronological_primary.json"
NAMES = {
    "tl3c_fbc1k_s5": "q14t_fbc_s5",
    "tl3c_fttall_tlead_s5": "q14t_tlead_s5",
    "tl3c_fttall_tlead_s6": "q14t_tlead_s6",
}


def digest(data: bytes) -> str:
    """Hash immutable input or newly serialized view bytes."""
    return hashlib.sha256(data).hexdigest()


def audit_archive(archive: zipfile.ZipFile) -> tuple[dict, dict]:
    """Verify source-qualified membership equivalence and the complete released ancestry."""
    old = json.loads(archive.read("shared/chronological_primary.json"))
    new = json.loads(SPLIT.read_text())
    assert {k: v for k, v in old.items() if k != "sources"} == {
        k: v for k, v in new.items() if k != "sources"
    }
    for source in ("train", "valid"):
        a, b = (
            old["sources"][f"datasets/{source}.parquet"],
            new["sources"][f"datasets/{source}.parquet"],
        )
        assert all(a[key] == b[key] for key in ("bytes", "footer_bytes", "footer_sha256"))
        a, b = (
            old["sources"][f"cache/{source}_index.npz"],
            new["sources"][f"cache/{source}_index.npz"],
        )
        assert all(a[key] == b[key] for key in ("bytes", "sha256", "index_sha256"))
    provenance = {
        name.split("/")[0]: json.loads(archive.read(name))
        for name in archive.namelist()
        if name.endswith("/metadata/provenance.json")
    }
    for name, record in provenance.items():
        args = record["args"]
        assert args["train_source"] == "train" and args["include_heldout_valid"] is False
        assert record["split_manifest_sha256"] == digest(
            archive.read("shared/chronological_primary.json")
        )
        for source, expected in new["fit"].items():
            actual = record["fit_row_groups"][source]
            assert len(actual) == len(set(actual)) == len(expected) and set(actual) == set(expected)
        assert set(record["evaluation_row_groups"]) == set(new["evaluation"]["valid"])
        parents = [args.get("init", "").partition("@")[0]]
        parents += [value.partition("*")[0] for value in args.get("distill", "").split(",")]
        assert all(parent in provenance for parent in parents if parent), name
        for binding in args.get("distill_provenance", []):
            parent = binding["run"]
            filename = binding["checkpoint_file"]
            parent_bytes = archive.read(f"{parent}/checkpoints/{filename}")
            assert digest(parent_bytes) == binding["checkpoint_sha256"]
    return new, provenance


def import_view(
    archive: zipfile.ZipFile, original: str, alias: str, original_provenance: dict
) -> dict:
    """Adapt metadata only, preserving the released model tensors and original identity."""
    raw = archive.read(f"{original}/checkpoints/model_ep3.pt")
    original_hash = digest(raw)
    assert original_hash == original_provenance["checkpoint_sha256"]["model_ep3.pt"]
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 3 and checkpoint["args"]["name"] == original
    for path, expected in (
        original_provenance.get("input_feature_schema", {}).get("source_sha256", {}).items()
    ):
        assert digest((SOURCE / path).read_bytes()) == expected, path
    tensors = {key: value.clone() for key, value in checkpoint["model"].items()}
    evidence = {
        "kind": "portable_metadata_view_no_training_or_tensor_change",
        "archive_sha256": ARCHIVE_SHA,
        "original_run": original,
        "original_checkpoint_sha256": original_hash,
        "original_provenance_sha256": digest(archive.read(f"{original}/metadata/provenance.json")),
        "original_split_sha256": original_provenance["split_manifest_sha256"],
        "local_split_sha256": digest(SPLIT.read_bytes()),
        "equivalence": "All non-source split fields identical; parquet footer/size and index content identities equal. Source timestamps and cache-identity serialization differ.",
        "new_fits": 0,
    }
    lineage = copy.deepcopy(original_provenance)
    for record in (checkpoint, lineage, checkpoint["provenance"]):
        if "args" in record:
            record["args"].update(
                name=alias,
                split_manifest=str(SPLIT),
                split_manifest_sha256=evidence["local_split_sha256"],
            )
        if "split_manifest_sha256" in record:
            record["split_manifest_sha256"] = evidence["local_split_sha256"]
        record["portable_import"] = evidence
    folder = RUNS / alias
    folder.mkdir(exist_ok=False)
    torch.save(checkpoint, folder / "model_ep3.pt")
    restored = torch.load(folder / "model_ep3.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(value, restored["model"][key]) for key, value in tensors.items())
    lineage["checkpoint_sha256"] = {"model_ep3.pt": digest((folder / "model_ep3.pt").read_bytes())}
    (folder / "provenance.json").write_text(json.dumps(lineage, indent=2) + "\n")
    return {"run": alias, "checkpoint_sha256": lineage["checkpoint_sha256"], "import": evidence}


def main() -> None:
    """Preserve producer bytes and emit a hash-bound import receipt for three teachers."""
    assert digest(ARCHIVE.read_bytes()) == ARCHIVE_SHA
    assert all(not (RUNS / alias).exists() for alias in NAMES.values())
    with zipfile.ZipFile(ARCHIVE) as archive:
        _, provenance = audit_archive(archive)
        records = [
            import_view(archive, name, alias, provenance[name]) for name, alias in NAMES.items()
        ]
    result = {
        "status": "imported_unchanged_tensors",
        "ancestry_runs_audited": len(provenance),
        "teachers": records,
    }
    (OUT / "IMPORT_R1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
