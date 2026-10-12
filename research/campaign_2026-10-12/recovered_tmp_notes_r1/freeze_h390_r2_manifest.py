import hashlib
import json
from pathlib import Path

root = Path.cwd()
packet = root / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/h390_all_data_homogeneous_r2"
source = packet / "child_source_v1"
teachers = {
    "vt_f1_muonall_h384_e3": "ca9f7c32910e5d71acddf78a474c06eee324b54d781ad9c736cd1f9a4a83deae",
    "vt_f1_muonall_h384_e3_s4": "e11ca9b3bfb9fc302f199514eb075960933e8fd27dfa49d58a31b76d5f5011f6",
    "vt_f1_muonall_h384_e3_s5": "45555945babdc6af0d56e1b5ab6a68585383bf32a73335fe8b92d84af7bdf03d",
}
def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
for run, digest in teachers.items():
    receipt = packet / "teacher_receipts_r2" / f"{run}.provenance.json"
    if sha(receipt) != digest:
        raise SystemExit(f"receipt pin mismatch: {run}")
sources = {
    path.relative_to(source).as_posix(): sha(path)
    for folder in (source / "scripts", source / "src/wnn")
    for path in folder.rglob("*.py")
    if path.is_file()
}
manifest = {
    "schema": "wnn.h390-source-snapshot.v2",
    "parent_archive_sha256": "35761b8cf6bc5bcf5079b134bfcce81e48b5fd83136630c51b89d48f8759caf1",
    "parent_manifest_sha256": "382beb68d556dd63215a9f7467bda09a23626ec3e1a2195739cffcab6819d9cc",
    "teacher_provenance_sha256": teachers,
    "sources": sources,
}
(source / "SOURCE_MANIFEST_H390.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
files = [
    "README.md",
    "h390_reference_r1.tar.gz",
    "parent_receipts_r1/vt_all_fin_mu256p0e4_s3.provenance.json",
    "parent_receipts_r1/vt_all_fin_mu256p0e4_s4.provenance.json",
    "teacher_receipts_r2/vt_f1_muonall_h384_e3.provenance.json",
    "teacher_receipts_r2/vt_f1_muonall_h384_e3_s4.provenance.json",
    "teacher_receipts_r2/vt_f1_muonall_h384_e3_s5.provenance.json",
    "child_source_v1/SOURCE_MANIFEST.json",
    "child_source_v1/SOURCE_MANIFEST_H390.json",
]
packet_manifest = {
    "schema": "wnn.h390-packet-manifest.v2",
    "files": {relative: sha(packet / relative) for relative in files},
}
(packet / "PACKET_MANIFEST_R2.json").write_text(json.dumps(packet_manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps({"source_files": len(sources), "packet_files": len(files), "manifest_sha256": sha(packet / "PACKET_MANIFEST_R2.json")}))
