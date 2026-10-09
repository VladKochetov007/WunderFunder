import hashlib
import json
import sys
from pathlib import Path

repo = Path("/tmp/wnn-h232-v7b-ep3-export-20261008")
base = repo / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_v7b_ep3_export_only_20261008"
sys.path.insert(0, str(base))
import build_export_packet as builder  # noqa: E402

packet = base / "packet_r4_metadata_slugfix"
old_packet = base / "packet_r3"
old = json.loads((base / "H232_V7B_EP3_EXPORT_R2_BUILD_RECEIPT.json").read_text())
new = json.loads((base / "H232_V7B_EP3_EXPORT_R3_BUILD_RECEIPT.json").read_text())
root_review = json.loads(Path("/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/H232_EP3_EXPORT_ROOT_INDEPENDENT_REVIEW_20261008_R1.json").read_text())
digest = lambda data: hashlib.sha256(data).hexdigest()
actual_files = {p.name: digest(p.read_bytes()) for p in packet.iterdir() if p.is_file()}
assert actual_files == new["packet_file_sha256"]
payload_zip, payload_hashes = builder._zip_payload(packet)
builder.verify_packet(packet, payload_zip, payload_hashes)
assert digest(payload_zip) == new["payload_zip_sha256"] == old["payload_zip_sha256"]
assert new["code_file_sha256"] == old["code_file_sha256"] == root_review["worker_sha256"] if "worker_sha256" not in root_review else new["payload_member_sha256"]["export_worker.py"] == root_review["worker_sha256"]
assert new["checkpoint_sha256"] == old["checkpoint_sha256"] == root_review["checkpoint_sha256"]
assert new["source_archive_sha256"] == old["source_archive_sha256"] == root_review["source_archive_sha256"]
assert len(payload_hashes) == 17
assert new["kernel_id"] == "vladkochetov07/h232-gru384-v7b-ep3-export-20261008-r3"
assert new["kernel_title"] == "H232 GRU384 V7B EP3 Export 20261008 R3"
differences = [
    name for name in old["packet_file_sha256"]
    if old["packet_file_sha256"][name] != new["packet_file_sha256"][name]
]
assert differences == ["kernel-metadata.json"]
assert all(old["payload_member_sha256"][name] == new["payload_member_sha256"][name] for name in payload_hashes)
print(json.dumps({
    "status": "PASS",
    "kernel_id": new["kernel_id"],
    "title": new["kernel_title"],
    "changed_file_only": differences,
    "packet_files_rehashed": len(actual_files),
    "embedded_payload_members_revalidated": len(payload_hashes),
    "code_file_sha256": new["code_file_sha256"],
    "payload_zip_sha256": new["payload_zip_sha256"],
    "checkpoint_sha256": new["checkpoint_sha256"],
    "source_archive_sha256": new["source_archive_sha256"]
}, sort_keys=True))
