import ast
import base64
import hashlib
import json
import zipfile
from pathlib import Path

packet = Path("/tmp/wnn-h232-v7b-ep3-export-20261008/research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_v7b_ep3_export_only_20261008/packet_r3")
root = packet.parent
receipt = json.loads((root / "H232_V7B_EP3_EXPORT_R2_BUILD_RECEIPT.json").read_text())
independent = json.loads(Path("/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/H232_EP3_EXPORT_ROOT_INDEPENDENT_REVIEW_20261008_R1.json").read_text())
sha = lambda b: hashlib.sha256(b).hexdigest()
assert receipt["kernel_id"] == "vladkochetov07/h232-gru384-v7b-ep3-export-20261008-r2"
assert receipt["packet_id"] == "E529_H232_V7B_EP3_FP32_EXPORT_ONLY_20261008_R2"
for name, expected in receipt["packet_file_sha256"].items():
    assert sha((packet / name).read_bytes()) == expected, name
assert len(receipt["payload_member_sha256"]) == 17
code = (packet / "h232_v7b_ep3_export_kernel_r2.py").read_text()
assignments = {}
for node in ast.parse(code).body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"PAYLOAD_ZIP_SHA256", "PAYLOAD_FILE_SHA256", "PAYLOAD_B64"}:
                assignments[target.id] = ast.literal_eval(node.value)
embedded = base64.b64decode("".join(assignments["PAYLOAD_B64"]), validate=True)
assert sha(embedded) == assignments["PAYLOAD_ZIP_SHA256"] == receipt["payload_zip_sha256"]
assert assignments["PAYLOAD_FILE_SHA256"] == receipt["payload_member_sha256"]
with zipfile.ZipFile(packet / "payload.zip", "r") if (packet / "payload.zip").exists() else zipfile.ZipFile(__import__("io").BytesIO(embedded)) as archive:
    assert set(archive.namelist()) == set(receipt["payload_member_sha256"])
    for name, expected in receipt["payload_member_sha256"].items():
        assert sha(archive.read(name)) == expected, name
assert receipt["payload_member_sha256"]["export_worker.py"] == independent["worker_sha256"]
assert receipt["checkpoint_sha256"] == independent["checkpoint_sha256"]
assert receipt["source_archive_sha256"] == independent["source_archive_sha256"]
assert receipt["payload_zip_bytes"] == len(embedded)
print(json.dumps({
    "status": "PASS",
    "packet_file_hashes": len(receipt["packet_file_sha256"]),
    "embedded_payload_hashes": len(receipt["payload_member_sha256"]),
    "code_sha256": receipt["code_file_sha256"],
    "payload_zip_sha256": receipt["payload_zip_sha256"],
    "checkpoint_sha256": receipt["checkpoint_sha256"],
    "source_archive_sha256": receipt["source_archive_sha256"],
    "root_reviewed_hash_match": True
}, sort_keys=True))
