import json
import re
from pathlib import Path

root = Path("/tmp/wnn-h232-v7b-ep3-export-20261008/research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_v7b_ep3_export_only_20261008")
old = json.loads((root / "H232_V7B_EP3_EXPORT_R2_BUILD_RECEIPT.json").read_text())
new = json.loads((root / "H232_V7B_EP3_EXPORT_R3_BUILD_RECEIPT.json").read_text())
new_packet = root / "packet_r4_metadata_slugfix"
old_packet = root / "packet_r3"
assert new["kernel_id"] == "vladkochetov07/h232-gru384-v7b-ep3-export-20261008-r3"
assert new["kernel_title"] == "H232 GRU384 V7B EP3 Export 20261008 R3"
assert new["payload_zip_sha256"] == old["payload_zip_sha256"]
assert new["code_file_sha256"] == old["code_file_sha256"]
assert new["source_archive_sha256"] == old["source_archive_sha256"]
assert new["checkpoint_sha256"] == old["checkpoint_sha256"]
assert set(new["packet_file_sha256"]) == set(old["packet_file_sha256"])
changed = []
for name, expected in old["packet_file_sha256"].items():
    actual = new["packet_file_sha256"][name]
    if actual != expected:
        changed.append(name)
        assert name == "kernel-metadata.json", (name, expected, actual)
assert changed == ["kernel-metadata.json"]
for name in old["payload_member_sha256"]:
    assert new["payload_member_sha256"][name] == old["payload_member_sha256"][name], name
old_meta = json.loads((old_packet / "kernel-metadata.json").read_text())
new_meta = json.loads((new_packet / "kernel-metadata.json").read_text())
assert old_meta["id"] != new_meta["id"] and old_meta["title"] != new_meta["title"]
assert new_meta["id"].split("/", 1)[1] == "-".join(
    re.findall(r"[a-z0-9]+", new_meta["title"].casefold())
)
print(json.dumps({
    "status": "PASS",
    "only_changed_packet_file": changed,
    "unchanged_payload_members": len(new["payload_member_sha256"]),
    "unchanged_code_sha256": new["code_file_sha256"],
    "unchanged_payload_zip_sha256": new["payload_zip_sha256"],
    "unchanged_checkpoint_sha256": new["checkpoint_sha256"],
    "canonical_id": new["kernel_id"],
    "title": new["kernel_title"]
}, sort_keys=True))
