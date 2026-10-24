"""Create two fixed TL3/H390 exploratory ZIPs from hash-pinned source bytes."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "submissions/quota_20261024"
RECIPE_PATH = PACKAGE_DIR / "BUILD_RECIPE_R1.json"
AUTHORITY_PATH = (
    ROOT / "research/campaign_2026-10-24/resume4h_r1/QUOTA_SUBMISSION_AUTHORITY_R1.json"
)
BASE_ZIP = ROOT / "submissions/team_tl3_original_20261018.zip"
BASE_RECEIPT = (
    ROOT / "research/campaign_2026-10-20/submission_delivery_r1/tl3_member_adaptation/"
    "SOURCE_ARCHIVE_CONTENTS_R1.json"
)
E_ZIP = ROOT / "artifacts/team_models_20261013/E.zip"
E_S3 = ROOT / "artifacts/team_models_20261013/E/weights_0.npz"
E_S4 = ROOT / "artifacts/team_models_20261013/E/weights_1.npz"
S3_CANARY_ZIP = ROOT / "submissions/10-11-67/11_h390_student_s3_h388_teacher_s5_all_q8vnni.zip"
S3_PROVENANCE = ROOT / "artifacts/runs/vt_all_h390_ds_homogeneous_s3_r3/provenance.json"
S4_PROVENANCE = ROOT / "artifacts/runs/vt_all_h390_ds_homogeneous_s4_r3/provenance.json"

AUTHORITY_SHA256 = "56f2725ceaeb3d9ff9b375e9a73a84ddf5e5fd42d6368729903ee8f95c18eec1"
BASE_ZIP_SHA256 = "d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569"
BASE_RECEIPT_SHA256 = "73cd201b165424f4b7ec72fe4bf4d9d573f56d93c0501519ed3bc6a3ea11a2bd"
E_ZIP_SHA256 = "2aed2137c347344146e3481c83388d64bd5060b1c13bb66554b0bf9cc0027024"
S3_CANARY_ZIP_SHA256 = "d83723831277cf7327ab84ba781a2f2ebe883aca40cd1f7bc6a222e30be28284"
S3_SHA256 = "62b11349b4372beda7abbd35f7e006d2c967237070cbd5318c7a2f15bc0d1ec6"
S4_SHA256 = "9f019d54a147296037b1bb2c6ad8b15b87070bde0402ff5426ce9c4804f52eaf"
S3_PROVENANCE_SHA256 = "40dc970d2b30efd79f83447de0fdc22e66e6a2a48fd39ee814232e3c3ba77c4c"
S4_PROVENANCE_SHA256 = "3e3fd49545ebcfe151a406092f81c10802a5c65e37adf0fb984cb6979b6755eb"
MAX_ZIP_BYTES = 20_000_000

S3_NAME = "vt_all_h390_ds_homogeneous_s3_r3@ep5"
S4_NAME = "vt_all_h390_ds_homogeneous_s4_r3@ep5"
TL5_NAME = "fttall_tlead_s5@ep3"
TL6_NAME = "fttall_tlead_s6@ep3"


def sha256_bytes(data: bytes) -> str:
    """Return a byte payload's SHA-256 digest."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    """Read a pinned JSON object and reject other top-level values."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def verify_file(path: Path, expected_sha256: str) -> None:
    """Require a regular source file to match its frozen digest."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"source is missing or is not a regular file: {path}")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"source SHA-256 changed: {path}")


def read_pinned_entry(archive_path: Path, entry_name: str, digest: str) -> bytes:
    """Read one ZIP entry and verify its uncompressed bytes."""
    with ZipFile(archive_path) as archive:
        payload = archive.read(entry_name)
    if sha256_bytes(payload) != digest:
        raise ValueError(f"ZIP entry SHA-256 changed: {archive_path}:{entry_name}")
    return payload


def verify_candidate_provenance(path: Path, digest: str, seed: int, checkpoint_sha: str) -> None:
    """Check the saved H390 run pins and record its all-data exposure contract."""
    verify_file(path, digest)
    provenance = read_json(path)
    args = provenance.get("args")
    checkpoint_hashes = provenance.get("checkpoint_sha256")
    membership = provenance.get("fit_membership_identity")
    if not isinstance(args, dict) or not isinstance(checkpoint_hashes, dict):
        raise ValueError(f"H390 source provenance has an unexpected schema: {path}")
    if not isinstance(membership, dict):
        raise ValueError(f"H390 source provenance has no fit membership: {path}")
    if args.get("seed") != seed or args.get("train_source") != "train+valid":
        raise ValueError(f"H390 source fit identity or exposure changed: {path}")
    if args.get("precision") != "fp16" or args.get("epochs") != 6:
        raise ValueError(f"H390 source precision or epoch schedule changed: {path}")
    if checkpoint_hashes.get("model_ep5.pt") != checkpoint_sha:
        raise ValueError(f"H390 epoch-5 checkpoint pin changed: {path}")
    if provenance.get("evaluation_row_groups") != []:
        raise ValueError(f"H390 source unexpectedly declares a held-out evaluation: {path}")
    if membership.get("dataset_length") != 12_479:
        raise ValueError(f"H390 all-data fit size changed: {path}")
    if membership.get("train", {}).get("count") != 10_606:
        raise ValueError(f"H390 train membership count changed: {path}")
    if membership.get("valid", {}).get("count") != 1_873:
        raise ValueError(f"H390 valid membership count changed: {path}")


def verify_source_assets() -> tuple[dict[str, object], bytes, bytes, dict[str, object]]:
    """Verify source archive entries, H390 bytes, and provenance before building."""
    verify_file(AUTHORITY_PATH, AUTHORITY_SHA256)
    authority = read_json(AUTHORITY_PATH)
    endpoints = authority.get("proposed_fixed_endpoints")
    if not isinstance(endpoints, list) or [item.get("id") for item in endpoints] != [
        "tl3_h390_s3",
        "tl3_h390_s3_s4",
    ]:
        raise ValueError("quota authority no longer pins these two endpoint IDs")
    if authority.get("new_fits") != 0 or authority.get("coefficient_search") is not False:
        raise ValueError("quota authority no longer limits this work to fixed existing weights")
    expected_authority_members = [
        {
            "id": "tl3_h390_s3",
            "members": {S3_NAME: 0.34, "original_tl5": 0.33, "original_tl6": 0.33},
        },
        {
            "id": "tl3_h390_s3_s4",
            "members": {S3_NAME: 0.17, S4_NAME: 0.17, "original_tl5": 0.33, "original_tl6": 0.33},
        },
    ]
    if endpoints != expected_authority_members:
        raise ValueError("quota authority's fixed coefficients differ from this recipe")
    verify_file(BASE_ZIP, BASE_ZIP_SHA256)
    verify_file(BASE_RECEIPT, BASE_RECEIPT_SHA256)
    verify_file(E_ZIP, E_ZIP_SHA256)
    verify_file(S3_CANARY_ZIP, S3_CANARY_ZIP_SHA256)
    verify_file(E_S3, S3_SHA256)
    verify_file(E_S4, S4_SHA256)

    receipt = read_json(BASE_RECEIPT)
    entries = receipt.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("pinned original TL3 archive receipt has no entries map")
    with ZipFile(BASE_ZIP) as archive:
        if archive.testzip() is not None:
            raise BadZipFile("original TL3 source archive has a CRC failure")
        if set(archive.namelist()) != set(entries):
            raise ValueError("original TL3 archive entry names changed")
        original_members_bytes = archive.read("members.json")
        original_members = json.loads(original_members_bytes)
        base_entries: dict[str, object] = {}
        for info in archive.infolist():
            payload = archive.read(info)
            pin = entries[info.filename]
            if len(payload) != pin.get("bytes") or sha256_bytes(payload) != pin.get("sha256"):
                raise ValueError(f"original TL3 source entry changed: {info.filename}")
            base_entries[info.filename] = {
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "crc32": f"{info.CRC:08x}",
            }
    if not isinstance(original_members, list) or len(original_members) != 3:
        raise ValueError("original TL3 member map does not contain three entries")
    expected_base = [
        ("weights_s5.npz", "distilled_s5@ep1", 0.34),
        ("weights_tl5.npz", TL5_NAME, 0.33),
        ("weights_tl6.npz", TL6_NAME, 0.33),
    ]
    for item, expected in zip(original_members, expected_base, strict=True):
        if (item.get("weights"), item.get("name"), item.get("coef")) != expected:
            raise ValueError("original TL3 fixed member map differs from the recipe")

    s3 = read_pinned_entry(S3_CANARY_ZIP, "weights_0.npz", S3_SHA256)
    if s3 != E_S3.read_bytes():
        raise ValueError("S3 bytes differ between source #11 ZIP and member bank E")
    s4 = E_S4.read_bytes()
    if read_pinned_entry(E_ZIP, "weights_1.npz", S4_SHA256) != s4:
        raise ValueError("S4 bytes differ between the extracted member and E ZIP")
    canary_map = json.loads(
        read_pinned_entry(
            S3_CANARY_ZIP,
            "members.json",
            "12bd11e82706de104e92733fcc4de4b7af8310bd2b1c1ccf36a2423e3a927259",
        )
    )
    e_map = json.loads(
        read_pinned_entry(
            E_ZIP,
            "members.json",
            "b5101bb42bf53a29afe801e0b6df7b51fecba79061ad45ac99a37b5cbdcdae60",
        )
    )
    if canary_map[0].get("source") != S3_NAME:
        raise ValueError("source #11 ZIP no longer maps weights_0 to H390 S3")
    if [item.get("source") for item in e_map[:2]] != [S3_NAME, S4_NAME]:
        raise ValueError("member bank E no longer maps weights_0/1 to H390 S3/S4")

    s3_checkpoint = "04675f8220ba949c1dfb6c5796a3f25a9b8f655bb48f83c188bf8f9bacd361e0"
    s4_checkpoint = "cefee5b51cae7b72247a26558444f526e915a9fb7ad3927abd6cc02e63018419"
    verify_candidate_provenance(S3_PROVENANCE, S3_PROVENANCE_SHA256, 3, s3_checkpoint)
    verify_candidate_provenance(S4_PROVENANCE, S4_PROVENANCE_SHA256, 4, s4_checkpoint)
    source_summary = {
        "authority_sha256": AUTHORITY_SHA256,
        "base_zip_sha256": BASE_ZIP_SHA256,
        "base_receipt_sha256": BASE_RECEIPT_SHA256,
        "e_zip_sha256": E_ZIP_SHA256,
        "source_11_zip_sha256": S3_CANARY_ZIP_SHA256,
        "s3_payload_sha256": S3_SHA256,
        "s4_payload_sha256": S4_SHA256,
        "s3_provenance_sha256": S3_PROVENANCE_SHA256,
        "s4_provenance_sha256": S4_PROVENANCE_SHA256,
        "base_entries": base_entries,
        "source_fit_exposure": "train+valid; 10,606 train + 1,873 valid; no held-out groups",
        "native_execution_performed": False,
    }
    return (
        source_summary,
        s3,
        s4,
        {"original_members": original_members, "base_entries": base_entries},
    )


def updated_members(
    original: list[dict[str, object]], include_s4: bool
) -> tuple[bytes, list[dict[str, object]]]:
    """Build the two authority-frozen coefficient maps without altering TL5/TL6."""
    members = copy.deepcopy(original)
    members[0]["name"] = S3_NAME
    members[0]["coef"] = 0.17 if include_s4 else 0.34
    if include_s4:
        members.insert(1, {"weights": "weights_s4.npz", "name": S4_NAME, "coef": 0.17})
    if [member["coef"] for member in members] != (
        [0.17, 0.17, 0.33, 0.33] if include_s4 else [0.34, 0.33, 0.33]
    ):
        raise ValueError("fixed blend coefficients changed")
    if members[-2:] != original[1:]:
        raise ValueError("original TL5/TL6 records changed")
    return (json.dumps(members, indent=2) + "\n").encode("utf-8"), members


def build_one(
    candidate: dict[str, object],
    source_summary: dict[str, object],
    s3_payload: bytes,
    s4_payload: bytes,
    original_members: list[dict[str, object]],
    base_entries: dict[str, object],
    recipe_sha256: str,
) -> dict[str, object]:
    """Write and verify one create-only ZIP and its receipt."""
    include_s4 = candidate["id"] == "tl3_h390_s3_s4"
    members_bytes, member_map = updated_members(original_members, include_s4)
    output_path = PACKAGE_DIR / str(candidate["zip"])
    receipt_path = PACKAGE_DIR / str(candidate["receipt"])
    if output_path.exists() or output_path.is_symlink() or receipt_path.exists():
        raise FileExistsError(f"create-only package output already exists: {output_path}")
    expected_names = set(base_entries)
    if include_s4:
        expected_names.add("weights_s4.npz")

    with (
        ZipFile(BASE_ZIP) as source,
        output_path.open("xb") as output_stream,
        ZipFile(output_stream, "w") as output,
    ):
        for info in source.infolist():
            if info.filename == "members.json":
                payload = members_bytes
            elif info.filename == "weights_s5.npz":
                payload = s3_payload
            else:
                payload = source.read(info)
            output.writestr(info, payload)
        if include_s4:
            new_info = copy.copy(source.getinfo("weights_s5.npz"))
            new_info.filename = "weights_s4.npz"
            output.writestr(new_info, s4_payload)

    output_entries: dict[str, object] = {}
    with ZipFile(output_path) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != expected_names:
            raise BadZipFile("candidate ZIP CRC or entry-set validation failed")
        for info in archive.infolist():
            payload = archive.read(info)
            entry_hash = sha256_bytes(payload)
            expected_hash = (
                S3_SHA256
                if info.filename == "weights_s5.npz"
                else S4_SHA256
                if info.filename == "weights_s4.npz"
                else sha256_bytes(members_bytes)
                if info.filename == "members.json"
                else base_entries[info.filename]["sha256"]
            )
            if entry_hash != expected_hash:
                raise ValueError(f"candidate entry changed unexpectedly: {info.filename}")
            output_entries[info.filename] = {
                "bytes": len(payload),
                "sha256": entry_hash,
                "crc32": f"{info.CRC:08x}",
            }
    if output_path.stat().st_size >= MAX_ZIP_BYTES:
        raise ValueError("candidate ZIP must be smaller than 20,000,000 bytes")
    receipt: dict[str, object] = {
        "schema": "wnn.tl3-h390-fixed-quota-package.v1",
        "status": "built; native ABI/reset/runtime and quality gates pending",
        "recipe": {"path": str(RECIPE_PATH.relative_to(ROOT)), "sha256": recipe_sha256},
        "authority": str(AUTHORITY_PATH.relative_to(ROOT)),
        "candidate_id": candidate["id"],
        "members": member_map,
        "sources": source_summary,
        "artifact": {
            "path": str(output_path.relative_to(ROOT)),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "zip_entries": output_entries,
        },
        "validation": {
            "all_source_entry_hashes_match": True,
            "S3_and_S4_hash_pins_match": True,
            "package_crc_passed": True,
            "under_20_000_000_bytes": True,
            "new_fits": 0,
            "target_reads_or_scores": 0,
            "native_import_or_prediction_run": False,
            "portal_upload": False,
        },
        "evidence_limits": [
            "H390 S3/S4 were fitted on train+valid, so their saved validation predictions are exposed.",
            "Original TL5/TL6 fit provenance and validation exposure are not established by the preserved TL3 ZIP receipt.",
            "No exact composite grade or expected public-score gain is established.",
            "Structural NPZ compatibility does not establish parity or runtime under the TL3 wnn_cpp wrapper.",
        ],
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def main() -> None:
    """Build both authority-frozen packages from verified local source bytes."""
    recipe = read_json(RECIPE_PATH)
    recipe_sha256 = sha256_file(RECIPE_PATH)
    source_summary, s3_payload, s4_payload, base = verify_source_assets()
    candidates = recipe.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("frozen recipe must define exactly two candidate packages")
    expected_candidates = [
        {
            "id": "tl3_h390_s3",
            "zip": "tl3_h390_s3_20261024.zip",
            "receipt": "tl3_h390_s3_BUILD_RECEIPT_R1.json",
            "member_map": [
                {"weights": "weights_s5.npz", "name": S3_NAME, "coef": 0.34},
                {"weights": "weights_tl5.npz", "name": TL5_NAME, "coef": 0.33},
                {"weights": "weights_tl6.npz", "name": TL6_NAME, "coef": 0.33},
            ],
            "estimated_bytes_from_entry_sizes": 12_812_144,
            "new_fits": 0,
        },
        {
            "id": "tl3_h390_s3_s4",
            "zip": "tl3_h390_s3_s4_20261024.zip",
            "receipt": "tl3_h390_s3_s4_BUILD_RECEIPT_R1.json",
            "added_member_path": "weights_s4.npz",
            "member_map": [
                {"weights": "weights_s5.npz", "name": S3_NAME, "coef": 0.17},
                {"weights": "weights_s4.npz", "name": S4_NAME, "coef": 0.17},
                {"weights": "weights_tl5.npz", "name": TL5_NAME, "coef": 0.33},
                {"weights": "weights_tl6.npz", "name": TL6_NAME, "coef": 0.33},
            ],
            "estimated_bytes_from_entry_sizes": 16_720_424,
            "new_fits": 0,
        },
    ]
    if candidates != expected_candidates:
        raise ValueError(
            "recipe candidates or coefficients differ from the fixed endpoint contract"
        )
    receipts = [
        build_one(
            candidate,
            source_summary,
            s3_payload,
            s4_payload,
            base["original_members"],
            base["base_entries"],
            recipe_sha256,
        )
        for candidate in candidates
    ]
    result_path = PACKAGE_DIR / "BUILD_RESULTS_R1.json"
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(
            {"schema": "wnn.tl3-h390-quota-build-results.v1", "packages": receipts},
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    print(json.dumps(receipts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
