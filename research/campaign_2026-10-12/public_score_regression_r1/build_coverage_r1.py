#!/usr/bin/env python3
"""Build source-qualified coverage tables from the frozen portal roster."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

MAIN_EVIDENCE_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OUT = Path(__file__).resolve().parent
RAW = OUT / "raw_portal_roster.tsv"

EVIDENCE_INPUTS = {
    "registry": "research/lb/registry.json",
    "portal_20261004_late": "research/campaign_2026-10-04/PORTAL_RESULTS_20261004_LATE.json",
    "portal_20261004": "research/campaign_2026-10-04/portal_results_20261004.json",
    "local_368": "research/campaign_2026-09-29/artifacts/nested_public_calibration_pair.json",
    "portal_20261010": "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/PUBLIC_RESULTS_20261010_R2.json",
    "public_scores_20261011": "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/PUBLIC_SCORES_20261011.md",
    "transfer_20261011": "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/PUBLIC_TRANSFER_RESPONSE_20261011.md",
    "upload_20261009_manifest": "submissions/upload_2026-10-09/MANIFEST.json",
    "upload_20261010_manifest": "submissions/upload_2026-10-10/MANIFEST.json",
    "upload_20261010_sums": "submissions/upload_2026-10-10/SHA256SUMS",
    "upload_20261011_sums": "submissions/upload_2026-10-11/SHA256SUMS",
}

EXTERNAL_CURRENT_RECEIPTS = [
    {
        "path": "submissions/10-11-67/7_dm_t3_a9_e6_fold1_q8vnni_build_receipt.json",
        "sha256": "7a7f6374aea6857d83fb23cb4036f9d3262c9c9f392114d626aca6f8b89480f2",
    },
    {
        "path": "submissions/10-11-67/9_muonall_prop0_e4_s4_fold1_q8vnni_build_receipt.json",
        "sha256": "a27d878b628500b5f1dd6db707cd93592830ef18e75fd40092e9419a9894ad59",
    },
    {
        "path": "submissions/10-11-67/8_dm_t3_a9_e6_muonall_prop0_e4_s4_pair_fold1_q8vnni_build_receipt.json",
        "sha256": "0f16fe6b8bb31c52ee1a689574c807c66ac3086659ed951877141d180ede543a",
    },
]

CURRENT = {
    "2i0hhdpa": {
        "package_alias": "12_h388_teacher_trio_all_fp16store_q8vnni",
        "zip_path": "submissions/10-11-67/12_h388_teacher_trio_all_fp16store_q8vnni.zip",
        "members": [
            ["vt_all_h388_muonall_h384_e3_s3@ep2", 1 / 3],
            ["vt_all_h388_muonall_h384_e3_s4@ep2", 1 / 3],
            ["vt_all_h388_muonall_h384_e3_s5@ep2", 1 / 3],
        ],
        "precision": "Q8VNNI runtime; selective FP16 storage (source receipt cited)",
        "fit_membership": "all-data fit; 12,479 fit groups, no held-out claim",
        "local_feature": 0.6725416214,
        "local_feature_kind": "same-recipe fold1 sibling proxy; target weights differ and fold1 is exposed",
        "local_feature_source": "wnn_submission_selector_20261011_r1/generated_r6/F1_SELECTOR_METRICS_R1.json",
        "ancestry_cluster": "h388_teacher_trio_all_data",
        "notes": "Timeout target is missing/censored; local feature is a different fold-trained sibling, not exact weights.",
    },
    "2ebpj2er": {
        "package_alias": "7_dm_t3_a9_e6_fold1_q8vnni",
        "zip_path": "submissions/10-11-67/7_dm_t3_a9_e6_fold1_q8vnni.zip",
        "zip_sha256": "e1f218e88b6e704b2c9b5f8753c3924388a91e44e69ab4ff30702561c418734f",
        "zip_bytes": 4928931,
        "members": [["vt_f1_dsq_dm_t3_a9_e6@ep5", 1.0]],
        "precision": "Q8VNNI (build receipt)",
        "fit_membership": "fold1: 11,949 fit groups; 457 exposed evaluation groups",
        "local_feature": 0.67103,
        "local_feature_kind": "receipt-bound checkpoint float prediction on exposed/selected fold1; native Q8 output differs",
        "local_feature_source": "ops_continuation_20261011/RESEARCH_PAIR_SCORES_R1.txt and build receipt",
        "ancestry_cluster": "fold1_student_scratch_pair_component",
        "notes": "Selected after exposed fold1 comparisons; score is not a fresh holdout or public forecast.",
    },
    "0srim121": {
        "package_alias": "9_muonall_prop0_e4_s4_fold1_q8vnni",
        "zip_path": "submissions/10-11-67/9_muonall_prop0_e4_s4_fold1_q8vnni.zip",
        "zip_sha256": "93366e8580b2fa00d7497aa3268b1aa8022c2ab9be6097ecf6ec758475dbe108",
        "zip_bytes": 4928070,
        "members": [["vt_f1_muonall_prop0_e4_s4@ep3", 1.0]],
        "precision": "Q8VNNI (build receipt)",
        "fit_membership": "fold1: 11,949 fit groups; 457 exposed evaluation groups",
        "local_feature": 0.67195,
        "local_feature_kind": "receipt-bound checkpoint float prediction on exposed/selected fold1; native Q8 output differs",
        "local_feature_source": "ops_continuation_20261011/RESEARCH_PAIR_SCORES_R1.txt and build receipt",
        "ancestry_cluster": "fold1_student_scratch_pair_component",
        "notes": "Standalone scratch sibling is a blend component in ID 0hikjxpr; group dependent.",
    },
    "0hikjxpr": {
        "package_alias": "8_dm_t3_a9_e6_muonall_prop0_e4_s4_pair_fold1_q8vnni",
        "zip_path": "submissions/10-11-67/8_dm_t3_a9_e6_muonall_prop0_e4_s4_pair_fold1_q8vnni.zip",
        "zip_sha256": "4d6198c2573dfbe9013025cbfa2111246102da53ea5b3226916d4b1a3f7f679d",
        "zip_bytes": 8555307,
        "members": [
            ["vt_f1_dsq_dm_t3_a9_e6@ep5", 0.5],
            ["vt_f1_muonall_prop0_e4_s4@ep3", 0.5],
        ],
        "precision": "Q8VNNI (build receipt)",
        "fit_membership": "fold1: 11,949 fit groups; 457 exposed evaluation groups",
        "local_feature": 0.67329,
        "local_feature_kind": "receipt-bound pair float prediction on exposed/selected fold1; native Q8 output differs",
        "local_feature_source": "ops_continuation_20261011/RESEARCH_PAIR_SCORES_R1.txt and build receipt",
        "ancestry_cluster": "fold1_student_scratch_pair_component",
        "notes": "Reuses both standalone members; must be grouped with both for validation.",
    },
    "19w7a96y": {
        "package_alias": "1_p2_student_distilled_from_scratch_h256_teachers",
        "zip_path": "submissions/upload_2026-10-10/1_p2_student_distilled_from_scratch_h256_teachers.zip",
        "zip_sha256": "5c8c4e88bb83b5a658cdd57c3e4f3a7fa77eedc06435a695ceec063c6d250892",
        "members": [["vt_f1_dsq_t256@ep5", 1.0]],
        "precision": "not stated in the source manifest; not inferred from filename",
        "fit_membership": "fold1 student; train folds 0,2,3; fold1 not fitted (manifest)",
        "local_feature": 0.66938,
        "local_feature_kind": "package-manifest-reported float fold1 metric; checkpoint digest absent",
        "local_feature_source": "PUBLIC_RESULTS_20261010_R2.json and upload manifest",
        "ancestry_cluster": "fold1_distilled_h256",
        "notes": "Exact checkpoint hash was not in the consulted upload manifest.",
    },
    "2j7iv02s": {
        "package_alias": "2_p3_student_distilled_from_scratch_h384_teachers",
        "zip_path": "submissions/upload_2026-10-10/2_p3_student_distilled_from_scratch_h384_teachers.zip",
        "zip_sha256": "a5499f891bbef90e602316dfb11b8d07d043107ffbcbd142e00b5907e43658bb",
        "members": [["vt_f1_dsq_t384@ep5", 1.0]],
        "precision": "not stated in the source manifest; not inferred from filename",
        "fit_membership": "fold1 student; train folds 0,2,3; fold1 not fitted (manifest)",
        "local_feature": 0.66836,
        "local_feature_kind": "exact-weight fold1 metric; exposed",
        "local_feature_source": "PUBLIC_RESULTS_20261010_R2.json and upload manifest",
        "ancestry_cluster": "fold1_distilled_h384",
        "notes": "Timeout target is missing/censored; no public score imputation.",
    },
    "1ib94vp5": {
        "package_alias": "3_p1_h230_ep3_student",
        "zip_path": "submissions/upload_2026-10-10/3_p1_h230_ep3_student.zip",
        "zip_sha256": "b604475dc1f97d54711a760e6e92018c3d4727ac2af21e4b174a602487648df9",
        "members": [["bd26_h230_f1_s3_ep3@ep2", 1.0]],
        "precision": "not stated in the source manifest; not inferred from filename",
        "fit_membership": "fold1 student; train folds 0,2,3; fold1 not fitted (manifest)",
        "local_feature": 0.66923,
        "local_feature_kind": "exact-weight fold1 metric; exposed",
        "local_feature_source": "PUBLIC_RESULTS_20261010_R2.json and upload manifest",
        "ancestry_cluster": "fold1_h230_ep3_student",
        "notes": "Timeout target is missing/censored; no public score imputation.",
    },
    "0o3jmcdi": {
        "package_alias": "4_z1c_z4_pair_q8vnni",
        "zip_path": "submissions/upload_2026-10-11/z1c_z4_pair_q8vnni/submission.zip",
        "zip_sha256": "ff498873027318b5aa20e6ca95517bdb843b4656989a75aae148f1c141705327",
        "members": [
            [
                "bd26_h230_all_data_student_gru256_seed4_retry1_v1_core@ep1|z1c_h230s4_student_hft_alldata_v3@ep0",
                0.5,
            ],
            [
                "bd26_h230_faithful_student_seed99_muonparent_v1_core_identity_v2@ep1|z4_seed99_student_hft_alldata_v2@ep0",
                0.5,
            ],
        ],
        "precision": "Q8VNNI (package precision file)",
        "fit_membership": "both components all-data; validation appears in fitting; no held-out local score",
        "local_feature": None,
        "local_feature_kind": "in-sample/missing comparable feature",
        "local_feature_source": None,
        "ancestry_cluster": "all_data_z1c_z4_pair",
        "notes": "Public pair gain is an outcome; do not use it as an input for this upload.",
    },
    "2px86azm": {
        "package_alias": "5_z1c_q8vnni_single",
        "zip_path": "submissions/upload_2026-10-11/z1c_q8vnni_single/submission.zip",
        "zip_sha256": "936f88d75f2950e17e0ba093328e5abec1cf6cbcc8d390baad8b8a3e38cdf83b",
        "members": [
            [
                "bd26_h230_all_data_student_gru256_seed4_retry1_v1_core@ep1|z1c_h230s4_student_hft_alldata_v3@ep0",
                1.0,
            ]
        ],
        "precision": "Q8VNNI (package precision file)",
        "fit_membership": "all-data; validation appears in fitting; no held-out local score",
        "local_feature": None,
        "local_feature_kind": "in-sample/missing comparable feature",
        "local_feature_source": None,
        "ancestry_cluster": "all_data_z1c_h230s4",
        "notes": "Same member as historical i16 z1c upload, different served precision/package.",
    },
    "1hv91iv3": {
        "package_alias": "5_z2c_alldata_s3_student_hft",
        "members": [
            [
                "bd26_h230_all_data_student_gru256_seed3_v1_core@ep1|z2c_h230s3_student_hft_alldata_v3@ep0",
                1.0,
            ]
        ],
        "precision": "i16 declared by 2026-10-09 upload manifest; package bytes not independently inspected here",
        "fit_membership": "all-data; validation appears in fitting; no held-out local score",
        "local_feature": None,
        "local_feature_kind": "in-sample/missing comparable feature",
        "ancestry_cluster": "all_data_z2c_h230s3",
        "notes": "Source package and ZIP SHA are in upload_2026-10-09/MANIFEST.json.",
    },
    "16zlko1e": {
        "package_alias": "2_z5_seed9_student_muonparent_hft",
        "members": [
            [
                "bd26_h230_faithful_student_seed9_muonparent_v1_core_identity_v2@ep1|z5_seed9_student_hft_alldata_v2@ep0",
                1.0,
            ]
        ],
        "precision": "i16 declared by 2026-10-09 upload manifest; package bytes not independently inspected here",
        "fit_membership": "all-data; validation appears in fitting; no held-out local score",
        "local_feature": None,
        "local_feature_kind": "in-sample/missing comparable feature",
        "ancestry_cluster": "all_data_z5_seed9",
        "notes": "Source package and ZIP SHA are in upload_2026-10-09/MANIFEST.json.",
    },
    "14mn8z0a": {
        "package_alias": "3_z3_student_from_publicbest_trunk",
        "members": [
            [
                "z3_combo_s4_parent_student_v1_core@ep1|z3h_combo_s4_parent_student_hft_alldata_v1@ep0",
                1.0,
            ]
        ],
        "precision": "i16 declared by 2026-10-09 upload manifest; package bytes not independently inspected here",
        "fit_membership": "all-data; validation appears in fitting; no held-out local score",
        "local_feature": None,
        "local_feature_kind": "in-sample/missing comparable feature",
        "ancestry_cluster": "all_data_z3_combo_s4",
        "notes": "Warm-start/public-best lineage; local source evidence is source-qualified, not a held-out score.",
    },
    "0fiposnk": {
        "package_alias": "1_z4_seed99_student_muonparent_hft",
        "members": [
            [
                "bd26_h230_faithful_student_seed99_muonparent_v1_core_identity_v2@ep1|z4_seed99_student_hft_alldata_v2@ep0",
                1.0,
            ]
        ],
        "precision": "i16 declared by 2026-10-09 upload manifest; package bytes not independently inspected here",
        "fit_membership": "all-data; validation appears in fitting; no held-out local score",
        "local_feature": None,
        "local_feature_kind": "in-sample/missing comparable feature",
        "ancestry_cluster": "all_data_z4_seed99",
        "notes": "Public outcome cannot be back-substituted as a pre-upload predictor.",
    },
    "19swjd19": {
        "package_alias": "4_z1c_alldata_s4_student_hft",
        "members": [
            [
                "bd26_h230_all_data_student_gru256_seed4_retry1_v1_core@ep1|z1c_h230s4_student_hft_alldata_v3@ep0",
                1.0,
            ]
        ],
        "precision": "i16 declared by 2026-10-09 upload manifest; package bytes not independently inspected here",
        "fit_membership": "all-data; validation appears in fitting; no held-out local score",
        "local_feature": None,
        "local_feature_kind": "in-sample/missing comparable feature",
        "ancestry_cluster": "all_data_z1c_h230s4",
        "notes": "Shares model member with q8 ID 2px86azm; repeated-weight pair is dependent.",
    },
    "20o154uy": {
        "package_alias": "c20261008_h230_all_data_pair_i16",
        "members": None,
        "precision": "i16 label from package name; exact package receipt needed",
        "fit_membership": "all-data per package lineage; no held-out local score",
        "local_feature": None,
        "local_feature_kind": "in-sample/missing comparable feature",
        "ancestry_cluster": "h230_all_data_pair",
        "notes": "Scored 0.6653 despite 61:22; portal success retained.",
    },
    "27t6d4ur": {
        "package_alias": "c20261008_h230_all_data_s3_history_swap_i16",
        "precision": "i16 label from package name; exact package receipt needed",
        "fit_membership": "all-data per package lineage; no held-out local score",
        "local_feature": None,
        "local_feature_kind": "in-sample/missing comparable feature",
        "ancestry_cluster": "h230_all_data_history_swap",
        "notes": "Outcome can be modeled only from prior source-qualified features, not its own runtime/status.",
    },
    "1n7tmww1": {
        "package_alias": "c20261008_h230_all_data_s3_normalized_i16",
        "precision": "i16 label from package name; exact package receipt needed",
        "fit_membership": "all-data per package lineage; no held-out local score",
        "local_feature": None,
        "local_feature_kind": "in-sample/missing comparable feature",
        "ancestry_cluster": "h230_all_data_normalized_s3",
        "notes": "All-data local evaluation is in-sample, not an OOS feature.",
    },
    "07d1xqsz": {
        "package_alias": "h230_hetero_s3_ep1_i16",
        "members": [["bd26_h230_hetero_s3_f0@ep1", 1.0]],
        "precision": "i16 label from package name; exact package receipt needed",
        "fit_membership": "fold0: 11,940 fit groups; 468 exposed evaluation groups (source receipt)",
        "local_feature": None,
        "local_feature_kind": "no comparable exact endpoint WP located",
        "ancestry_cluster": "h230_fold0_hetero_s3",
        "notes": "Held-out membership is known but an exact local WP feature was not found in the consulted source.",
    },
    "1cymqpkk": {
        "package_alias": "c20261007_c1_s4_i16",
        "members": [["bd26_c1v2_s4_f0@ep1", 1.0]],
        "precision": "i16 label from package name; exact package receipt needed",
        "fit_membership": "fold0: 11,940 fit groups; 468 exposed evaluation groups (source receipt)",
        "local_feature": None,
        "local_feature_kind": "local deltas only; no exact absolute endpoint WP located",
        "ancestry_cluster": "c1_fold0_seed_pair",
        "notes": "Published local delta vs C0 is not an absolute local score feature.",
    },
    "1n6uxwb6": {
        "package_alias": "c20261007_c1_s3_i16",
        "members": [["bd26_c1v2_s3_f0@ep1", 1.0]],
        "precision": "i16 label from package name; exact package receipt needed",
        "fit_membership": "fold0: 11,940 fit groups; 468 exposed evaluation groups (source receipt)",
        "local_feature": None,
        "local_feature_kind": "local deltas only; no exact absolute endpoint WP located",
        "ancestry_cluster": "c1_fold0_seed_pair",
        "notes": "Published local delta vs C0 is not an absolute local score feature.",
    },
    "1yv88h4s": {
        "package_alias": "E462_public_q8_build_v2q8r",
        "precision": "unknown; check-error before usable score",
        "fit_membership": "unknown/not material to missing public score",
        "local_feature": None,
        "local_feature_kind": "not scored; check error",
        "ancestry_cluster": None,
        "notes": "Registry records prerun ImportError for libpython3.13.so.1.0; public score missing.",
    },
}

LOCAL_368_GROUP = {
    "1jdh5rtz": "old_368_s4_baseline_blends",
    "1x8nyiwi": "old_368_s4_baseline_blends",
    "05ytuk8y": "old_368_s4_baseline_blends",
    "21svxfgi": "old_368_s4_baseline_blends",
    "1bmxngta": "old_368_shared_since_ema",
    "246401vj": "old_368_shared_since_ema",
    "1ykwb95u": "old_368_shared_since_ema",
}
LOCAL_368_MEMBER_PROXY_IDS = {"246401vj", "1ykwb95u"}

FOLD1_LOCAL = {
    "19w7a96y": (0.66938, "fold1_package_reported", "fold1_distilled_h256"),
    "2j7iv02s": (0.66836, "fold1_package_reported", "fold1_distilled_h384"),
    "1ib94vp5": (0.66923, "fold1_package_reported", "fold1_h230_ep3"),
    "2ebpj2er": (0.67103, "fold1_exact_receipt", "fold1_student_scratch_pair_component"),
    "0srim121": (0.67195, "fold1_exact_receipt", "fold1_student_scratch_pair_component"),
    "0hikjxpr": (0.67329, "fold1_exact_receipt", "fold1_student_scratch_pair_component"),
}

SOURCE_ADDITIONS: dict[str, list[str]] = {
    "2i0hhdpa": [
        "submissions/10-11-67/TOP_CANDIDATES_R3.md",
        "submissions/10-11-67/READY_TOP_CANDIDATES_R6.json",
    ],
    "2ebpj2er": ["submissions/10-11-67/7_dm_t3_a9_e6_fold1_q8vnni_build_receipt.json"],
    "0srim121": ["submissions/10-11-67/9_muonall_prop0_e4_s4_fold1_q8vnni_build_receipt.json"],
    "0hikjxpr": [
        "submissions/10-11-67/8_dm_t3_a9_e6_muonall_prop0_e4_s4_pair_fold1_q8vnni_build_receipt.json"
    ],
    "19w7a96y": [
        "submissions/upload_2026-10-10/MANIFEST.json",
        "submissions/upload_2026-10-10/SHA256SUMS",
    ],
    "2j7iv02s": [
        "submissions/upload_2026-10-10/MANIFEST.json",
        "submissions/upload_2026-10-10/SHA256SUMS",
    ],
    "1ib94vp5": [
        "submissions/upload_2026-10-10/MANIFEST.json",
        "submissions/upload_2026-10-10/SHA256SUMS",
    ],
    "0o3jmcdi": [
        "submissions/upload_2026-10-11/SHA256SUMS",
        "submissions/upload_2026-10-11/z1c_z4_pair_q8vnni/pkg/members.json",
    ],
    "2px86azm": [
        "submissions/upload_2026-10-11/SHA256SUMS",
        "submissions/upload_2026-10-11/z1c_q8vnni_single/pkg/members.json",
    ],
    "16zlko1e": ["submissions/upload_2026-10-09/MANIFEST.json"],
    "14mn8z0a": ["submissions/upload_2026-10-09/MANIFEST.json"],
    "0fiposnk": ["submissions/upload_2026-10-09/MANIFEST.json"],
    "19swjd19": ["submissions/upload_2026-10-09/MANIFEST.json"],
    "1hv91iv3": ["submissions/upload_2026-10-09/MANIFEST.json"],
}


def read_json(relative_path: str) -> Any:
    return json.loads((MAIN_EVIDENCE_ROOT / relative_path).read_text(encoding="utf-8"))


def as_json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if value is not None else ""


def main() -> None:
    with RAW.open(newline="", encoding="utf-8") as stream:
        portal_rows = list(csv.DictReader(stream, delimiter="|"))

    registry = read_json(EVIDENCE_INPUTS["registry"])
    reg_by_id = {row["submission_id"]: row for row in registry["subs"]}
    reg_by_id.update({row["submission_id"]: row for row in registry["failed_submissions"]})
    portal_late = read_json(EVIDENCE_INPUTS["portal_20261004_late"])
    portal_early = read_json(EVIDENCE_INPUTS["portal_20261004"])
    portal_sources: dict[str, dict[str, Any]] = {}
    for key in ("new_results", "reconfirmed_results"):
        for row in portal_late.get(key, []):
            portal_sources[row["submission_id"]] = row
    for row in portal_early.get("records", []):
        portal_sources[row["submission_id"]] = row

    local_368 = read_json(EVIDENCE_INPUTS["local_368"])
    holdout_local_by_name = dict(
        zip(
            local_368["submission_order"],
            local_368["local_scores_by_variant_by_submission"][0],
            strict=True,
        )
    )
    name_to_id = {row["name"]: row["submission_id"] for row in registry["subs"]}
    local_368_by_id = {
        name_to_id[name]: score
        for name, score in holdout_local_by_name.items()
        if name in name_to_id and name in LOCAL_368_GROUP_NAME_SET
    }

    uploads_20261009 = read_json(EVIDENCE_INPUTS["upload_20261009_manifest"])
    upload_20261009_by_name = {
        row["upload_file"].removesuffix(".zip"): row for row in uploads_20261009["packages"]
    }
    uploads_20261010 = read_json(EVIDENCE_INPUTS["upload_20261010_manifest"])
    upload_20261010_hashes: dict[str, str] = {}
    for line in (
        (MAIN_EVIDENCE_ROOT / "submissions/upload_2026-10-10/SHA256SUMS")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        digest, filename = line.split(maxsplit=1)
        upload_20261010_hashes[filename.strip()] = digest

    output_rows: list[dict[str, Any]] = []
    for raw in portal_rows:
        sid = raw["submission_id"]
        base = reg_by_id.get(sid, {}).copy()
        base.update(portal_sources.get(sid, {}))
        extra = CURRENT.get(sid, {})
        if not extra:
            extra = {}

        members = extra.get("members")
        if members is None:
            members = base.get("members") or base.get("routed")
        source = ";".join(
            sorted(
                {
                    value
                    for value in (
                        "user-provided raw_portal_roster.tsv",
                        "research/lb/registry.json" if sid in reg_by_id else "",
                        base.get("evidence") or "",
                        base.get("latest_portal_evidence") or "",
                        extra.get("local_feature_source") or "",
                    )
                    if value
                }
            )
        )
        if sid in SOURCE_ADDITIONS:
            source = ";".join(sorted(set(filter(None, [source, *SOURCE_ADDITIONS[sid]]))))

        package_name = extra.get("package_alias", base.get("name", ""))
        zip_path = extra.get("zip_path", base.get("zip_path", ""))
        zip_hash = extra.get("zip_sha256", base.get("zip_sha256", ""))
        zip_bytes: int | None = None
        manifest_item = None
        if sid in {"16zlko1e", "14mn8z0a", "0fiposnk", "19swjd19", "1hv91iv3"}:
            # User portal aliases are not necessarily the upload manifest names.
            manifest_name_by_id = {
                "16zlko1e": "2_z5_seed9_student_muonparent_hft",
                "14mn8z0a": "3_z3_student_from_publicbest_trunk",
                "0fiposnk": "1_z4_seed99_student_muonparent_hft",
                "19swjd19": "4_z1c_alldata_s4_student_hft",
                "1hv91iv3": "5_z2c_alldata_s3_student_hft",
            }
            manifest_item = upload_20261009_by_name.get(manifest_name_by_id[sid])
            if manifest_item:
                zip_path = f"submissions/upload_2026-10-09/{manifest_item['upload_file']}"
                zip_hash = manifest_item["zip_sha256"]
                zip_bytes = manifest_item["bytes"]
        if sid in {"19w7a96y", "2j7iv02s", "1ib94vp5"}:
            item = next(
                row
                for row in uploads_20261010["packages"]
                if row["upload_file"] == raw["zip_filename"]
            )
            zip_path = f"submissions/upload_2026-10-10/{item['upload_file']}"
            members = [[item.get("run", "unknown"), 1.0]]
            package_name = item.get("source_package", package_name)
            zip_hash = upload_20261010_hashes.get(item["upload_file"], "")

        local_value = extra.get("local_feature")
        local_kind = extra.get("local_feature_kind", "")
        local_source = extra.get("local_feature_source", "")
        ancestry = extra.get("ancestry_cluster")
        if sid in local_368_by_id:
            local_value = local_368_by_id[sid]
            local_kind = (
                "conditional v2-CPU reconstruction; EMA6 collapsed to member0, not exact served ensemble; retrospective"
                if sid in LOCAL_368_MEMBER_PROXY_IDS
                else "source-reported 368-group metric; script asserts holdout exclusion but ancestry is not receipt-verified; retrospective/exposed"
            )
            local_source = "research/campaign_2026-09-29/artifacts/nested_public_calibration_pair.json; scripts/lb_calibration.py"
            ancestry = LOCAL_368_GROUP[sid]
            extra = {
                **extra,
                "fit_membership": (
                    "metric uses 368 reserved groups; lb_calibration.py asserts all members excluded them, "
                    "but actual teacher/initializer/checkpoint receipts were not verified here"
                ),
            }
        if sid in FOLD1_LOCAL:
            value, kind, cluster = FOLD1_LOCAL[sid]
            local_value = value
            local_kind = kind
            local_source = extra.get(
                "local_feature_source",
                "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/PUBLIC_RESULTS_20261010_R2.json or ops RESEARCH_PAIR_SCORES_R1.txt",
            )
            ancestry = cluster

        # Upload-side fields are outcomes only. They are retained for feasibility
        # analysis and are intentionally not included in candidate model features.
        output_rows.append(
            {
                "display_timestamp": raw["display_timestamp"],
                "submission_id": sid,
                "portal_filename": raw["zip_filename"],
                "portal_status": raw["status"],
                "public_score_outcome": raw["score"],
                "duration_mm_ss_outcome": raw["duration_mm_ss"],
                "memory_mb_outcome": raw["memory_mb"],
                "package_alias": package_name,
                "zip_path": zip_path,
                "zip_sha256": zip_hash,
                "zip_bytes": zip_bytes,
                "member_specs_and_weights": as_json_cell(members),
                "precision_backend": extra.get(
                    "precision", "not established in consulted exact receipt"
                ),
                "fit_membership_and_exposure": extra.get(
                    "fit_membership", "not established in consulted source"
                ),
                "local_metric": local_value if local_value is not None else "",
                "local_metric_protocol": local_kind,
                "local_metric_precision_scope": (
                    "float checkpoint/pair research prediction; exact native Q8 score not measured"
                    if sid in FOLD1_LOCAL and FOLD1_LOCAL[sid][1] != "fold1_package_reported"
                    else "package-reported float metric; checkpoint digest and precision conversion not established"
                    if sid in FOLD1_LOCAL
                    else "conditional v2-CPU EMA member0 reconstruction; not exact served ensemble"
                    if sid in LOCAL_368_MEMBER_PROXY_IDS
                    else "historical v2-CPU calibration; native/package equivalence not established"
                    if sid in local_368_by_id
                    else "fold1 sibling float prediction; not the target all-data package's native output"
                    if sid == "2i0hhdpa"
                    else "not available"
                ),
                "local_metric_ancestry_status": (
                    "fold1 exclusion stated by upload manifest; checkpoint digest absent; evaluation exposed/selected"
                    if sid in {"19w7a96y", "2j7iv02s", "1ib94vp5"}
                    else "fold1 producer/build receipts bind checkpoint; evaluation exposed/selected; prediction is pre-native-quantization float"
                    if sid in {"2ebpj2er", "0srim121", "0hikjxpr"}
                    else "same-recipe fold1 sibling proxy; target all-data weights differ"
                    if sid == "2i0hhdpa"
                    else "368-group holdout asserted by calibration script; teacher/initializer/checkpoint exclusions not receipt-verified"
                    if sid in local_368_by_id
                    else "no comparable held-out feature; all-data or ancestry/membership not verified"
                ),
                "local_metric_source": local_source,
                "ancestry_cluster_for_validation": ancestry
                or "unresolved/not needed for local feature model",
                "source_evidence_paths": source,
                "audit_note": extra.get(
                    "notes",
                    "Exact ZIP/checkpoint/local-score relationship remains partial; no filename-based metric imputation.",
                ),
                "model_target_usable": raw["score"] != "" and raw["status"] == "SCORE",
                "predictor_input_policy": "public score/status/duration/memory are outcomes only; feature chronology retrospective or unverified",
            }
        )

    ids = [row["submission_id"] for row in output_rows]
    if len(ids) != 42 or len(set(ids)) != 42:
        raise ValueError(
            f"expected 42 unique portal IDs; got {len(ids)} rows / {len(set(ids))} IDs"
        )
    status_counts = {
        status: sum(row["portal_status"] == status for row in output_rows)
        for status in {r["portal_status"] for r in output_rows}
    }
    if status_counts != {"SCORE": 38, "TIMEOUT": 3, "CHECK_ERROR": 1}:
        raise ValueError(f"unexpected portal outcome counts: {status_counts}")

    csv_path = OUT / "endpoint_coverage_r1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    # Local-score rows have a public target only for successful submissions.
    candidate_rows = [
        row for row in output_rows if row["model_target_usable"] and row["local_metric"] != ""
    ]
    feature_path = OUT / "candidate_features_r1.csv"
    fields = [
        "display_timestamp",
        "submission_id",
        "public_score_outcome",
        "local_metric",
        "local_metric_protocol",
        "local_metric_precision_scope",
        "local_metric_ancestry_status",
        "ancestry_cluster_for_validation",
        "member_specs_and_weights",
        "fit_membership_and_exposure",
        "package_alias",
        "feature_time_status",
        "source_evidence_paths",
    ]
    with feature_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow(
                {key: row.get(key, "") for key in fields if key != "feature_time_status"}
                | {
                    "feature_time_status": "retrospective or unverified; not a prospective pre-upload feature"
                }
            )

    inputs = []
    for key, relative in EVIDENCE_INPUTS.items():
        path = MAIN_EVIDENCE_ROOT / relative
        content = path.read_bytes()
        inputs.append(
            {
                "id": key,
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    known_paths = {row["path"] for row in inputs}
    addition_paths = sorted({path for paths in SOURCE_ADDITIONS.values() for path in paths})
    for relative in addition_paths:
        if relative in known_paths:
            continue
        content = (MAIN_EVIDENCE_ROOT / relative).read_bytes()
        inputs.append(
            {
                "id": "row_source_addition",
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    for receipt in EXTERNAL_CURRENT_RECEIPTS:
        path = MAIN_EVIDENCE_ROOT / receipt["path"]
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != receipt["sha256"]:
            raise ValueError(f"current receipt digest mismatch: {receipt['path']}")
        inputs.append({"id": "current_build_receipt", "path": receipt["path"], "sha256": observed})
    (OUT / "SOURCE_INPUTS_R1.json").write_text(
        json.dumps({"inputs": inputs}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = {
        "visible_portal_rows": len(output_rows),
        "portal_total_reported": 45,
        "unlisted_unknown_rows": 3,
        "outcomes": status_counts,
        "scored_rows_with_receipt_bound_fold1_float_metric": sum(
            row["model_target_usable"]
            and row["submission_id"] in {"2ebpj2er", "0srim121", "0hikjxpr"}
            for row in output_rows
        ),
        "scored_rows_with_manifest_reported_fold1_float_metric": sum(
            row["model_target_usable"] and row["submission_id"] == "19w7a96y" for row in output_rows
        ),
        "scored_rows_with_source_reported_368_metric_ancestry_unverified": sum(
            row["model_target_usable"]
            and row["submission_id"] in local_368_by_id
            and row["submission_id"] not in LOCAL_368_MEMBER_PROXY_IDS
            for row in output_rows
        ),
        "scored_rows_with_conditional_ema_member0_proxy": sum(
            row["model_target_usable"] and row["submission_id"] in LOCAL_368_MEMBER_PROXY_IDS
            for row in output_rows
        ),
        "scored_rows_with_sibling_proxy": sum(
            row["model_target_usable"] and "sibling proxy" in row["local_metric_protocol"]
            for row in output_rows
        ),
        "scored_rows_without_comparable_local_feature": sum(
            row["model_target_usable"] and row["local_metric"] == "" for row in output_rows
        ),
        "timeout_or_error_rows_excluded_from_score_model": sum(
            not row["model_target_usable"] for row in output_rows
        ),
        "visible_rows_with_any_local_feature_including_censored_targets": sum(
            row["local_metric"] != "" for row in output_rows
        ),
        "visible_censored_rows_with_local_proxy": sum(
            not row["model_target_usable"] and row["local_metric"] != "" for row in output_rows
        ),
        "scored_local_feature_ids": [row["submission_id"] for row in candidate_rows],
    }
    (OUT / "COVERAGE_SUMMARY_R1.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


LOCAL_368_GROUP_NAME_SET = {
    "final_s4_base35",
    "final_s4ft_base35",
    "final_s4lpft_base35",
    "final_ema4_lpft_base35",
    "final_since2_gclp",
    "final_mixed_ema6_shared",
    "final_ema6_shared",
}


if __name__ == "__main__":
    main()
