"""Bounded metadata-only teacher lineage and feature-graph exclusion audit."""

from __future__ import annotations

import hashlib
import json
import logging
import resource
import time
from pathlib import Path

import numpy as np

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
LOGGER = logging.getLogger(__name__)
HASHES: dict[str, dict] = {}


def digest(path: Path) -> str:
    """Stream artifact bytes without parsing model/prediction values."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def record(path: Path, *, maximum: int = 64 * 1024 * 1024) -> dict:
    """Bind small metadata and checkpoint bytes; avoid large cache/model reads."""
    name = str(path.relative_to(ROOT))
    if not path.exists():
        info = {"exists": False}
    else:
        before = path.stat()
        value = digest(path) if before.st_size <= maximum else None
        after = path.stat()
        info = {
            "exists": True,
            "bytes": before.st_size,
            "sha256": value,
            "stable_during_read": (before.st_size, before.st_mtime_ns)
            == (after.st_size, after.st_mtime_ns),
        }
    HASHES[name] = info
    return info


def load(path: Path) -> dict:
    record(path)
    return json.loads(path.read_text())


def graph_components() -> dict[tuple[str, int], int]:
    """Recover undirected graph partition from bounded cached successor IDs."""
    path = ROOT / "cache/chains.npz"
    record(path)
    with np.load(path, allow_pickle=False) as data:
        nxt, split, group = data["next"], data["split"], data["row_group"]
    parent = list(range(len(nxt)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index, successor in enumerate(nxt):
        if int(successor) >= 0:
            a, b = find(index), find(int(successor))
            parent[max(a, b)] = min(a, b)
    roots = [find(index) for index in range(len(nxt))]
    dense = {root: index for index, root in enumerate(sorted(set(roots)))}
    return {
        (str(sp), int(rg)): dense[root]
        for sp, rg, root in zip(split, group, roots, strict=True)
    }


def audit_run(
    name: str,
    manifest: dict,
    component_of: dict,
    excluded: set[int],
    manifest_hash: str,
) -> dict:
    path = ROOT / "artifacts/runs" / name / "provenance.json"
    if not path.exists():
        record(path)
        return {
            "available": False,
            "status": "not available at audit; no inference about future job",
        }
    prov = load(path)
    initial_metadata_hash = HASHES[str(path.relative_to(ROOT))]["sha256"]
    args = prov.get("args", {})
    overlaps, counts, unknown = {}, {}, {}
    for split in ("train", "valid"):
        groups = prov.get("fit_row_groups", {}).get(split, [])
        counts[split] = len(groups)
        unknown[split] = [g for g in groups if (split, g) not in component_of]
        overlaps[split] = [
            {"row_group": int(g), "component": component_of[(split, g)]}
            for g in groups
            if (split, g) in component_of and component_of[(split, g)] in excluded
        ]
    checkpoints = {}
    for filename, expected in prov.get("checkpoint_sha256", {}).items():
        observed = record(path.parent / filename)
        checkpoints[filename] = {
            **observed,
            "recorded_sha256": expected,
            "matches_recorded": observed.get("sha256") == expected
            if observed.get("sha256")
            else None,
        }
    history_path = path.parent / "history.json"
    history = load(history_path) if history_path.exists() else None
    membership = prov.get("fit_membership_identity", {})
    hashes = {}
    for split in ("train", "valid"):
        if split in membership:
            values = np.asarray(prov["fit_row_groups"].get(split, []), dtype="<i8")
            hashes[split] = (
                membership[split].get("count") == len(values)
                and membership[split].get("ordered_int64_le_sha256")
                == hashlib.sha256(values.tobytes()).hexdigest()
            )
    relations = {
        key: args.get(key) for key in ("init", "distill", "propensity", "prop_input")
    }
    selected = prov.get("selected_epoch")
    return {
        "available": True,
        "metadata_sha256": HASHES[str(path.relative_to(ROOT))]["sha256"],
        "metadata_stable": record(path)["sha256"] == initial_metadata_hash,
        "fit_counts": counts,
        "reported_fit_component_intersections": overlaps,
        "unknown_component_rows": unknown,
        "reported_excludes_evaluation_components": not any(overlaps.values())
        and not any(unknown.values()),
        "reported_fit_matches_manifest": {
            sp: set(prov.get("fit_row_groups", {}).get(sp, []))
            == set(manifest["fit"].get(sp, []))
            for sp in ("train", "valid")
        },
        "split_hash_matches": prov.get("split_manifest_sha256") == manifest_hash,
        "evaluation_rows_match": set(prov.get("evaluation_row_groups", []))
        == set(manifest["evaluation"]["valid"]),
        "declared_relations": relations,
        "selected_epoch": selected,
        "checkpoint_policy": args.get("checkpoint_policy"),
        "checkpoint_epoch": args.get("checkpoint_epoch"),
        "no_eval_during_fit": args.get("no_eval_during_fit"),
        "history": history,
        "fixed_epoch_record_consistent": args.get("checkpoint_policy") == "fixed"
        and selected == args.get("checkpoint_epoch"),
        "checkpoints": checkpoints,
        "propensity_checkpoint_sha256": prov.get("propensity_checkpoint_sha256"),
        "group_membership_hash_checks": hashes,
        "source_bound_fields": {
            key: value for key, value in prov.items() if "source_sha" in key
        },
        "execution_evidence": "Recorded loader fit pools; no per-batch row log. Student group fingerprints/counters stronger when present; older teacher/propensity provenance lacks those and exact executed trainer-source identity.",
        "distill_bindings_recorded": args.get("distill_provenance", []),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start = time.monotonic()
    components = graph_components()
    indices = {}
    for split in ("train", "valid"):
        path = ROOT / "cache" / f"{split}_targets.npz"
        # Lazy NPZ: ONLY two ID arrays, never targets/masks/predictions.
        with np.load(path, allow_pickle=False) as data:
            groups, seqs = data["row_group"], data["seq_ix"]
        h = hashlib.sha256()
        for values in (groups, seqs):
            h.update(np.asarray(values, dtype="<i8").tobytes())
        indices[split] = {
            "row_groups": groups.tolist(),
            "seq_ix": seqs.tolist(),
            "index_sha256": h.hexdigest(),
        }
    result = {
        "folds": {},
        "indices": indices,
        "scope": "metadata only; no model load, prediction arrays or parquet",
    }
    for fold in range(4):
        mpath = ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
        manifest = load(mpath)
        mhash = digest(mpath)
        cmap = {
            (sp, int(row["row_group"])): int(row["component"])
            for sp, rows in manifest["sequences"].items()
            for row in rows
        }
        excluded = {cmap["valid", int(g)] for g in manifest["evaluation"]["valid"]}
        entry = {
            "manifest_sha256": mhash,
            "graph_mapping_exact": components == cmap,
            "evaluation_components": sorted(excluded),
            "evaluation_valid_groups": manifest["evaluation"]["valid"],
            "linked_train_groups_in_evaluation_components": sorted(
                g for (sp, g), c in cmap.items() if sp == "train" and c in excluded
            ),
            "nodes": {},
            "edges": [],
            "sidecars": {},
            "index_identity_checks": {},
        }
        for name, identity in manifest["sources"].items():
            for sp in indices:
                if name == f"cache/{sp}_targets.npz":
                    entry["index_identity_checks"][sp] = indices[sp][
                        "index_sha256"
                    ] == identity.get("index_sha256")
        student = f"c20261003_combo_s3_f{fold}"
        teachers = [f"c20260929_since_s{seed}_f{fold}" for seed in (3, 4)]
        prop = f"c20260929_prop_f{fold}"
        for name in [student, *teachers, prop]:
            entry["nodes"][name] = audit_run(name, manifest, cmap, excluded, mhash)
        for teacher in teachers:
            node = entry["nodes"][teacher]
            for sp in ("train", "valid"):
                path = ROOT / f"cache/pred_{teacher}_{sp}.json"
                side = load(path)
                checkpoint = node.get("checkpoints", {}).get(
                    f"model_ep{side.get('epoch')}.pt", {}
                )
                entry["sidecars"][f"{teacher}/{sp}"] = {
                    "metadata_sha256": digest(path),
                    "run_split_match": side.get("run") == teacher
                    and side.get("split") == sp,
                    "split_manifest_matches": side.get("split_manifest_sha256")
                    == mhash,
                    "row_order_matches_index": side.get("row_groups")
                    == indices[sp]["row_groups"],
                    "epoch": side.get("epoch"),
                    "checkpoint_sha256": side.get("checkpoint_sha256"),
                    "checkpoint_actual_bytes_match": checkpoint.get("sha256")
                    == side.get("checkpoint_sha256"),
                    "prediction_sha256_recorded": side.get("prediction_sha256"),
                    "prediction_bytes_rehashed": False,
                    "other_source_fields": {
                        k: v for k, v in side.items() if "source" in k
                    },
                }
            pnode = entry["nodes"][prop]
            expected_prop = node.get("propensity_checkpoint_sha256")
            actual_prop = pnode.get("checkpoints", {}).get("model.pt", {}).get("sha256")
            entry["edges"].append(
                {
                    "from": teacher,
                    "to": prop,
                    "kind": "propensity weights",
                    "declared_by_actual_teacher": node.get(
                        "declared_relations", {}
                    ).get("propensity")
                    == prop,
                    "checkpoint_bytes_link_match": expected_prop == actual_prop
                    and actual_prop is not None,
                }
            )
            bindings = entry["nodes"][student].get("distill_bindings_recorded", [])
            entry["edges"].append(
                {
                    "from": student,
                    "to": teacher,
                    "kind": "cached distillation",
                    "declared_by_actual_student": teacher
                    in (
                        entry["nodes"][student]
                        .get("declared_relations", {})
                        .get("distill")
                        or ""
                    ).split(","),
                    "recorded_bindings": [
                        b for b in bindings if b.get("run") == teacher
                    ],
                    "student_not_available": not entry["nodes"][student]["available"],
                }
            )
            edge = entry["edges"][-1]
            edge["bindings_match_current_sidecars"] = (
                (
                    len(edge["recorded_bindings"]) == 2
                    and all(
                        b.get("epoch")
                        == entry["sidecars"][f"{teacher}/{b.get('split')}"]["epoch"]
                        and b.get("checkpoint_sha256")
                        == entry["sidecars"][f"{teacher}/{b.get('split')}"][
                            "checkpoint_sha256"
                        ]
                        and b.get("prediction_sha256")
                        == entry["sidecars"][f"{teacher}/{b.get('split')}"][
                            "prediction_sha256_recorded"
                        ]
                        for b in edge["recorded_bindings"]
                    )
                )
                if entry["nodes"][student]["available"]
                else None
            )
        snode, pnode = entry["nodes"][student], entry["nodes"][prop]
        entry["edges"].append(
            {
                "from": student,
                "to": prop,
                "kind": "propensity weights",
                "declared_by_actual_student": snode.get("declared_relations", {}).get(
                    "propensity"
                )
                == prop,
                "checkpoint_bytes_link_match": snode.get("propensity_checkpoint_sha256")
                == pnode.get("checkpoints", {}).get("model.pt", {}).get("sha256")
                if snode["available"]
                else None,
            }
        )
        result["folds"][str(fold)] = entry
    for name in (
        "scripts/train_rnn.py",
        "scripts/train_propensity.py",
        "src/wnn/splits.py",
        "src/wnn/run_identity.py",
    ):
        record(ROOT / name)
    result["cost"] = {
        "wall_seconds": time.monotonic() - start,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    result["audit_source_sha256"] = digest(Path(__file__))
    (OWN / "lineage_map.json").write_text(json.dumps(result, indent=2) + "\n")
    (OWN / "provenance_hashes.json").write_text(json.dumps(HASHES, indent=2) + "\n")
    LOGGER.info(
        "SUMMARY %s",
        json.dumps(
            {
                fold: {
                    "graph": e["graph_mapping_exact"],
                    "linked_train": len(
                        e["linked_train_groups_in_evaluation_components"]
                    ),
                    "nodes": {
                        name: {
                            k: n.get(k)
                            for k in (
                                "available",
                                "fit_counts",
                                "reported_excludes_evaluation_components",
                                "fixed_epoch_record_consistent",
                            )
                        }
                        for name, n in e["nodes"].items()
                    },
                }
                for fold, e in result["folds"].items()
            }
        ),
    )


if __name__ == "__main__":
    main()
