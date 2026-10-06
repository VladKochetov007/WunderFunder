#!/usr/bin/env python3
"""Build and inspect a provenance-preserving research bandit tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

SCHEMA_VERSION = 1
HASH_LIMIT_BYTES = 1_048_576
DEFAULT_CONFIG = "research/bandit_tree/config.json"
DEFAULT_OUTPUT = "research/bandit_tree/generated"
HYPOTHESES_PATH = "research/hypotheses.jsonl"
EXPERIMENTS_PATH = "research/experiments.jsonl"
LEGACY_PATH = "research/bandit/arms.json"
TEXT_SUFFIXES = {
    ".c",
    ".cfg",
    ".cpp",
    ".css",
    ".csv",
    ".diff",
    ".html",
    ".htm",
    ".ini",
    ".ipynb",
    ".js",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".patch",
    ".py",
    ".rst",
    ".rs",
    ".sh",
    ".sql",
    ".tex",
    ".toml",
    ".ts",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
TEXT_NAMES = {"dockerfile", "license", "makefile", "readme"}


def _sha256(data: bytes) -> str:
    """Return the SHA-256 digest of exact input bytes."""
    return hashlib.sha256(data).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON number is not allowed: {value}")


def _load_json(data: bytes, source: str) -> dict[str, object]:
    """Load one JSON object and report malformed or non-object input clearly."""
    try:
        value = json.loads(data, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {source}: {error.msg} at line {error.lineno}, column {error.colno}"
        ) from error
    except ValueError as error:
        raise ValueError(f"Invalid JSON in {source}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {source}")
    return value


def _rooted(root: Path, value: str | Path) -> Path:
    """Resolve a configured path and reject paths that escape the repository."""
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes --root: {value}")
    return resolved


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _load_jsonl(data: bytes, path: str, require_id: bool) -> list[dict[str, object]]:
    """Decode JSONL rows while preserving original line numbers and byte hashes."""
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(data.splitlines(keepends=True), start=1):
        if not raw_line.strip():
            continue
        record = _load_json(raw_line, f"{path}:{line_number}")
        record_id = record.get("id")
        if require_id and (not isinstance(record_id, str) or not record_id):
            raise ValueError(f"Missing non-empty string id in {path}:{line_number}")
        rows.append(
            {
                "record": record,
                "raw_record": raw_line.decode("utf-8"),
                "source": {"path": path, "line": line_number, "sha256": _sha256(raw_line)},
            }
        )
    return rows


def _safe_sequence(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def _node_text(record: dict[str, object]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True).casefold()


def _classify_hypothesis(
    record: dict[str, object], families: list[dict[str, object]]
) -> dict[str, object]:
    text = _node_text(record)
    scores: list[tuple[str, int]] = []
    for family in families:
        family_id = family["id"]
        keywords = family.get("keywords", [])
        if not isinstance(family_id, str) or not isinstance(keywords, list):
            continue
        matches = {word for word in keywords if isinstance(word, str) and word.casefold() in text}
        if matches:
            scores.append((family_id, len(matches)))
    if not scores:
        return {"family_id": "family:unclassified", "rule": "no_keyword_match", "matches": []}
    best_score = max(score for _, score in scores)
    matches = [family_id for family_id, score in scores if score == best_score]
    if len(matches) != 1:
        return {
            "family_id": "family:unclassified",
            "rule": "ambiguous_keyword_match",
            "matches": matches,
        }
    return {"family_id": matches[0], "rule": "keyword_match", "matches": matches}


def _catalog(root: Path, documents: list[object], output: Path) -> dict[str, object]:
    """Catalog all configured document roots without following symlinks."""
    files: list[dict[str, object]] = []
    roots: list[str] = []

    def walk(directory: Path) -> None:
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            path = Path(entry.path)
            if path == output or output in path.parents:
                continue
            stat = entry.stat(follow_symlinks=False)
            item: dict[str, object] = {
                "path": _rel(root, path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "mode": stat.st_mode,
            }
            if entry.is_symlink():
                item["type"] = "symlink"
                item["target"] = os.readlink(path)
            elif entry.is_dir(follow_symlinks=False):
                walk(path)
                continue
            elif entry.is_file(follow_symlinks=False):
                item["type"] = "file"
                suffix = path.suffix.casefold()
                is_text_name = (
                    path.name.casefold() in TEXT_NAMES or path.name.casefold().startswith("readme.")
                )
                if stat.st_size <= HASH_LIMIT_BYTES and (suffix in TEXT_SUFFIXES or is_text_name):
                    with path.open("rb") as handle:
                        data = handle.read(HASH_LIMIT_BYTES + 1)
                    try:
                        data.decode("utf-8")
                    except UnicodeDecodeError:
                        item["sha256"] = None
                        item["hash_reason"] = "not_utf8_text"
                    else:
                        item["sha256"] = _sha256(data)
                else:
                    with path.open("rb") as handle:
                        handle.read(0)
                    item["sha256"] = None
                    item["hash_reason"] = (
                        "over_1mb" if stat.st_size > HASH_LIMIT_BYTES else "not_small_text"
                    )
            else:
                item["type"] = "special"
                item["sha256"] = None
                item["hash_reason"] = "non_regular_file"
            files.append(item)

    for document in documents:
        if not isinstance(document, str):
            raise ValueError("documents entries must be repository-relative paths")
        directory = _rooted(root, document)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Document root is missing or is not a directory: {document}")
        roots.append(_rel(root, directory))
        walk(directory)
    files.sort(key=lambda item: str(item["path"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "roots": roots,
        "excluded_output": _rel(root, output),
        "file_count": len(files),
        "files": files,
    }


def _artifact_refs(
    root: Path, artifacts: object, catalog: dict[str, object]
) -> list[dict[str, object]]:
    catalog_files = catalog["files"]
    paths = [
        item["path"]
        for item in catalog_files
        if isinstance(item, dict) and item.get("type") in {"file", "symlink"}
    ]
    refs: list[dict[str, object]] = []
    for artifact in _safe_sequence(artifacts):
        if not isinstance(artifact, str):
            refs.append({"path": str(artifact), "status": "missing", "candidates": []})
            continue
        try:
            resolved = _rooted(root, artifact)
            rel_path = _rel(root, resolved)
        except ValueError:
            rel_path = artifact
            resolved = root
        if rel_path in paths and (resolved.is_file() or resolved.is_symlink()):
            refs.append({"path": artifact, "status": "resolved", "candidates": [rel_path]})
            continue
        candidates = [path for path in paths if Path(str(path)).name == Path(artifact).name]
        refs.append(
            {
                "path": artifact,
                "status": "ambiguous" if len(candidates) > 1 else "missing",
                "candidates": candidates,
            }
        )
    return refs


def _edge(
    source_id: str, target_id: object, relation: str, node_ids: set[str], source: object = None
) -> dict[str, object]:
    target = target_id if isinstance(target_id, str) else str(target_id)
    result: dict[str, object] = {
        "source_id": source_id,
        "target_id": target,
        "relation": relation,
        "resolved": target in node_ids,
    }
    if source is not None:
        result["source"] = source
    return result


def _validate_tree(tree: dict[str, object]) -> None:
    nodes = tree.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("tree.json has no nodes list")
    ids = [node.get("id") for node in nodes if isinstance(node, dict)]
    if len(ids) != len(nodes) or not all(isinstance(node_id, str) for node_id in ids):
        raise ValueError("Every tree node must have a string id")
    if len(ids) != len(set(ids)):
        raise ValueError("Tree node ids are not unique")
    node_map = {node["id"]: node for node in nodes}
    root_id = tree.get("root_id")
    if root_id not in node_map or node_map[root_id].get("type") != "root":
        raise ValueError("Tree must contain its declared root node")
    for node in nodes:
        node_id = node["id"]
        parent_id = node.get("parent_id")
        if node_id == root_id:
            if parent_id is not None:
                raise ValueError("Root node cannot have a parent")
        elif parent_id not in node_map:
            raise ValueError(f"Missing parent {parent_id!r} for node {node_id}")
    state: dict[str, int] = {}

    def visit(node_id: str) -> None:
        if state.get(node_id) == 1:
            raise ValueError(f"Cycle in parent tree at {node_id}")
        if state.get(node_id) == 2:
            return
        state[node_id] = 1
        parent_id = node_map[node_id].get("parent_id")
        if isinstance(parent_id, str):
            visit(parent_id)
        state[node_id] = 2

    for node_id in node_map:
        visit(node_id)


def _build_documents(root: Path, config_path: Path, output: Path) -> dict[str, bytes]:
    config_bytes = config_path.read_bytes()
    config = _load_json(config_bytes, _rel(root, config_path))
    if config.get("version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported config version: {config.get('version')!r}")
    root_meta = config.get("root")
    families_raw = config.get("families")
    curated_raw = config.get("curated_nodes")
    if (
        not isinstance(root_meta, dict)
        or not isinstance(families_raw, list)
        or not isinstance(curated_raw, list)
    ):
        raise ValueError("Config must contain root, families, and curated_nodes")
    if root_meta.get("id") != "root":
        raise ValueError("Config root.id must be 'root'")
    families = [item for item in families_raw if isinstance(item, dict)]
    curated = [item for item in curated_raw if isinstance(item, dict)]
    if len(families) != len(families_raw) or len(curated) != len(curated_raw):
        raise ValueError("families and curated_nodes entries must be objects")

    input_paths = [HYPOTHESES_PATH, EXPERIMENTS_PATH, LEGACY_PATH]
    journal_value = config.get("journal")
    journal_rel: str | None = None
    if journal_value is not None:
        if not isinstance(journal_value, str):
            raise ValueError("journal must be a repository-relative path")
        journal_path = _rooted(root, journal_value)
        journal_rel = _rel(root, journal_path)
        input_paths.append(journal_rel)
    source_bytes: dict[str, bytes] = {_rel(root, config_path): config_bytes}
    for path in input_paths:
        source_bytes[path] = _rooted(root, path).read_bytes()
    hypothesis_rows = _load_jsonl(source_bytes[HYPOTHESES_PATH], HYPOTHESES_PATH, True)
    experiment_rows = _load_jsonl(source_bytes[EXPERIMENTS_PATH], EXPERIMENTS_PATH, True)
    journal_rows = _load_jsonl(source_bytes[journal_rel], journal_rel, False) if journal_rel else []
    legacy = _load_json(source_bytes[LEGACY_PATH], LEGACY_PATH)
    old_arms = legacy.get("arms")
    if not isinstance(old_arms, dict):
        raise ValueError(f"Expected an arms object in {LEGACY_PATH}")

    documents = config.get("documents")
    if not isinstance(documents, list):
        raise ValueError("Config documents must be a list")
    catalog = _catalog(root, documents, output)
    catalog_bytes = _json_bytes(catalog)
    catalog_paths = {
        item["path"]
        for item in catalog["files"]
        if item.get("type") in {"file", "symlink"}
    }

    nodes: list[dict[str, object]] = []
    root_node: dict[str, object] = {
        "id": "root",
        "type": "root",
        "title": root_meta.get("title", "Research"),
        "parent_id": None,
        "metadata": root_meta,
        "source": {"path": _rel(root, config_path), "sha256": _sha256(config_bytes)},
    }
    nodes.append(root_node)
    for family in families:
        family_id = family.get("id")
        if not isinstance(family_id, str) or not family_id.startswith("family:"):
            raise ValueError(f"Family ids must start with 'family:': {family_id!r}")
        nodes.append(
            {
                "id": family_id,
                "type": "family",
                "title": family.get("title", family_id),
                "parent_id": "root",
                "keywords": family.get("keywords", []),
                "source": {"path": _rel(root, config_path), "sha256": _sha256(config_bytes)},
            }
        )
    curated_ids: set[str] = set()
    curated_hypotheses: dict[str, list[str]] = defaultdict(list)
    curated_experiments: dict[str, list[str]] = defaultdict(list)
    for item in curated:
        node_id = item.get("id")
        kind = item.get("kind")
        if not isinstance(node_id, str) or kind not in {"arm", "lineage"}:
            raise ValueError(f"Invalid curated node id or kind: {node_id!r}, {kind!r}")
        if (kind == "arm" and not node_id.startswith("arm:")) or (
            kind == "lineage" and not node_id.startswith("lineage:")
        ):
            raise ValueError(f"Curated id prefix does not match kind: {node_id}")
        if node_id in curated_ids:
            raise ValueError(f"Duplicate curated node id: {node_id}")
        curated_ids.add(node_id)
        for hypothesis_id in _safe_sequence(item.get("hypothesis_ids")):
            if isinstance(hypothesis_id, str):
                curated_hypotheses[hypothesis_id].append(node_id)
        for experiment_id in _safe_sequence(item.get("experiment_ids")):
            if isinstance(experiment_id, str):
                curated_experiments[experiment_id].append(node_id)
        current_override = item.get("current_override") is True
        allocation = item.get("allocation")
        if not isinstance(allocation, dict):
            if kind == "arm":
                raise ValueError(f"Curated arm {node_id} needs an allocation object")
            allocation = {"eligible": False, "priority": 0, "launch_authorized": False}
        effective_allocation = dict(allocation)
        if effective_allocation.get("launch_authorized") is True:
            raise ValueError(f"Curated node {node_id} cannot authorize a launch")
        effective_allocation["launch_authorized"] = False
        legacy_id = item.get("legacy_id")
        stale_legacy = isinstance(legacy_id, str) and not current_override
        if stale_legacy:
            effective_allocation["eligible"] = False
            effective_allocation["reason"] = (
                str(effective_allocation.get("reason", ""))
                + " Historical arm requires an explicit current_override."
            ).strip()
        node: dict[str, object] = dict(item)
        node.update(
            {
                "type": kind,
                "parent_id": item.get("parent_id"),
                "allocation_config": allocation,
                "allocation": effective_allocation,
                "current_override": current_override,
                "legacy_reference_stale": stale_legacy,
                "source": {"path": _rel(root, config_path), "sha256": _sha256(config_bytes)},
                "artifact_refs": _artifact_refs(root, item.get("artifacts", []), catalog),
            }
        )
        nodes.append(node)

    hypothesis_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in hypothesis_rows:
        record = row["record"]
        assert isinstance(record, dict)
        hypothesis_groups[str(record["id"])].append(row)
    experiment_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in experiment_rows:
        record = row["record"]
        assert isinstance(record, dict)
        experiment_groups[str(record["id"])].append(row)

    edges: list[dict[str, object]] = []
    warnings: list[str] = []
    for hypothesis_id, rows in hypothesis_groups.items():
        explicit = curated_hypotheses.get(hypothesis_id, [])
        classification = _classify_hypothesis(rows[-1]["record"], families)
        if len(explicit) == 1:
            parent_id = explicit[0]
            classification = {"family_id": None, "rule": "curated_membership", "matches": explicit}
        elif len(explicit) > 1:
            parent_id = "family:unclassified"
            classification = {
                "family_id": parent_id,
                "rule": "ambiguous_curated_membership",
                "matches": explicit,
            }
            warnings.append(
                f"Hypothesis {hypothesis_id} belongs to multiple curated nodes: {', '.join(explicit)}"
            )
        else:
            parent_id = classification["family_id"]
        latest_record = rows[-1]["record"]
        assert isinstance(latest_record, dict)
        node = {
            "id": f"hypothesis:{hypothesis_id}",
            "type": "hypothesis",
            "title": str(latest_record.get("statement", hypothesis_id)),
            "parent_id": parent_id,
            "classification": classification,
            "latest_recorded_status": latest_record.get("status"),
            "raw_rows": rows,
            "evidence_refs": [],
        }
        nodes.append(node)
        for row in rows:
            record = row["record"]
            assert isinstance(record, dict)
            source = row["source"]
            for parent in _safe_sequence(record.get("parent_ids")):
                target = (
                    f"hypothesis:{parent}"
                    if isinstance(parent, str) and parent in hypothesis_groups
                    else parent
                )
                edges.append(_edge(node["id"], target, "ledger_parent_id", set(), source))
            for evidence in _safe_sequence(record.get("evidence")):
                if not isinstance(evidence, str):
                    node["evidence_refs"].append(
                        {"value": evidence, "kind": "raw_value", "source": source}
                    )
                elif evidence in experiment_groups or re.fullmatch(
                    r"E-[A-Za-z0-9][A-Za-z0-9_-]*", evidence
                ):
                    target = f"experiment:{evidence}"
                    edges.append(
                        _edge(node["id"], target, "ledger_evidence_reference", set(), source)
                    )
                    node["evidence_refs"].append(
                        {
                            "value": evidence,
                            "kind": "experiment",
                            "target_id": target,
                            "resolved": evidence in experiment_groups,
                            "source": source,
                        }
                    )
                elif evidence in catalog_paths:
                    edges.append(
                        {
                            "source_id": node["id"],
                            "target_path": evidence,
                            "relation": "ledger_evidence_document",
                            "resolved": True,
                            "source": source,
                        }
                    )
                    node["evidence_refs"].append(
                        {
                            "value": evidence,
                            "kind": "document",
                            "resolved": True,
                            "source": source,
                        }
                    )
                elif evidence.startswith(("research/", "reflect_05_day/")) or (
                    Path(evidence).suffix.casefold() in TEXT_SUFFIXES
                ):
                    candidates = [
                        path for path in catalog_paths if Path(path).name == Path(evidence).name
                    ]
                    edges.append(
                        {
                            "source_id": node["id"],
                            "target_path": evidence,
                            "relation": "ledger_evidence_document",
                            "resolved": evidence in catalog_paths,
                            "source": source,
                        }
                    )
                    node["evidence_refs"].append(
                        {
                            "value": evidence,
                            "kind": "document",
                            "status": "resolved"
                            if evidence in catalog_paths
                            else "ambiguous"
                            if len(candidates) > 1
                            else "missing",
                            "candidates": candidates,
                            "source": source,
                        }
                    )
                else:
                    node["evidence_refs"].append(
                        {"value": evidence, "kind": "note", "source": source}
                    )

    valid_hypotheses = set(hypothesis_groups)
    for experiment_id, rows in experiment_groups.items():
        explicit = curated_experiments.get(experiment_id, [])
        latest_record = rows[-1]["record"]
        assert isinstance(latest_record, dict)
        linked = [
            value
            for row in rows
            for value in _safe_sequence(row["record"].get("hypotheses"))
            if isinstance(value, str)
        ]
        valid_linked = sorted(set(linked) & valid_hypotheses)
        if len(explicit) == 1:
            parent_id = explicit[0]
            classification = {"rule": "curated_membership", "matches": explicit}
        elif len(explicit) > 1:
            parent_id = "family:unclassified"
            classification = {"rule": "ambiguous_curated_membership", "matches": explicit}
            warnings.append(
                f"Experiment {experiment_id} belongs to multiple curated nodes: {', '.join(explicit)}"
            )
        elif len(valid_linked) == 1:
            parent_id = f"hypothesis:{valid_linked[0]}"
            classification = {"rule": "single_linked_hypothesis", "matches": valid_linked}
        else:
            parent_id = "family:unclassified"
            classification = {
                "rule": "multiple_or_missing_linked_hypotheses",
                "matches": valid_linked,
            }
        nodes.append(
            {
                "id": f"experiment:{experiment_id}",
                "type": "experiment",
                "title": str(latest_record.get("kind", experiment_id)),
                "parent_id": parent_id,
                "classification": classification,
                "latest_recorded_status": latest_record.get("status"),
                "raw_rows": rows,
            }
        )
        for row in rows:
            record = row["record"]
            assert isinstance(record, dict)
            for hypothesis_id in _safe_sequence(record.get("hypotheses")):
                target = (
                    f"hypothesis:{hypothesis_id}"
                    if isinstance(hypothesis_id, str)
                    else hypothesis_id
                )
                edges.append(
                    _edge(
                        f"experiment:{experiment_id}",
                        target,
                        "ledger_hypothesis_reference",
                        set(),
                        row["source"],
                    )
                )

    legacy_source = {"path": LEGACY_PATH, "sha256": _sha256(source_bytes[LEGACY_PATH])}
    legacy_nodes: list[dict[str, object]] = []
    for legacy_id, raw_arm in old_arms.items():
        node_id = f"legacy:{legacy_id}"
        legacy_nodes.append(
            {
                "id": node_id,
                "type": "legacy_arm",
                "title": str(raw_arm.get("title", legacy_id))
                if isinstance(raw_arm, dict)
                else str(legacy_id),
                "parent_id": "root",
                "status": "historical_needs_refresh",
                "allocation": {
                    "eligible": False,
                    "reason": "Stale historical prior; refresh and review explicitly before current ranking.",
                    "launch_authorized": False,
                },
                "raw_arm": raw_arm,
                "source": legacy_source,
            }
        )
    nodes.extend(legacy_nodes)
    node_ids = {str(node["id"]) for node in nodes}

    for item in curated:
        source_id = str(item["id"])
        for hypothesis_id in _safe_sequence(item.get("hypothesis_ids")):
            target_id = (
                f"hypothesis:{hypothesis_id}" if isinstance(hypothesis_id, str) else hypothesis_id
            )
            edges.append(_edge(source_id, target_id, "curated_hypothesis_membership", node_ids))
        for experiment_id in _safe_sequence(item.get("experiment_ids")):
            target_id = (
                f"experiment:{experiment_id}" if isinstance(experiment_id, str) else experiment_id
            )
            edges.append(_edge(source_id, target_id, "curated_experiment_membership", node_ids))
        for related_id in _safe_sequence(item.get("related_nodes")):
            edges.append(_edge(source_id, related_id, "curated_related_node", node_ids))
    for link in _safe_sequence(config.get("links")):
        if not isinstance(link, dict):
            raise ValueError("links entries must be objects")
        source_id = link.get("source_id", link.get("from"))
        target_id = link.get("target_id", link.get("to"))
        relation = link.get("relation", "configured_link")
        if not isinstance(source_id, str) or not isinstance(relation, str):
            raise ValueError(f"Invalid configured link: {link!r}")
        edges.append(_edge(source_id, target_id, relation, node_ids))
    for edge in edges:
        if isinstance(edge.get("target_path"), str):
            edge["resolved"] = (
                edge["source_id"] in node_ids and edge["target_path"] in catalog_paths
            )
            continue
        source_id = edge["source_id"]
        if isinstance(source_id, str) and not edge["resolved"]:
            if source_id in hypothesis_groups:
                edge["source_id"] = f"hypothesis:{source_id}"
            elif source_id in experiment_groups:
                edge["source_id"] = f"experiment:{source_id}"
        target_id = edge["target_id"]
        if isinstance(target_id, str) and not edge["resolved"]:
            if target_id in hypothesis_groups:
                edge["target_id"] = f"hypothesis:{target_id}"
            elif target_id in experiment_groups:
                edge["target_id"] = f"experiment:{target_id}"
        edge["resolved"] = edge["source_id"] in node_ids and edge["target_id"] in node_ids
    for edge in edges:
        if not edge["resolved"]:
            warnings.append(
                f"Unresolved {edge['relation']} link: {edge['source_id']} -> "
                f"{edge.get('target_id', edge.get('target_path', 'unknown'))}"
            )
    for node in nodes:
        for artifact in node.get("artifact_refs", []):
                if isinstance(artifact, dict) and artifact.get("status") != "resolved":
                    warnings.append(
                        f"{artifact['status']} artifact for {node['id']}: {artifact['path']}"
                    )
        for evidence in node.get("evidence_refs", []):
            if isinstance(evidence, dict) and evidence.get("status") in {"missing", "ambiguous"}:
                warnings.append(
                    f"{evidence['status']} evidence document for {node['id']}: {evidence['value']}"
                )

    snapshot_name = f"snapshots/config-{_sha256(config_bytes)}.json"
    snapshot_path = output / snapshot_name
    tree: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "root_id": "root",
        "metadata": {
            "root": root_meta,
            "raw_config": config,
            "classification_policy": config.get("classification_policy"),
            "allocation_policy": config.get("allocation_policy"),
            "config_snapshot": _rel(root, snapshot_path),
            "journal_records": journal_rows,
            "legacy_archive": {
                "source": legacy_source,
                "raw": legacy,
            },
            "source_rows": {
                "hypotheses": len(hypothesis_rows),
                "experiments": len(experiment_rows),
                "journal": len(journal_rows),
            },
            "exposure_note": root_meta.get("evaluation_exposure"),
        },
        "nodes": nodes,
        "related_edges": edges,
        "warnings": sorted(set(warnings)),
    }
    _validate_tree(tree)
    coverage_rows = []
    for node in nodes:
        if node.get("type") in {"hypothesis", "experiment"}:
            coverage_rows.extend(
                (row["source"]["path"], row["source"]["line"], row["source"]["sha256"])
                for row in node["raw_rows"]
            )
    coverage_rows.extend(
        (row["source"]["path"], row["source"]["line"], row["source"]["sha256"])
        for row in journal_rows
    )
    coverage_rows.sort()
    coverage = {
        "hypothesis_rows": len(hypothesis_rows),
        "hypothesis_ids": len(hypothesis_groups),
        "experiment_rows": len(experiment_rows),
        "experiment_ids": len(experiment_groups),
        "journal_rows": len(journal_rows),
        "legacy_arms": len(old_arms),
        "ledger_row_manifest_sha256": _sha256(_json_bytes(coverage_rows)),
    }
    input_manifest = [
        {"path": path, "size_bytes": len(data), "sha256": _sha256(data)}
        for path, data in sorted(source_bytes.items())
    ]
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": _sha256(config_bytes),
        "config_snapshot": _rel(root, snapshot_path),
        "input_manifest": input_manifest,
        "catalog_file_count": catalog["file_count"],
        "catalog_sha256": _sha256(catalog_bytes),
        "coverage": coverage,
        "node_count": len(nodes),
        "related_edge_count": len(edges),
    }
    index = _build_index(root, output, tree, integrity, catalog)
    return {
        "tree.json": _json_bytes(tree),
        "INDEX.md": index.encode("utf-8"),
        "integrity.json": _json_bytes(integrity),
        "catalog.json": catalog_bytes,
    }


def _markdown_link(root: Path, output: Path, path: str, line: int | None = None) -> str:
    target_path = os.path.relpath(root / path, output)
    target = quote(Path(target_path).as_posix(), safe="/#")
    if line is not None:
        target += f"#L{line}"
    return f"[{path}](<{target}>)"


def _build_index(
    root: Path,
    output: Path,
    tree: dict[str, object],
    integrity: dict[str, object],
    catalog: dict[str, object],
) -> str:
    nodes = tree["nodes"]
    families = [node for node in nodes if node.get("type") == "family"]
    curated = [node for node in nodes if node.get("type") in {"arm", "lineage"}]
    children: dict[str, list[dict[str, object]]] = defaultdict(list)
    for node in curated:
        if isinstance(node.get("parent_id"), str):
            children[node["parent_id"]].append(node)
    lines = [
        "# Research bandit tree",
        "",
        "Generated navigational index. Original ledgers and legacy bandit remain archived; every ledger event is retained with its original line bytes and SHA-256.",
        "",
        "Statuses are displayed as recorded or curated and have not been independently verified by this index. Validation, holdout, and fold data remain exposed. Links describe recorded relationships, not causal effects. Unknown priors and costs remain unknown; no score is inferred.",
        "",
        f"Root: **{tree['metadata']['root']['title']}** · metric `{tree['metadata']['root'].get('metric_id', 'unknown')}` · nodes {integrity['node_count']} · raw hypothesis rows {integrity['coverage']['hypothesis_rows']} · raw experiment rows {integrity['coverage']['experiment_rows']} · legacy arms {integrity['coverage']['legacy_arms']} · catalog files {catalog['file_count']}.",
        "",
        f"Config snapshot: {_markdown_link(root, output, str(integrity['config_snapshot']))}.",
        "",
        "Missing or ambiguous artifact and relationship links are called out below and in `tree.json` warnings. The recursive file inventory is in `catalog.json`; generated output is excluded.",
        "",
    ]

    def render_curated(node: dict[str, object], depth: int) -> None:
        marker = "  " * depth + "- "
        status = str(node.get("status", "status unknown"))
        allocation = node.get("allocation", {})
        status += f"; eligible={allocation.get('eligible', False)}"
        title = node.get("title", node["id"])
        lines.append(f"{marker}**{title}** (`{node['id']}`) — {status}.")
        for field in ("next_test", "status_scope", "claim_scope"):
            value = node.get(field)
            if value:
                lines.append(f"{'  ' * (depth + 1)}- **{field}:** {value}")
        evidence_states = node.get("evidence_states")
        if evidence_states is not None:
            if isinstance(evidence_states, list):
                for state in evidence_states:
                    if not isinstance(state, dict):
                        continue
                    source = state.get("source")
                    source_text = (
                        _markdown_link(root, output, source)
                        if isinstance(source, str)
                        else str(source or "source unknown")
                    )
                    lines.append(
                        f"{'  ' * (depth + 1)}- **evidence state:** "
                        f"{state.get('scope', 'scope unknown')} — {state.get('state', 'state unknown')} "
                        f"(source: {source_text})"
                    )
            else:
                compact = json.dumps(evidence_states, ensure_ascii=False, sort_keys=True)
                lines.append(f"{'  ' * (depth + 1)}- **evidence_states:** `{compact}`")
        for field in ("planned_experiment_ids", "registration_note"):
            value = node.get(field)
            if value:
                lines.append(f"{'  ' * (depth + 1)}- **{field}:** {value}")
        for artifact in node.get("artifact_refs", []):
            if not isinstance(artifact, dict):
                continue
            if artifact.get("status") == "resolved":
                value = _markdown_link(root, output, str(artifact["path"]))
            else:
                value = f"{artifact['path']} ({artifact['status']})"
            lines.append(f"{'  ' * (depth + 1)}- artifact: {value}")
        for child in children.get(str(node["id"]), []):
            render_curated(child, depth + 1)

    lines.extend(["## Curated decision tree", ""])
    for family in families:
        lines.extend([f"### {family['title']} (`{family['id']}`)", ""])
        for node in children.get(str(family["id"]), []):
            render_curated(node, 0)
        if not children.get(str(family["id"])):
            lines.append("No curated nodes.")
        lines.append("")

    lines.extend(
        [
            "## Node and source index",
            "",
            "Each raw ledger event appears as its own source link, including repeated IDs and status updates.",
            "",
            "| Node ID | Title | Type / latest recorded status | Parent | Source rows | Evidence refs |",
            "|---|---|---|---|---|---|",
        ]
    )
    for node in nodes:
        raw_sources: list[str] = []
        for row in node.get("raw_rows", []):
            source = row.get("source", {})
            raw_sources.append(_markdown_link(root, output, source["path"], source.get("line")))
        source = node.get("source", {})
        if not raw_sources and isinstance(source, dict) and isinstance(source.get("path"), str):
            raw_sources.append(_markdown_link(root, output, source["path"]))
        status = node.get("latest_recorded_status", node.get("status", ""))
        parent = node.get("parent_id") or "—"
        evidence_refs = []
        for evidence in node.get("evidence_refs", []):
            if not isinstance(evidence, dict):
                continue
            value = str(evidence.get("value", ""))
            if evidence.get("kind") == "document" and evidence.get("status", "resolved") == "resolved":
                evidence_refs.append(_markdown_link(root, output, value))
            elif evidence.get("kind") == "experiment":
                evidence_refs.append(f"{evidence.get('target_id', value)} ({'linked' if evidence.get('resolved') else 'missing'})")
            elif evidence.get("kind") == "document":
                evidence_refs.append(f"{value} ({evidence.get('status', 'missing')})")
            else:
                evidence_refs.append(value)
        fields = [
            str(node["id"]).replace("|", "\\|"),
            str(node.get("title", "")).replace("|", "\\|"),
            str(node.get("type", "")) + (f" · {status}" if status else ""),
            str(parent).replace("|", "\\|"),
            ", ".join(raw_sources) or "—",
            ", ".join(evidence_refs) or "—",
        ]
        lines.append("| " + " | ".join(field.replace("\n", " ") for field in fields) + " |")
    journal_rows = tree["metadata"].get("journal_records", [])
    lines.extend(["", "## Journal events", ""])
    if journal_rows:
        for row in journal_rows:
            source = row["source"]
            summary = json.dumps(row["record"], ensure_ascii=False, sort_keys=True)
            lines.append(
                f"- {_markdown_link(root, output, source['path'], source['line'])} "
                f"`sha256:{source['sha256']}` — `{summary}`"
            )
    else:
        lines.append("No journal events are recorded.")
    lines.extend(["", "## Link and artifact warnings", ""])
    warnings = tree.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("No unresolved configured links or artifacts were found.")
    lines.extend(
        [
            "",
            "## Config snapshot fields",
            "",
            "The parsed config is retained in `tree.json` metadata and the exact config bytes are archived in the linked snapshot.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_snapshot(output: Path, config_sha: str, config_bytes: bytes) -> None:
    snapshot = output / "snapshots" / f"config-{config_sha}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        if snapshot.read_bytes() != config_bytes:
            raise ValueError(f"Config snapshot hash collision or modified snapshot: {snapshot}")
        return
    with snapshot.open("xb") as handle:
        handle.write(config_bytes)


def build(root: Path, config: Path, output: Path) -> int:
    """Generate the tree, index, catalog, and integrity receipt."""
    documents = _build_documents(root, config, output)
    config_bytes = config.read_bytes()
    output.mkdir(parents=True, exist_ok=True)
    _write_snapshot(output, _sha256(config_bytes), config_bytes)
    for name, data in documents.items():
        with tempfile.NamedTemporaryFile(dir=output, prefix=f".{name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        os.replace(temporary, output / name)
    print(
        f"Built {len(documents)} generated files in {_rel(root, output)}; "
        f"config snapshot sha256={_sha256(config_bytes)}"
    )
    return 0


def check(root: Path, config: Path, output: Path) -> int:
    """Validate parent structure, source coverage, snapshots, and freshness."""
    try:
        expected = _build_documents(root, config, output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"check failed: {error}", file=sys.stderr)
        return 1
    expected_tree = json.loads(expected["tree.json"])
    _validate_tree(expected_tree)
    actual_tree_path = output / "tree.json"
    if not actual_tree_path.is_file():
        print(f"check failed: missing {_rel(root, actual_tree_path)}", file=sys.stderr)
        return 1
    try:
        actual_tree = _load_json(actual_tree_path.read_bytes(), _rel(root, actual_tree_path))
        _validate_tree(actual_tree)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"check failed: invalid generated tree: {error}", file=sys.stderr)
        return 1
    expected_rows: Counter[tuple[object, ...]] = Counter()
    actual_rows: Counter[tuple[object, ...]] = Counter()
    for source_tree, counter in ((expected_tree, expected_rows), (actual_tree, actual_rows)):
        for node in source_tree.get("nodes", []):
            if isinstance(node, dict) and node.get("type") in {"hypothesis", "experiment"}:
                for row in node.get("raw_rows", []):
                    source = row.get("source", {})
                    raw_record = row.get("raw_record")
                    if not isinstance(raw_record, str) or _sha256(
                        raw_record.encode("utf-8")
                    ) != source.get("sha256"):
                        print(
                            "check failed: a raw ledger row differs from its source SHA",
                            file=sys.stderr,
                        )
                        return 1
                    counter[(source.get("path"), source.get("line"), source.get("sha256"))] += 1
        for row in source_tree.get("metadata", {}).get("journal_records", []):
            source = row.get("source", {})
            raw_record = row.get("raw_record")
            if not isinstance(raw_record, str) or _sha256(raw_record.encode("utf-8")) != source.get(
                "sha256"
            ):
                print(
                    "check failed: a raw journal row differs from its source SHA", file=sys.stderr
                )
                return 1
            counter[(source.get("path"), source.get("line"), source.get("sha256"))] += 1
    if expected_rows != actual_rows:
        print("check failed: ledger row coverage or source hashes differ", file=sys.stderr)
        return 1
    stale: list[str] = []
    for name, expected_bytes in expected.items():
        path = output / name
        if not path.is_file() or path.read_bytes() != expected_bytes:
            stale.append(name)
    integrity = json.loads(expected["integrity.json"])
    snapshot = _rooted(root, str(integrity["config_snapshot"]))
    try:
        if snapshot.read_bytes() != config.read_bytes():
            stale.append(str(integrity["config_snapshot"]))
    except OSError:
        stale.append(str(integrity["config_snapshot"]))
    if stale:
        print(
            "check failed: missing or stale generated files: " + ", ".join(stale), file=sys.stderr
        )
        return 1
    print(
        f"Check passed: {len(actual_tree['nodes'])} unique nodes, "
        f"{len(expected_rows)} preserved ledger rows, parents acyclic, inputs current."
    )
    return 0


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _numeric_kg(node: dict[str, object]) -> tuple[float, float] | None:
    belief = node.get("belief")
    if not isinstance(belief, dict) or belief.get("reviewed") is not True:
        return None
    if belief.get("unit") != "absolute_global_wp_delta":
        return None
    mu = _number(belief.get("mu"))
    sd = _number(belief.get("sd"))
    noise_sd = _number(belief.get("noise_sd"))
    cost = _number(node.get("cost_h"))
    if mu is None or sd is None or noise_sd is None or cost is None:
        return None
    if sd < 0 or noise_sd < 0 or cost <= 0:
        return None
    posterior_mean_sd = (
        sd * sd / math.sqrt(sd * sd + noise_sd * noise_sd) if sd or noise_sd else 0.0
    )
    return mu, posterior_mean_sd


def _expected_max(baseline: float, mu: float, sd: float) -> float:
    if sd == 0:
        return max(baseline, mu)
    z = (mu - baseline) / sd
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    density = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return baseline * (1 - cdf) + mu * cdf + sd * density


def _family_for(node: dict[str, object], node_map: dict[str, dict[str, object]]) -> str:
    parent_id = node.get("parent_id")
    seen: set[str] = set()
    while isinstance(parent_id, str) and parent_id not in seen:
        seen.add(parent_id)
        parent = node_map.get(parent_id)
        if parent is None:
            break
        if parent.get("type") == "family":
            return parent_id
        parent_id = parent.get("parent_id")
    return "family:unclassified"


def rank(tree: dict[str, object], limit: int | None, max_per_family: int | None) -> int:
    """Print eligible curated arms with numeric KG or honest manual priority."""
    nodes = tree["nodes"]
    node_map = {node["id"]: node for node in nodes}
    eligible = [
        node
        for node in nodes
        if node.get("type") == "arm"
        and isinstance(node.get("allocation"), dict)
        and node["allocation"].get("eligible") is True
    ]
    scored: dict[str, tuple[float, float]] = {}
    for node in eligible:
        value = _numeric_kg(node)
        if value is not None and (
            not node.get("legacy_reference_stale") or node.get("current_override")
        ):
            scored[node["id"]] = value
    means = {node_id: value[0] for node_id, value in scored.items()}
    numeric: list[tuple[float, int, str, dict[str, object], float]] = []
    manual: list[tuple[int, str, dict[str, object]]] = []
    for node in eligible:
        allocation = node["allocation"]
        priority = allocation.get("priority")
        priority_value = (
            priority if isinstance(priority, int) and not isinstance(priority, bool) else -1
        )
        if node["id"] in scored:
            mu, posterior_sd = scored[node["id"]]
            other_best = max(
                [0.0, *(mean for other_id, mean in means.items() if other_id != node["id"])]
            )
            kg = max(0.0, _expected_max(other_best, mu, posterior_sd) - max(other_best, mu))
            cost = _number(node.get("cost_h"))
            assert cost is not None and cost > 0
            numeric.append((kg / cost, priority_value, node["id"], node, cost))
        else:
            manual.append((priority_value, node["id"], node))
    numeric.sort(key=lambda entry: (-entry[0], -entry[1], entry[2]))
    manual.sort(key=lambda entry: (-entry[0], entry[1]))
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    if max_per_family is not None and max_per_family < 1:
        raise ValueError("--max-per-family must be at least 1")

    def diversify(entries: list[tuple[object, ...]]) -> list[tuple[object, ...]]:
        selected: list[tuple[object, ...]] = []
        family_counts: Counter[str] = Counter()
        for entry in entries:
            node = entry[-2] if len(entry) == 5 else entry[-1]
            family_id = _family_for(node, node_map)
            if max_per_family is not None and family_counts[family_id] >= max_per_family:
                continue
            family_counts[family_id] += 1
            selected.append(entry)
            if limit is not None and len(selected) >= limit:
                break
        return selected

    print(
        "Eligible curated arms only; eligibility means next-decision/preparation, not launch authorization."
    )
    if numeric:
        print(
            "Numeric knowledge gradient per declared cost hour (reviewed absolute global-WP belief):"
        )
        for position, entry in enumerate(diversify(numeric), start=1):
            kg_per_hour, _, node_id, node, cost = entry
            print(
                f"{position}. {node_id} | {_family_for(node, node_map)} | "
                f"KG/cost={kg_per_hour:.8g} | cost_h={cost:g} | "
                f"launch_authorized={node['allocation'].get('launch_authorized', False)}"
            )
    if manual:
        print(
            "Manual priority only; expected public gain is unknown and this section is incomparable to KG:"
        )
        for position, entry in enumerate(diversify(manual), start=1):
            priority, node_id, node = entry
            print(
                f"{position}. {node_id} | {_family_for(node, node_map)} | "
                f"manual_priority={priority} | expected_public_gain=unknown | "
                f"launch_authorized={node['allocation'].get('launch_authorized', False)}"
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run build, check, or rank."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="repository root")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="versioned tree config path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="generated output directory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="generate tree, index, catalog, and integrity receipt")
    subparsers.add_parser("check", help="validate generated files and their source freshness")
    rank_parser = subparsers.add_parser("rank", help="rank eligible curated arms")
    rank_parser.add_argument("--limit", type=int, help="maximum rows in each ranking section")
    rank_parser.add_argument("--max-per-family", type=int, help="optional family diversity ceiling")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"--root is not a directory: {root}")
        config = _rooted(root, args.config)
        output = _rooted(root, args.output)
        if args.command == "build":
            return build(root, config, output)
        if args.command == "check":
            return check(root, config, output)
        if check(root, config, output) != 0:
            return 1
        tree_path = output / "tree.json"
        tree = _load_json(tree_path.read_bytes(), _rel(root, tree_path))
        return rank(tree, args.limit, args.max_per_family)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"research_tree: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
