#!/usr/bin/env python3
"""Build a filename-only research review index and stat-only artifact inventory.

The index includes tracked research evidence plus eligible current research
documents. File contents are never opened. The separate numerical-artifact
section reads filesystem metadata only: no artifact bytes or hashes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

ROOT = Path(__file__).resolve().parents[1]
INDEX_MD = Path("research/REVIEW_FILE_INDEX_2026-10-05.md")
INDEX_JSON = Path("research/REVIEW_FILE_INDEX_2026-10-05.json")
EVIDENCE_SUFFIXES = {
    ".md",
    ".json",
    ".jsonl",
    ".txt",
    ".png",
    ".svg",
    ".pdf",
    ".html",
    ".ipynb",
    ".csv",
}
CODE_SUFFIXES = {".py", ".rs"}
SELECTED_TOP_LEVEL = {
    "README.md",
    "METRIC.md",
    "WunderFunder_audit_evidence.md",
    "utils.py",
    "scripts/train_rnn.py",
    "scripts/predict_run.py",
    "scripts/package_submission.py",
    "scripts/screen_score.py",
    "scripts/run_campaign.py",
    "scripts/build_review_index.py",
    "submission/solution_multi.py",
}
CANONICAL_CODE_PREFIXES = ("src/wnn/", "native/src/")
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "site-packages",
    "vendor",
    "target",
    "build",
    "dist",
    "code_snapshots",
    "bundle_source",
    "runtime",
    "runtime_vendor",
    "prepared_v1",
    "prepared_v2",
    "prepared_v3",
    "remote_v1",
    "remote_v2",
    "remote_v3",
    "source_root",
    "provider_dataset_download",
}
EXCLUDED_PREFIXES = (
    "research/campaign_2026-10-03/continuation/review/pause_after_exit/",
    "research/campaign_2026-10-03/continuation/review/failed_launch/",
    "research/campaign_2026-10-03/continuation/review/nonzero_job/",
    "research/campaign_2026-10-03/continuation/review/successful_queue/",
    "research/campaign_2026-10-03/continuation/review/restart_failed/",
    "research/campaign_2026-10-03/continuation/review/finished_nonzero/",
    "research/campaign_2026-10-03/continuation/review/stale_running/",
    "research/campaign_2026-10-03/continuation/review/journal_probe/",
    "research/campaign_2026-10-03/continuation/review/deadline_probe/",
    "research/campaign_2026-10-03/continuation/review/malformed_startup/",
    "research/campaign_2026-10-03/continuation/review/service_observation_error/",
    "research/campaign_2026-10-03/continuation/review/cpu_barrier_race/",
    "research/campaign_2026-10-03/continuation/review/campaign_mock/",
    "research/campaign_2026-10-03/continuation/review/archive_fix_cli_restore/",
    "research/campaign_2026-10-03/continuation/review/archive_fix_fixtures/",
    "research/campaign_2026-10-03/continuation/review/large_state_probe_195201/",
    "research/campaign_2026-10-03/continuation/review/E385_preserved/",
    "research/campaign_2026-10-04/E462_python311_repair_v1/cargo/",
    "research/campaign_2026-10-04/E462_python311_repair_v1/native_source/",
    "research/campaign_2026-10-04/E462_python311_repair_v1/q8r/",
    "research/campaign_2026-10-04/E462_python311_repair_v1/q8x/",
    "research/campaign_2026-10-04/E462_python311_repair_v1/wheels/",
    "research/campaign_2026-10-04/E462_python311_repair_v1/target-v2/",
    "research/campaign_2026-10-04/E462_python311_repair_v1/target-v3/",
)
NUMERICAL_SUFFIXES = {
    ".parquet",
    ".arrow",
    ".feather",
    ".npy",
    ".npz",
    ".memmap",
    ".f16",
    ".f32",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".pkl",
}
NUMERICAL_MIN_BYTES = 1 << 20
PRIVATE_PATH_WORDS = {"transcripts", "transcript", "conversations", "private_context"}


def run_git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE).stdout


def tracked_paths() -> set[str]:
    return {
        path.decode("utf-8", errors="strict")
        for path in run_git("ls-files", "-z").split(b"\0")
        if path
    }


def working_tree_states() -> dict[str, str]:
    raw = run_git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    states: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        status = item[:2].decode("ascii", errors="replace")
        path = item[3:].decode("utf-8", errors="replace")
        if status == "??":
            states[path] = "current_untracked"
        elif "D" in status:
            states[path] = "deleted_in_worktree"
        elif status.strip():
            states[path] = "modified_in_worktree"
    return states


def path_is_excluded(relative: str) -> bool:
    path = Path(relative)
    parts = set(path.parts)
    if relative.startswith(EXCLUDED_PREFIXES):
        return True
    if parts & EXCLUDED_PARTS or parts & PRIVATE_PATH_WORDS:
        return True
    if any(part.lower() in {"cargo", "wheels"} for part in path.parts):
        return True
    if any(part.lower().startswith("target-") for part in path.parts):
        return True
    if "kaggle" in parts and "output" in parts:
        return True
    if any(part.startswith("provider_output") for part in path.parts):
        return True
    if any(part.startswith("provider_error") for part in path.parts):
        return True
    return any(
        part.startswith("restored")
        or part.startswith("restore_")
        or part.startswith("final_restore_")
        or part.startswith("duplicate_restore")
        for part in path.parts
    )


def eligible_path(relative: str) -> bool:
    if relative in {INDEX_MD.as_posix(), INDEX_JSON.as_posix()}:
        return False
    if path_is_excluded(relative):
        return False
    path = Path(relative)
    if relative in SELECTED_TOP_LEVEL:
        return True
    if relative.startswith("docs/"):
        return path.suffix.lower() in EVIDENCE_SUFFIXES
    if relative.startswith("research/"):
        suffix = path.suffix.lower()
        if suffix in EVIDENCE_SUFFIXES:
            return True
        return suffix in CODE_SUFFIXES
    if relative.startswith(CANONICAL_CODE_PREFIXES):
        return path.suffix.lower() in CODE_SUFFIXES
    return False


def purpose_for(relative: str) -> tuple[str, str]:
    """Filename-only navigation labels; these are not scientific assessments."""
    path = Path(relative)
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in CODE_SUFFIXES:
        if relative.startswith("research/"):
            return "Research implementation", "Research source, grader, test, or plotting code"
        return (
            "Canonical implementation",
            "Core model, metric, feature, inference, or review-index code",
        )
    if suffix in {".png", ".svg", ".pdf"}:
        return "Figure / visualization", "Visual evidence; inspect axes, samples, and controls"
    if suffix == ".ipynb":
        return "Notebook", "Interactive analysis notebook; inspect cells and embedded outputs"
    if "prereg" in name or "design" in name or "contract" in name or "spec" in name:
        return (
            "Design / specification",
            "Registered design, constraints, or source-bound specification",
        )
    if "review" in name or "critic" in name or "second_pair" in name:
        return (
            "Independent review",
            "Review or critique; compare its exact source and evidence bindings",
        )
    if "report" in name or "synthesis" in name or "retrospective" in name or "wrap" in name:
        return "Research report", "Narrative findings, review, synthesis, or handoff"
    if "plan" in name or "proposal" in name or "options" in name:
        return "Proposed directions", "Proposal or plan; execution requires separate evidence"
    if any(token in name for token in ("attempt", "receipt", "proof", "manifest", "provenance")):
        return (
            "Run / provenance receipt",
            "Execution status or provenance metadata; inspect source bindings",
        )
    if "result" in name or "score" in name or "metric" in name:
        return "Result / score", "Saved quantitative result; inspect split, metric, and uncertainty"
    if "log" in name:
        return "Execution log", "Saved command output or diagnostic log"
    if suffix == ".jsonl":
        return "Research ledger", "Append-only experiment or hypothesis record"
    if suffix in {".html", ".csv"}:
        return "Research artifact", "Saved interactive or tabular evidence"
    if relative.startswith("docs/") or relative in {"README.md", "METRIC.md"}:
        return "Competition specification", "Task, rules, metric, data, or submission contract"
    return "Research documentation", "Supporting research documentation or notes"


def list_candidate_paths(tracked: set[str]) -> tuple[set[str], list[dict[str, str]]]:
    candidates = {path for path in tracked if eligible_path(path)}
    for top in (ROOT / "research", ROOT / "docs"):
        if not top.exists():
            continue
        for current, dirs, files in os.walk(top, followlinks=False):
            current_path = Path(current)
            dirs[:] = [
                directory
                for directory in dirs
                if not path_is_excluded((current_path / directory).relative_to(ROOT).as_posix())
            ]
            for filename in files:
                absolute = current_path / filename
                try:
                    relative = absolute.relative_to(ROOT).as_posix()
                except ValueError:
                    continue
                if eligible_path(relative) and not absolute.is_symlink():
                    candidates.add(relative)

    unavailable: list[dict[str, str]] = []
    states = working_tree_states()
    for path in sorted(candidates):
        if path in tracked and ((ROOT / path).is_symlink() or not (ROOT / path).is_file()):
            reason = states.get(path, "missing_from_worktree")
            unavailable.append(
                {
                    "path": path,
                    "status": reason,
                    "action": "exclude; do not restore",
                }
            )
    candidates = {
        path for path in candidates if not (ROOT / path).is_symlink() and (ROOT / path).is_file()
    }
    return candidates, unavailable


def build_records(
    paths: set[str], tracked: set[str], states: dict[str, str]
) -> list[dict[str, Any]]:
    records = []
    for relative in sorted(paths):
        stat = (ROOT / relative).stat(follow_symlinks=False)
        category, purpose = purpose_for(relative)
        if relative in tracked:
            state = states.get(relative, "tracked_clean")
        else:
            state = "current_untracked_review_material"
        records.append(
            {
                "path": relative,
                "bytes": stat.st_size,
                "format": Path(relative).suffix.lower() or "[no extension]",
                "tracking_state": state,
                "category": category,
                "purpose": purpose,
            }
        )
    return records


def campaign_breakdown(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: Counter[str] = Counter()
    bytes_by: Counter[str] = Counter()
    for row in records:
        match = re.search(r"campaign_(\d{4}-\d{2}-\d{2})", row["path"])
        group = match.group(1) if match else "earlier_or_root_research"
        counts[group] += 1
        bytes_by[group] += row["bytes"]
    return {key: {"files": counts[key], "bytes": bytes_by[key]} for key in sorted(counts)}


def large_numerical_inventory() -> dict[str, Any]:
    """Inventory numerical files by path and stat only; never open or hash them."""
    roots = [ROOT / "research", ROOT / "cache", ROOT / "submissions", ROOT / "artifacts"]
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for current, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            dirs[:] = [
                directory
                for directory in dirs
                if not path_is_excluded((current_path / directory).relative_to(ROOT).as_posix())
            ]
            for filename in files:
                path = current_path / filename
                suffix = path.suffix.lower()
                if suffix not in NUMERICAL_SUFFIXES or path.is_symlink():
                    continue
                try:
                    stat = path.stat(follow_symlinks=False)
                    relative = path.relative_to(ROOT).as_posix()
                except (OSError, ValueError):
                    continue
                if stat.st_size < NUMERICAL_MIN_BYTES:
                    continue
                rows.append(
                    {
                        "path": relative,
                        "bytes": stat.st_size,
                        "format": suffix,
                        "tracked": relative in tracked_paths_cache,
                        "inspection": "filesystem metadata only; content and hash not read",
                    }
                )
    rows.sort(key=lambda row: row["path"])
    by_format: Counter[str] = Counter()
    bytes_by_format: Counter[str] = Counter()
    for row in rows:
        by_format[row["format"]] += 1
        bytes_by_format[row["format"]] += row["bytes"]
    return {
        "metadata_only": True,
        "minimum_bytes": NUMERICAL_MIN_BYTES,
        "files": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "formats": {
            ext: {"files": by_format[ext], "bytes": bytes_by_format[ext]}
            for ext in sorted(by_format)
        },
        "artifacts": rows,
    }


def make_index() -> dict[str, Any]:
    global tracked_paths_cache
    tracked_paths_cache = tracked_paths()
    states = working_tree_states()
    paths, unavailable = list_candidate_paths(tracked_paths_cache)
    files = build_records(paths, tracked_paths_cache, states)
    formats = Counter(row["format"] for row in files)
    tracking = Counter(row["tracking_state"] for row in files)
    numerical = large_numerical_inventory()
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_commit": run_git("rev-parse", "HEAD").decode().strip(),
        "scope": "Available tracked research evidence, current research documents, and explicitly selected canonical implementation code; filename-derived navigation labels only.",
        "excluded": [
            "Index files themselves to avoid self-reference",
            "Private transcripts/context, app settings, tokens, and local agent state",
            "Generated transport bundles, provider output copies, Cargo/build/package trees, runtime/vendor trees, restored/duplicate trees, and copied controller fixtures",
            "Large numerical artifacts from the primary document index; these appear in the separate metadata-only inventory below",
            "Unavailable tracked paths; these are listed explicitly and are not restored",
        ],
        "purpose_labels": "Filename-based navigation, not scientific conclusions",
        "file_count": len(files),
        "indexed_bytes": sum(row["bytes"] for row in files),
        "format_counts": dict(sorted(formats.items())),
        "tracking_state_counts": dict(sorted(tracking.items())),
        "campaign_breakdown": campaign_breakdown(files),
        "unavailable_tracked_files": unavailable,
        "large_numerical_artifacts_metadata_only": numerical,
        "files": files,
    }


def render_markdown(index: dict[str, Any]) -> str:
    by_directory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index["files"]:
        parent = str(Path(row["path"]).parent)
        by_directory[parent].append(row)
    lines = [
        "# Complete research review file index — October 5, 2026",
        "",
        f"Base commit: `{index['base_commit']}`. Generated `{index['generated_at_utc']}` UTC.",
        f"{index['file_count']} available files; {index['indexed_bytes']:,} bytes.",
        "",
        "See [the current second-pair guide](SECOND_PAIR_OF_EYES_2026-10-05.md) and [the elephants review](ELEPHANTS_REVIEW_2026-10-05.md) for synthesis and evaluation. This index is a filename-level navigation aid; purpose labels describe likely use and make no scientific claim.",
        "",
        "## Scope and counts",
        "",
        "The primary index contains available tracked research documents plus current research files in the selected evidence formats. It also includes explicitly selected canonical Python/Rust implementation files used to train, score, package, or reproduce reviewed work. Current untracked items are labeled. Generated transport/provider copies, Cargo/build/package trees, vendored runtimes, restored duplicates, private context, and the index files themselves are excluded.",
        "",
        "| Format | Files |",
        "|---|---:|",
    ]
    for ext, count in index["format_counts"].items():
        lines.append(f"| `{ext}` | {count} |")
    lines.extend(["", "| Campaign/date group | Files | Bytes |", "|---|---:|---:|"])
    for campaign, summary in index["campaign_breakdown"].items():
        lines.append(f"| `{campaign}` | {summary['files']} | {summary['bytes']:,} |")
    if index["unavailable_tracked_files"]:
        lines.extend(["", "## Unavailable tracked files", ""])
        lines.append(
            "These paths are absent from the current worktree and are deliberately not restored."
        )
        lines.append("")
        lines.append("| Path | Status |")
        lines.append("|---|---|")
        for row in index["unavailable_tracked_files"]:
            lines.append(f"| `{row['path']}` | {row['status']} |")
    lines.extend(
        [
            "",
            "## Large numerical artifacts — separate metadata-only inventory",
            "",
            f"The generator found {index['large_numerical_artifacts_metadata_only']['files']} numerical files at least {index['large_numerical_artifacts_metadata_only']['minimum_bytes']:,} bytes, totaling {index['large_numerical_artifacts_metadata_only']['total_bytes']:,} bytes. It used filesystem metadata only: artifact contents and hashes were not read. Exact paths, byte sizes, formats, and tracked status are in the JSON field `large_numerical_artifacts_metadata_only.artifacts`.",
            "",
            "| Format | Files | Bytes |",
            "|---|---:|---:|",
        ]
    )
    for ext, summary in index["large_numerical_artifacts_metadata_only"]["formats"].items():
        lines.append(f"| `{ext}` | {summary['files']} | {summary['bytes']:,} |")
    lines.extend(["", "## Complete file list", ""])
    for directory, rows in sorted(by_directory.items()):
        lines.extend(
            [
                f"### `{directory}`",
                "",
                "| File | Bytes | Category | Purpose | Status |",
                "|---|---:|---|---|---|",
            ]
        )
        for row in rows:
            label = quote(Path(row["path"]).name, safe="-_.()")
            target = "../" + quote(row["path"], safe="/-_.()")
            purpose = row["purpose"].replace("|", "\\|")
            category = row["category"].replace("|", "\\|")
            lines.append(
                f"| [{label}]({target}) | {row['bytes']:,} | {category} | {purpose} | {row['tracking_state']} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def validate(index: dict[str, Any], markdown: str) -> None:
    files = index["files"]
    paths = [row["path"] for row in files]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate path in review index")
    if INDEX_MD.as_posix() in paths or INDEX_JSON.as_posix() in paths:
        raise ValueError("index files must not reference themselves")
    for row in files:
        if (ROOT / row["path"]).is_symlink() or not (ROOT / row["path"]).is_file():
            raise ValueError(f"indexed path is missing: {row['path']}")
        if row["bytes"] != (ROOT / row["path"]).stat().st_size:
            raise ValueError(f"indexed size changed: {row['path']}")
        link = "../" + quote(row["path"], safe="/-_.()")
        if f"]({link})" not in markdown:
            raise ValueError(f"Markdown index link missing: {row['path']}")
        linked_path = ((ROOT / INDEX_MD).parent / unquote(link)).resolve()
        if linked_path != (ROOT / row["path"]).resolve():
            raise ValueError(f"Markdown link resolves to the wrong path: {row['path']}")
    for required in (
        "research/SECOND_PAIR_OF_EYES_2026-10-05.md",
        "research/ELEPHANTS_REVIEW_2026-10-05.md",
    ):
        if required not in paths:
            raise ValueError(f"required review document is not indexed: {required}")
        link = Path(required).name
        if f"]({link})" not in markdown or not (ROOT / required).is_file():
            raise ValueError(f"required review link is missing: {required}")
    for row in index["large_numerical_artifacts_metadata_only"]["artifacts"]:
        if Path(row["path"]).suffix.lower() not in NUMERICAL_SUFFIXES:
            raise ValueError(f"non-numerical file in large-artifact inventory: {row['path']}")
        if row["bytes"] < NUMERICAL_MIN_BYTES:
            raise ValueError(f"small file in large-artifact inventory: {row['path']}")
        if "hash" in row or "sha256" in row:
            raise ValueError("large numerical artifacts must not be hashed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the two dated index outputs")
    parser.add_argument("--check", action="store_true", help="build and validate in memory only")
    args = parser.parse_args()
    if args.write == args.check:
        parser.error("choose exactly one of --write or --check")
    index = make_index()
    markdown = render_markdown(index)
    validate(index, markdown)
    if args.write:
        (ROOT / INDEX_JSON).write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
        (ROOT / INDEX_MD).write_text(markdown)
    print(
        json.dumps(
            {
                "status": "validated",
                "base_commit": index["base_commit"],
                "files": index["file_count"],
                "indexed_bytes": index["indexed_bytes"],
                "large_numerical_metadata_only_count": index[
                    "large_numerical_artifacts_metadata_only"
                ]["files"],
                "unavailable_tracked": index["unavailable_tracked_files"],
                "outputs_written": args.write,
            },
            indent=2,
        )
    )


tracked_paths_cache: set[str] = set()


if __name__ == "__main__":
    main()
