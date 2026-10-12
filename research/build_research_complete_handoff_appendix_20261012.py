#!/usr/bin/env python3
"""Regenerate the source-preserving appendices for the October 12 handoff.

The three public scientific JSONL ledgers are parsed one record at a time and
embedded verbatim. The document catalog is metadata-only: it never opens
cataloged Markdown or JSON files. Private-path and credential-like names are
excluded from the catalog, and high-confidence secret/private-reasoning keys
in a ledger fail closed without printing their values.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import quote


LEDGERS: tuple[tuple[str, str, str], ...] = (
    ("hypotheses", "research/hypotheses.jsonl", "Hypothesis ledger"),
    ("experiments", "research/experiments.jsonl", "Experiment ledger"),
    ("journal", "research/bandit_tree/journal.jsonl", "Research-tree journal"),
)
CATALOG_ROOTS = ("research", "specs", "writeups", "submissions", "artifacts", "hpo", "docs")
CATALOG_SUFFIXES = {".md", ".markdown", ".json", ".jsonl"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
PRIVATE_PATH_PARTS = {
    ".env",
    "credential",
    "credentials",
    "secret",
    "secrets",
    "private-key",
    "private_keys",
    "api-key",
    "access-token",
    "cookies",
    "passwords",
}
PRIVATE_NAME_RE = re.compile(
    r"(?:credential|secret|password|api[-_]?key|access[-_]?token|refresh[-_]?token|private[-_]?key|cookie)",
    re.IGNORECASE,
)
CHALLENGE3_RE = re.compile(r"(?:^|[^a-z0-9])challenge[-_]?3(?:$|[^0-9])", re.IGNORECASE)
FORBIDDEN_LEDGER_KEYS = {
    "api_key",
    "access_token",
    "refresh_token",
    "password",
    "cookie",
    "private_reasoning",
    "chain_of_thought",
    "cot_transcript",
}
MAX_LEDGER_LINE_BYTES = 8 * 1024 * 1024
MAX_HANDOFF_BYTES = 32 * 1024 * 1024
BEGIN_MARKER = "<!-- BEGIN GENERATED LEDGER AND FILE APPENDICES -->"
END_MARKER = "<!-- END GENERATED LEDGER AND FILE APPENDICES -->"


class ArchiveError(RuntimeError):
    """Raised when source records or the target document fail validation."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forbidden_key(value: Any) -> str | None:
    """Return a sensitive key name, never its value, if one is present."""
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in FORBIDDEN_LEDGER_KEYS:
                return normalized
            nested = _forbidden_key(child)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for child in value:
            nested = _forbidden_key(child)
            if nested is not None:
                return nested
    return None


def _iter_ledger(path: Path) -> Iterator[tuple[int, bytes, dict[str, Any]]]:
    with path.open("rb") as stream:
        line_number = 0
        while True:
            raw = stream.readline(MAX_LEDGER_LINE_BYTES + 1)
            if not raw:
                break
            line_number += 1
            if len(raw) > MAX_LEDGER_LINE_BYTES:
                raise ArchiveError(f"ledger line exceeds fixed bound: {path}:{line_number}")
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArchiveError(f"invalid JSON record: {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ArchiveError(f"ledger record is not an object: {path}:{line_number}")
            sensitive = _forbidden_key(record)
            if sensitive is not None:
                raise ArchiveError(
                    f"sensitive/private-reasoning key {sensitive!r} found in {path}:{line_number}; values withheld"
                )
            yield line_number, raw.rstrip(b"\r\n"), record


def _read_handoff(path: Path) -> str:
    """Read a pre-existing handoff only after checking a fixed size ceiling."""
    try:
        initial_stat = path.stat()
    except OSError as exc:
        raise ArchiveError(f"cannot stat handoff input: {path}") from exc
    if initial_stat.st_size > MAX_HANDOFF_BYTES:
        raise ArchiveError(f"handoff exceeds fixed {MAX_HANDOFF_BYTES}-byte input bound: {path}")
    try:
        with path.open("rb") as stream:
            opened_stat = os.fstat(stream.fileno())
            if opened_stat.st_size > MAX_HANDOFF_BYTES:
                raise ArchiveError(f"opened handoff exceeds fixed {MAX_HANDOFF_BYTES}-byte input bound: {path}")
            raw = stream.read(MAX_HANDOFF_BYTES + 1)
    except OSError as exc:
        raise ArchiveError(f"cannot read handoff input: {path}") from exc
    if len(raw) > MAX_HANDOFF_BYTES:
        raise ArchiveError(f"handoff grew beyond fixed {MAX_HANDOFF_BYTES}-byte input bound: {path}")
    if len(raw) != opened_stat.st_size:
        raise ArchiveError(f"handoff changed size during bounded read: {path}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArchiveError(f"handoff is not valid UTF-8: {path}") from exc


def _record_id(record: dict[str, Any], line_number: int) -> str:
    for key in ("id", "event_id", "experiment_id", "hypothesis_id"):
        value = record.get(key)
        if value is not None and value != "":
            return str(value)
    return f"(no id; source line {line_number})"


def _compact_meta(record: dict[str, Any]) -> str:
    bits: list[str] = []
    for key in ("status", "outcome", "result", "hypothesis_id", "parent_id", "parent_ids", "parents"):
        value = record.get(key)
        if value is None or value == "":
            continue
        try:
            rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            rendered = str(value)
        if len(rendered) > 240:
            rendered = rendered[:237] + "..."
        bits.append(f"{key}={rendered}")
    return "; ".join(bits) if bits else "no compact status/parent fields"


def _fence(raw_text: str) -> str:
    runs = [len(match.group(0)) for match in re.finditer(r"`+", raw_text)]
    return "`" * max(3, (max(runs) + 1) if runs else 3)


def _md_link(label: str, relative_path: str) -> str:
    normalized = relative_path.replace(os.sep, "/")
    target, separator, fragment = normalized.partition("#")
    encoded = quote(target, safe="/._-()")
    if separator:
        encoded += "#" + quote(fragment, safe="._-")
    return f"[{label}]({encoded})"


def _ledger_appendix(root: Path, research_dir: Path) -> str:
    chunks = [
        "## Complete append-only ledger records",
        "",
        "Every nonblank source row is reproduced below in original order, without deduplicating revisions, retries, or repeated IDs. Each heading links to the exact source JSONL line; the fenced JSON is the original record text. These are project-authored scientific records, not private agent transcripts. Row counts are not counts of unique experiments.",
        "",
    ]
    for key, rel_source, title in LEDGERS:
        source = root / rel_source
        ids: Counter[str] = Counter()
        row_count = 0
        for number, _, record in _iter_ledger(source):
            row_count += 1
            ids[_record_id(record, number)] += 1
        digest = _sha256_file(source)
        duplicates = sum(count > 1 for count in ids.values())
        source_from_research = os.path.relpath(source, research_dir).replace(os.sep, "/")
        chunks.extend(
            [
                f"### {title}: {row_count} rows, {len(ids)} distinct IDs",
                "",
                f"Source: {_md_link(rel_source, source_from_research)}; SHA-256 `{digest}`. Distinct-ID values are counted after converting each source ID to text. Repeated IDs: {sum(count - 1 for count in ids.values())}; IDs occurring more than once: {duplicates}.",
                "",
            ]
        )
        for ordinal, (line_number, raw, record) in enumerate(_iter_ledger(source), start=1):
            raw_text = raw.decode("utf-8")
            identifier = html.escape(_record_id(record, line_number), quote=False)
            line_link = _md_link(f"{Path(rel_source).name} line {line_number}", f"{source_from_research}#L{line_number}")
            fence = _fence(raw_text)
            chunks.extend(
                [
                    f"#### {ordinal}. {identifier}",
                    "",
                    f"Source: {line_link}. {_compact_meta(record)}.",
                    "",
                    fence + "json",
                    raw_text,
                    fence,
                    "",
                ]
            )
    return "\n".join(chunks)


def _is_sensitive_path(parts: Sequence[str]) -> bool:
    for part in parts:
        normalized = part.lower().replace("_", "-")
        if normalized in PRIVATE_PATH_PARTS:
            return True
        if PRIVATE_NAME_RE.search(part) or CHALLENGE3_RE.search(part):
            return True
        if normalized.startswith(".env"):
            return True
    return False


def _catalog_files(root: Path) -> tuple[list[Path], int]:
    selected: list[Path] = []
    excluded = 0
    candidates: list[Path] = [root / name for name in ("README.md", "AGENTS.md") if (root / name).is_file()]
    for root_name in CATALOG_ROOTS:
        catalog_root = root / root_name
        if catalog_root.is_dir():
            candidates.append(catalog_root)
    for candidate in candidates:
        if candidate.is_file():
            rel = candidate.relative_to(root)
            if _is_sensitive_path(rel.parts):
                excluded += 1
            elif candidate.suffix.lower() in CATALOG_SUFFIXES:
                selected.append(candidate)
            continue
        for dirpath, dirnames, filenames in os.walk(candidate, followlinks=False):
            base = Path(dirpath)
            kept_dirs: list[str] = []
            for dirname in dirnames:
                child = base / dirname
                rel_parts = child.relative_to(root).parts
                if dirname in SKIP_DIRS or child.is_symlink() or _is_sensitive_path(rel_parts):
                    excluded += 1
                else:
                    kept_dirs.append(dirname)
            dirnames[:] = kept_dirs
            for filename in filenames:
                path = base / filename
                if path.is_symlink():
                    excluded += 1
                    continue
                rel_parts = path.relative_to(root).parts
                if _is_sensitive_path(rel_parts):
                    excluded += 1
                    continue
                if path.suffix.lower() in CATALOG_SUFFIXES:
                    selected.append(path)
    return sorted(set(selected)), excluded


def _catalog_class(path: Path) -> str:
    name = path.name.lower()
    claim_terms = ("grade", "result", "terminal", "receipt", "review", "validation", "admission", "release", "manifest", "score", "quality", "evidence", "handoff")
    snapshot_terms = ("snapshot", "source", "log", "input", "plan", "notes", "index", "proposal", "contract", "readme", "status", "frontier")
    if any(term in name for term in claim_terms):
        return "claim-receipt-like by filename"
    if any(term in name for term in snapshot_terms):
        return "source snapshot / plan by filename"
    return "documentation / metadata by filename"


def _catalog_appendix(root: Path, research_dir: Path) -> str:
    files, excluded = _catalog_files(root)
    groups: dict[str, list[Path]] = {}
    for path in files:
        groups.setdefault(path.parent.relative_to(root).as_posix(), []).append(path)
    suffix_counts = Counter(path.suffix.lower() for path in files)
    chunks = [
        "## Grouped documentation and receipt catalog",
        "",
        f"Metadata-only inventory: {len(files)} Markdown/JSON/JSONL files under README.md, AGENTS.md, research/, specs/, writeups/, submissions/, artifacts/, hpo/, and docs/; {excluded} directory/file entries were excluded by private-path, challenge-3, symlink, or generated-environment filters. Suffix counts: "
        + ", ".join(f"`{suffix or '(none)'}` {count}" for suffix, count in sorted(suffix_counts.items()))
        + ". No cataloged file contents are opened by this generator. JSON files larger than 2 MiB are explicitly marked; JSON files of every size remain metadata-only here.",
        "",
        "Filename-based labels are navigational hints only. A claim-bearing result still requires checking its named source, manifest, terminal, and grade tuple. Historical source snapshots and duplicate copies are retained and should not be mistaken for independent evidence.",
        "",
    ]
    for directory in sorted(groups):
        chunks.extend([f"### `{directory}/`", ""])
        for path in sorted(groups[directory], key=lambda item: item.name.casefold()):
            stat = path.stat()
            size = stat.st_size
            modified = dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
            relative_from_research = os.path.relpath(path, research_dir).replace(os.sep, "/")
            link = _md_link(path.name, relative_from_research)
            large_note = "; >2 MiB, content not opened" if path.suffix.lower() == ".json" and size > 2 * 1024 * 1024 else ""
            chunks.append(f"- {link} — {size:,} bytes; modified {modified}; { _catalog_class(path) }{large_note}.")
        chunks.append("")
    return "\n".join(chunks)


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _generated_block(root: Path, research_dir: Path) -> str:
    head = _git_head(root)
    return "\n".join(
        [
            BEGIN_MARKER,
            "",
            "## Appendix generation record",
            "",
            f"Generated from source repository HEAD `{head}`. The generator streams each source JSONL record with an 8 MiB per-line bound, preserves the raw row, and catalogs documentation by filesystem metadata only.",
            "",
            _ledger_appendix(root, research_dir),
            "",
            _catalog_appendix(root, research_dir),
            END_MARKER,
        ]
    )


def _replace_generated_block(document: str, block: str) -> str:
    if document.count(BEGIN_MARKER) != 1 or document.count(END_MARKER) != 1:
        raise ArchiveError("handoff must contain exactly one generated-appendix marker pair")
    start = document.index(BEGIN_MARKER)
    end_marker = document.index(END_MARKER)
    if end_marker < start:
        raise ArchiveError("generated-appendix markers are out of order")
    end = end_marker + len(END_MARKER)
    return document[:start] + block + document[end:]


def _validate_local_links(
    document: str, research_dir: Path, staged_research_dir: Path
) -> int:
    """Fail closed on broken repository-relative Markdown links outside fences."""
    markdown_link = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
    fence_char: str | None = None
    fence_size = 0
    checked = 0
    missing: list[str] = []
    for line_number, line in enumerate(document.splitlines(), start=1):
        fence_match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence_match:
            marker = fence_match.group(1)
            if fence_char is None:
                fence_char, fence_size = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_size:
                fence_char, fence_size = None, 0
            continue
        if fence_char is not None:
            continue
        for match in markdown_link.finditer(line):
            target = match.group(1).strip()
            if target.startswith("<"):
                target = target[1:].split(">", 1)[0]
            else:
                target = target.split()[0]
            if target.startswith(("https://", "http://", "mailto:", "data:")):
                continue
            path_part = target.split("#", 1)[0].split("?", 1)[0]
            if not path_part:
                continue
            checked += 1
            relative = quote_decode(path_part)
            destination = (research_dir / relative).resolve()
            staged_destination = (staged_research_dir / relative).resolve()
            if not destination.exists() and not staged_destination.exists():
                missing.append(f"line {line_number}: {path_part}")
    if missing:
        sample = "; ".join(missing[:10])
        raise ArchiveError(f"{len(missing)} broken local Markdown links (of {checked} checked): {sample}")
    return checked


def quote_decode(value: str) -> str:
    """Decode URL-escaped path components for filesystem link validation."""
    from urllib.parse import unquote

    return unquote(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="authoritative source repository")
    parser.add_argument("--output", type=Path, help="handoff Markdown to update; defaults under --root/research")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = (args.output or root / "research/RESEARCH_COMPLETE_HANDOFF_20261012.md").resolve()
    # The output may be staged in a sibling worktree. Link targets must be
    # relative to its eventual repository location, not the temporary path.
    research_dir = root / "research"
    if not (root / "research").is_dir():
        raise ArchiveError("source root has no research directory")
    if not output.is_file():
        raise ArchiveError("handoff output file does not exist")
    original = _read_handoff(output)
    block = _generated_block(root, research_dir)
    updated = _replace_generated_block(original, block)
    checked_links = _validate_local_links(updated, research_dir, output.parent)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(updated, encoding="utf-8")
    os.replace(temp, output)
    print(f"wrote {output}")
    print(f"output_bytes={output.stat().st_size}")
    print(f"local_markdown_links_checked={checked_links}")
    for _, rel_source, label in LEDGERS:
        source = root / rel_source
        count = sum(1 for _ in _iter_ledger(source))
        print(f"{label}: rows={count} sha256={_sha256_file(source)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ArchiveError as exc:
        print(f"archive generation stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
