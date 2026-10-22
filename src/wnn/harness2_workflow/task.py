"""Load one bounded, source-qualified task for the Harness2 workflow."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

TASK_SCHEMA = "wnn-harness2-task.v1"
MAX_SOURCE_BYTES = 256 * 1024
MAX_TOTAL_SOURCE_BYTES = 2 * 1024 * 1024
MAX_TASK_JSON_BYTES = 256 * 1024
MAX_PROTECTED_BYTES = 4 * 1024 * 1024
MAX_TOTAL_PROTECTED_BYTES = 8 * 1024 * 1024
MAX_FILES = 32
_SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_CREDENTIAL_PATTERN = re.compile(
    r"(?:hf_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|"
    r"sk-[A-Za-z0-9]{16,}|[\"']?(?:api[_-]?key|access[_-]?token|secret[_-]?key)"
    r"[\"']?\s*[:=]\s*[\"']?\S{12,})",
    re.IGNORECASE,
)
_FORBIDDEN_SOURCE_NAMES = frozenset(
    {
        "answer",
        "answers",
        "answer_key",
        "answer-key",
        "checkpoint",
        "checkpoints",
        "credential",
        "credentials",
        "dataset",
        "datasets",
        "grader",
        "grading",
        "ground_truth",
        "ground-truth",
        "metric",
        "metrics",
        "private",
        "private_assets",
        "raw",
        "raw_data",
        "raw-dataset",
        "raw_dataset",
        "reference",
        "reference_answer",
        "reference-answer",
        "rubric",
        "score",
        "scores",
        "secret",
        "secrets",
        "solution",
        "solutions",
        "token",
        "tokens",
        "verdict",
        "verdicts",
    }
)
_FORBIDDEN_SOURCE_SUFFIXES = frozenset(
    {".ckpt", ".npy", ".npz", ".parquet", ".pth", ".pt", ".safetensors"}
)
_SECRET_NAMES = frozenset({"credential", "credentials", "secret", "secrets", "token", "tokens"})
_FORBIDDEN_PROTECTED_NAMES = frozenset(
    {
        "checkpoint",
        "checkpoints",
        "private",
        "private_assets",
        "raw",
        "raw_data",
        "raw-dataset",
        "raw_dataset",
    }
)


def _sha256(data: bytes) -> str:
    """Return a lowercase SHA-256 digest for exact file bytes."""
    return hashlib.sha256(data).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Keep a duplicate JSON field from silently replacing its first value."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate task JSON key {key!r}")
        result[key] = value
    return result


def _is_link(path: Path) -> bool:
    """Recognize symbolic links and Windows reparse points before reading a file."""
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _relative_path(value: object, *, label: str) -> PurePosixPath:
    """Validate the portable repository-relative path used in the task contract."""
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty POSIX-style relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"{label} must be repository-relative: {value!r}")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{label} contains an unsafe path component: {value!r}")
    if posix.as_posix() != value:
        raise ValueError(f"{label} must use a normalized relative path: {value!r}")
    if any("\n" in part or "\r" in part for part in posix.parts):
        raise ValueError(f"{label} contains a control character: {value!r}")
    return posix


def _check_sensitive_path(path: PurePosixPath, *, source: bool) -> None:
    """Reject credentials and paths that could expose hidden task outcomes."""
    parts = tuple(part.casefold() for part in path.parts)
    for part in parts:
        if part == ".env" or part.startswith(".env."):
            raise ValueError(f"task paths may not include environment files: {path.as_posix()!r}")
        stem = PurePosixPath(part).stem
        if stem in _SECRET_NAMES or any(
            marker in stem for marker in ("credential", "secret", "token")
        ):
            raise ValueError(
                f"task paths may not include credentials or secrets: {path.as_posix()!r}"
            )
        if source and (part in _FORBIDDEN_SOURCE_NAMES or stem in _FORBIDDEN_SOURCE_NAMES):
            raise ValueError(f"task source is not human task evidence: {path.as_posix()!r}")
        if source and PurePosixPath(part).suffix in _FORBIDDEN_SOURCE_SUFFIXES:
            raise ValueError(
                f"raw dataset/checkpoint files may not be task sources: {path.as_posix()!r}"
            )
        if not source and (
            part in _FORBIDDEN_PROTECTED_NAMES or stem in _FORBIDDEN_PROTECTED_NAMES
        ):
            raise ValueError(
                f"protected paths may not include private or raw data: {path.as_posix()!r}"
            )
        if PurePosixPath(part).suffix in _FORBIDDEN_SOURCE_SUFFIXES:
            raise ValueError(
                f"raw dataset/checkpoint files are not task inputs: {path.as_posix()!r}"
            )


def _repository_file(repository: Path, value: object, *, label: str, source: bool) -> Path:
    """Resolve one existing regular file while refusing links and repository escapes."""
    relative = _relative_path(value, label=label)
    _check_sensitive_path(relative, source=source)
    current = repository
    for part in relative.parts:
        current = current / part
        try:
            if _is_link(current):
                raise ValueError(f"{label} may not follow symlinks: {relative.as_posix()!r}")
        except FileNotFoundError:
            raise FileNotFoundError(f"{label} does not exist: {relative.as_posix()!r}") from None
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(repository):
        raise ValueError(f"{label} escapes the repository: {relative.as_posix()!r}")
    if not resolved.is_file():
        raise ValueError(f"{label} must name a regular file: {relative.as_posix()!r}")
    return resolved


def _fingerprint(
    task_json_hash: str, source_hashes: dict[str, str], protected: dict[str, str]
) -> str:
    """Bind the exact task document and every named file digest into one identity."""
    payload = {
        "task_json_sha256": task_json_hash,
        "sources": dict(sorted(source_hashes.items())),
        "protected": dict(sorted(protected.items())),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256(encoded.encode("utf-8"))


@dataclass(frozen=True)
class Task:
    """An immutable task contract and hashes for its source and protected files."""

    task_id: str
    domain: str
    prompt: str
    sources: dict[str, Path]
    protected: dict[str, str]
    fingerprint: str
    _task_path: Path = field(repr=False, compare=False)
    _task_json_hash: str = field(repr=False, compare=False)
    _source_hashes: dict[str, str] = field(repr=False, compare=False)
    _repository: Path = field(repr=False, compare=False)

    @classmethod
    def load(cls, path: Path, repository: Path) -> Task:
        """Load a versioned task JSON file after validating all selected paths and caps.

        Args:
            path: JSON task contract.
            repository: Root against which each source/protected path is resolved.

        Raises:
            ValueError: The contract, paths, or size limits are invalid.
            FileNotFoundError: A named file does not exist.
        """
        task_path = Path(path)
        if _is_link(task_path):
            raise ValueError("task JSON may not be a symlink")
        task_path = task_path.resolve(strict=True)
        if not task_path.is_file():
            raise ValueError("task JSON must be a regular file")
        if task_path.stat().st_size > MAX_TASK_JSON_BYTES:
            raise ValueError(f"task JSON exceeds {MAX_TASK_JSON_BYTES} bytes")
        raw = task_path.read_bytes()
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
        if not isinstance(document, dict):
            raise ValueError("task JSON must contain one object")
        expected_keys = {"schema", "id", "domain", "prompt", "sources", "protected_files"}
        if set(document) != expected_keys:
            raise ValueError(f"task JSON fields must be exactly {sorted(expected_keys)}")
        if document["schema"] != TASK_SCHEMA:
            raise ValueError(f"task schema must be {TASK_SCHEMA!r}")
        task_id = document["id"]
        domain = document["domain"]
        prompt = document["prompt"]
        if not isinstance(task_id, str) or not _SLUG.fullmatch(task_id):
            raise ValueError("task id must be a lowercase safe slug")
        if not isinstance(domain, str) or not _SLUG.fullmatch(domain):
            raise ValueError("task domain must be a lowercase safe slug")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("task prompt must be nonempty text")
        if _CREDENTIAL_PATTERN.search(prompt):
            raise ValueError("task prompt contains a credential-like token")
        raw_sources = document["sources"]
        raw_protected = document["protected_files"]
        if not isinstance(raw_sources, dict):
            raise ValueError(
                "task sources must be an object mapping destination to repository path"
            )
        if not isinstance(raw_protected, list) or not all(
            isinstance(item, str) for item in raw_protected
        ):
            raise ValueError("protected_files must be an array of repository-relative paths")
        if len(raw_sources) + len(raw_protected) > MAX_FILES:
            raise ValueError(f"task may name at most {MAX_FILES} source and protected files")

        repo = Path(repository).resolve(strict=True)
        if not repo.is_dir():
            raise ValueError("repository must be a directory")
        sources: dict[str, Path] = {}
        source_hashes: dict[str, str] = {}
        total_bytes = 0
        normalized_destinations: set[str] = set()
        for destination, source_name in raw_sources.items():
            destination_path = _relative_path(destination, label="source destination")
            _check_sensitive_path(destination_path, source=True)
            normalized = destination_path.as_posix().casefold()
            if normalized in normalized_destinations:
                raise ValueError(
                    f"source destinations collide on case-insensitive filesystems: {destination!r}"
                )
            normalized_destinations.add(normalized)
            source_path = _repository_file(repo, source_name, label="task source", source=True)
            size = source_path.stat().st_size
            if size > MAX_SOURCE_BYTES:
                raise ValueError(f"task source exceeds {MAX_SOURCE_BYTES} bytes: {source_name!r}")
            total_bytes += size
            if total_bytes > MAX_TOTAL_SOURCE_BYTES:
                raise ValueError(f"combined task sources exceed {MAX_TOTAL_SOURCE_BYTES} bytes")
            source_bytes = source_path.read_bytes()
            if _CREDENTIAL_PATTERN.search(source_bytes.decode("utf-8", errors="ignore")):
                raise ValueError(f"task source contains a credential-like token: {source_name!r}")
            sources[destination_path.as_posix()] = source_path
            source_hashes[destination_path.as_posix()] = _sha256(source_bytes)

        protected: dict[str, str] = {}
        total_protected_bytes = 0
        for protected_name in raw_protected:
            protected_path = _repository_file(
                repo, protected_name, label="protected file", source=False
            )
            protected_size = protected_path.stat().st_size
            if protected_size > MAX_PROTECTED_BYTES:
                raise ValueError(
                    f"protected file exceeds {MAX_PROTECTED_BYTES} bytes: {protected_name!r}"
                )
            total_protected_bytes += protected_size
            if total_protected_bytes > MAX_TOTAL_PROTECTED_BYTES:
                raise ValueError(
                    f"combined protected files exceed {MAX_TOTAL_PROTECTED_BYTES} bytes"
                )
            absolute = str(protected_path)
            if absolute in protected:
                raise ValueError(f"duplicate protected file {protected_name!r}")
            if any(protected_path.samefile(source) for source in sources.values()):
                raise ValueError("a file may not be both exposed as a source and protected")
            if any(protected_path.samefile(Path(existing)) for existing in protected):
                raise ValueError(f"duplicate protected file {protected_name!r}")
            protected[absolute] = _sha256(protected_path.read_bytes())

        task_json_hash = _sha256(raw)
        fingerprint = _fingerprint(task_json_hash, source_hashes, protected)
        return cls(
            task_id=task_id,
            domain=domain,
            prompt=prompt,
            sources=sources,
            protected=protected,
            fingerprint=fingerprint,
            _task_path=task_path,
            _task_json_hash=task_json_hash,
            _source_hashes=source_hashes,
            _repository=repo,
        )

    def assert_unchanged(self) -> None:
        """Re-read the task contract, every source and protected file; reject any drift."""
        try:
            if (
                _is_link(self._task_path)
                or _sha256(self._task_path.read_bytes()) != self._task_json_hash
            ):
                raise RuntimeError("task JSON changed after it was loaded")
        except OSError as error:
            raise RuntimeError("task JSON was removed or became unreadable") from error
        actual_source_hashes: dict[str, str] = {}
        for destination, path in self.sources.items():
            try:
                relative = path.relative_to(self._repository).as_posix()
                checked_path = _repository_file(
                    self._repository, relative, label="task source", source=True
                )
            except (OSError, ValueError) as error:
                raise RuntimeError(f"task source path became unsafe: {path}") from error
            if checked_path != path:
                raise RuntimeError(f"task source path changed after it was loaded: {destination}")
            actual_source_hashes[destination] = _sha256(path.read_bytes())
            if actual_source_hashes[destination] != self._source_hashes.get(destination):
                raise RuntimeError(f"task source changed after it was loaded: {destination}")
        actual_protected: dict[str, str] = {}
        for absolute, expected_hash in self.protected.items():
            path = Path(absolute)
            try:
                relative = path.relative_to(self._repository).as_posix()
                checked_path = _repository_file(
                    self._repository, relative, label="protected file", source=False
                )
            except (OSError, ValueError) as error:
                raise RuntimeError(f"protected file path became unsafe: {absolute}") from error
            if checked_path != path:
                raise RuntimeError(f"protected file path changed after it was loaded: {absolute}")
            actual_hash = _sha256(path.read_bytes())
            actual_protected[absolute] = actual_hash
            if actual_hash != expected_hash:
                raise RuntimeError(f"protected file changed after it was loaded: {absolute}")
        if (
            _fingerprint(self._task_json_hash, actual_source_hashes, actual_protected)
            != self.fingerprint
        ):
            raise RuntimeError("task fingerprint no longer matches its inputs")
