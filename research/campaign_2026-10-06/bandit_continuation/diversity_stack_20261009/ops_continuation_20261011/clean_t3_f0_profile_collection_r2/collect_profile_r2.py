from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
CLOUD = MAIN / "research/campaign_2026-10-06/cloud_preflight"
sys.path.insert(0, str(CLOUD))

import collect_terminal_outputs as terminal  # noqa: E402
import poll_provider  # noqa: E402

KERNEL = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r1-runner"
DATASET = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r1"
KERNEL_VERSION = 1
ARCHIVE_SHA256 = "0ba1b1f7b99dc5dabd892c7d99aa030733465aebc73bb652455c9bc527e0d65f"
R1_COMMIT = "9080f2ae21780595bbed0d163f6f562eb475a34e"
TOKEN_PREFIX = "Next Page Token = "
PAGE_SIZE = 200
MAX_INVENTORY_PAGES = 20
KERNEL_OUTPUT_LOG_NAME = f"{KERNEL.split('/')[-1]}.log"
EXPECTED_SOURCE_SHAS = {
    "collect_terminal_outputs.py": "c1c65d1d55e46eff446817993cbf0258c321cfefc4bc9e60ab7400022ea9c85f",
    "poll_provider.py": "aab0e1b0972d46e1c61699f770948d55a797cbedf139f3ff3ab194eac4e518c6",
}
EXPECTED_KAGGLE_SOURCE_SHAS = {
    "cli.py": "8ed42fe894ba75631cec80b99bf736e1c69c48799df54ce89457f8d0ae9a7edd",
    "kaggle_api_extended.py": "210e589ab1f7359ffb560aacfe1c003ac1bc844c366f4ff709bf3d13de4d5458",
}
EXPECTED_SESSION_OUTPUT_MODULE_SHA256 = (
    "a7dd65206e7fc27950385e7b163cda48a40766eb8ff7f89d7a334b9bfb151b5c"
)
STATUS_TARGET_PATH = HERE / "PROVIDER_STATUS_TARGET_R2.json"


def sha256(path: Path) -> str:
    """Hash a file in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_provider_name(name: str) -> bool:
    """Accept only normalized, relative POSIX output paths."""
    path = PurePosixPath(name)
    return (
        bool(name)
        and "\\" not in name
        and not any(ord(char) < 32 or ord(char) == 127 for char in name)
        and not path.is_absolute()
        and path.as_posix() == name
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def parse_inventory_page(stdout: str) -> tuple[list[dict[str, Any]], str | None]:
    """Parse Kaggle CLI 2.2.4's token preamble and JSON file-list page."""
    text = stdout.strip()
    if not text:
        raise ValueError("empty provider inventory response")

    next_token = None
    lines = text.splitlines()
    if lines[0].startswith(TOKEN_PREFIX):
        next_token = lines[0][len(TOKEN_PREFIX) :].strip()
        if not next_token:
            raise ValueError("provider inventory has an empty next-page token")
        text = "\n".join(lines[1:]).strip()
    if not text.startswith("["):
        raise ValueError("provider inventory response does not begin with a JSON list")

    try:
        files = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"provider inventory JSON is invalid: {error}") from error
    if not isinstance(files, list):
        raise ValueError("provider inventory JSON must be a list")

    required_fields = {"name", "size", "creationDate"}
    for index, item in enumerate(files):
        if not isinstance(item, dict) or set(item) != required_fields:
            raise ValueError(f"provider inventory row {index} has an unexpected schema")
        if not isinstance(item["name"], str) or not safe_provider_name(item["name"]):
            raise ValueError(f"provider inventory row {index} has an unsafe filename")
        if isinstance(item["size"], bool) or not isinstance(item["size"], int) or item["size"] < 0:
            raise ValueError(f"provider inventory row {index} has an invalid byte size")
        if item["creationDate"] is not None and not isinstance(item["creationDate"], str):
            raise ValueError(f"provider inventory row {index} has an invalid creation date")
    return files, next_token


def verify_root_bindings() -> dict[str, dict[str, Any]]:
    """Fail closed unless copied root admission, push and READY receipts match."""
    pins = {
        "ROOT_PROFILE_DISPATCH_ADMISSION_R1.json": "1368c5283b532dbee61567edcf50b595db3dca0f2d9a238d6cc2afdb10a33948",
        "ROOT_PROFILE_PUSH_R1.json": "19eb640c6e8425398a7244c0244154ca23c338dc6151c30f47a1d8c4ce9d82f1",
        "ROOT_PRIVATE_DATASET_READY_READBACK_R1.json": "8c2e4c846584bdb228da40a9c843e9aade4a6b8b354b30a3f3c5fea7484f9f81",
        "ROOT_PROFILE_PROVIDER_POLL_R1.json": "851e97e446c6afa6ab4ef8ac6305c76bd81a75d40afd9f884560fb30ad5c6b5d",
    }
    bindings: dict[str, dict[str, Any]] = {}
    for name, expected in pins.items():
        path = HERE / "root_bindings" / name
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"root receipt hash mismatch: {name}")
        bindings[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": observed}

    admission = json.loads(
        (HERE / "root_bindings" / "ROOT_PROFILE_DISPATCH_ADMISSION_R1.json").read_text()
    )
    push = json.loads((HERE / "root_bindings" / "ROOT_PROFILE_PUSH_R1.json").read_text())
    ready = json.loads(
        (HERE / "root_bindings" / "ROOT_PRIVATE_DATASET_READY_READBACK_R1.json").read_text()
    )
    if (
        admission.get("scope") != "two T4 profile jobs only, no fullfit"
        or admission.get("archive_sha256") != ARCHIVE_SHA256
    ):
        raise ValueError("root admission scope/archive binding mismatch")
    if (
        push.get("kernel_id") != KERNEL
        or push.get("returncode") != 0
        or push.get("profile_only") is not True
    ):
        raise ValueError("root push receipt does not bind the admitted profile kernel")
    if (
        push.get("dataset_version") != KERNEL_VERSION
        or push.get("readback_sha256")
        != bindings["ROOT_PRIVATE_DATASET_READY_READBACK_R1.json"]["sha256"]
    ):
        raise ValueError("push receipt does not bind exact READY readback")
    if ready.get("dataset") != DATASET or ready.get("kernel_push") is not False:
        raise ValueError("READY receipt dataset or scope mismatch")
    expected_remote = {
        "bundle.wnn": (217267, ARCHIVE_SHA256),
        "bundle-manifest.json": (
            24149,
            "8529006db1c07b9935aef4c1a4dc63e18a08b0dc1aeef129f3e8bfe4a8c6d960",
        ),
    }
    observed_remote = {
        row["name"]: (row["size_bytes"], row["sha256"])
        for row in ready.get("verified_remote_files", [])
    }
    if observed_remote != expected_remote:
        raise ValueError("remote READY file inventory differs from admitted archive")
    return bindings


def verify_status_target() -> str:
    """Bind one read-only unsuffixed status target for this exact profile kernel."""
    target = json.loads(STATUS_TARGET_PATH.read_text())
    if target != {
        "schema": "wnn.provider.targets.r1",
        "targets": [{"kernel": KERNEL, "resource": "gpu"}],
    }:
        raise ValueError("provider status target must bind only the exact unsuffixed kernel slug")
    return sha256(STATUS_TARGET_PATH)


def provider_status_available(status_row: dict[str, Any]) -> bool:
    """Distinguish a real provider state from the poller's encoded HTTP failures."""
    status = status_row.get("status")
    return isinstance(status, str) and bool(status)


def verify_collector_sources() -> dict[str, str]:
    """Ensure this adapter uses the reviewed collector and provider poller."""
    paths = {
        "collect_terminal_outputs.py": CLOUD / "collect_terminal_outputs.py",
        "poll_provider.py": CLOUD / "poll_provider.py",
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    if observed != EXPECTED_SOURCE_SHAS:
        raise ValueError("reviewed collector or provider-poller source changed")

    package = MAIN / ".venv/lib/python3.13/site-packages/kaggle"
    kaggle_paths = {
        "cli.py": package / "cli.py",
        "kaggle_api_extended.py": package / "api/kaggle_api_extended.py",
    }
    session_output_module = (
        MAIN / ".venv/lib/python3.13/site-packages/kagglesdk/kernels/types/kernels_api_service.py"
    )
    kaggle_observed = {name: sha256(path) for name, path in kaggle_paths.items()}
    session_output_sha = sha256(session_output_module)
    if kaggle_observed != EXPECTED_KAGGLE_SOURCE_SHAS:
        raise ValueError("installed Kaggle CLI source differs from the inspected 2.2.4 interface")
    if session_output_sha != EXPECTED_SESSION_OUTPUT_MODULE_SHA256:
        raise ValueError(
            "installed Kaggle session-output SDK types differ from the inspected interface"
        )
    interface_path = HERE / "KAGGLE_FILES_CLI_INTERFACE_R2.json"
    interface = json.loads(interface_path.read_text())
    if interface.get("cli_version") != "Kaggle CLI 2.2.4":
        raise ValueError("saved Kaggle CLI interface receipt has an unexpected version")
    help_text = interface.get("help_stdout", "")
    if (
        hashlib.sha256(help_text.encode()).hexdigest() != interface.get("help_stdout_sha256")
        or any(flag not in help_text for flag in ("--format", "--page-token", "--page-size"))
        or "max is 200" not in help_text
    ):
        raise ValueError("saved Kaggle CLI help does not verify JSON pagination support")
    if interface.get("source_bindings") != {
        name: {"path": str(path), "sha256": kaggle_observed[name]}
        for name, path in {
            "cli.py": MAIN / ".venv/lib/python3.13/site-packages/kaggle/cli.py",
            "kaggle_api_extended.py": MAIN
            / ".venv/lib/python3.13/site-packages/kaggle/api/kaggle_api_extended.py",
        }.items()
    }:
        raise ValueError("saved Kaggle CLI interface receipt does not bind inspected source")
    if interface.get("source_observed_contract", {}).get("kernels_output_log_side_effect") != (
        "kernels output writes <kernel-slug>.log from response.log into the output directory"
    ):
        raise ValueError("saved Kaggle output-side-effect contract is missing")
    output_interface_path = HERE / "KAGGLE_SESSION_OUTPUT_INTERFACE_R3.json"
    output_interface = json.loads(output_interface_path.read_text())
    if (
        output_interface.get("installed_cli_version") != "Kaggle CLI 2.2.4"
        or output_interface.get("session_output_module_sha256") != session_output_sha
        or output_interface.get("kaggle_api_extended_sha256")
        != kaggle_observed["kaggle_api_extended.py"]
        or output_interface.get("file_descriptor_fields") != ["file_name", "url"]
        or output_interface.get("server_size_or_checksum_fields") != []
        or output_interface.get("kernel_version_sent_by_installed_cli") is not False
    ):
        raise ValueError(
            "saved session-output API interface receipt does not match installed sources"
        )
    return {
        **observed,
        **{f"kaggle/{name}": digest for name, digest in kaggle_observed.items()},
        "kagglesdk/kernels_api_service.py": session_output_sha,
        "KAGGLE_FILES_CLI_INTERFACE_R2.json": sha256(interface_path),
        "KAGGLE_SESSION_OUTPUT_INTERFACE_R3.json": sha256(output_interface_path),
        "collect_profile_r2.py": sha256(HERE / "collect_profile_r2.py"),
    }


def collect_provider_inventory(
    *,
    cli_path: str,
    kernel: str,
    inventory_dir: Path,
    deadline: float,
    min_free_bytes: int,
    max_bytes: int,
    runner: Callable[..., dict[str, Any]] | None = None,
    max_pages: int = MAX_INVENTORY_PAGES,
) -> dict[str, Any]:
    """Fetch every paginated provider page and preserve its raw CLI output."""
    if runner is None:
        runner = terminal.run_bounded_command
    pages: list[dict[str, Any]] = []
    unique_files: dict[str, dict[str, Any]] = {}
    request_tokens: set[str] = set()
    token: str | None = None

    for page_index in range(max_pages):
        if token is not None and token in request_tokens:
            return {
                "complete": False,
                "reason": "pagination_token_cycle",
                "files": list(unique_files.values()),
                "pages": pages,
            }
        if token is not None:
            request_tokens.add(token)

        page_stdout = inventory_dir / f"provider_files_page_{page_index + 1:03d}.stdout.txt"
        page_stderr = inventory_dir / f"provider_files_page_{page_index + 1:03d}.stderr.txt"
        argv = [
            cli_path,
            "kernels",
            "files",
            "--format",
            "json",
            "--page-size",
            str(PAGE_SIZE),
        ]
        if token is not None:
            argv.extend(["--page-token", token])
        argv.append(f"{kernel}/{KERNEL_VERSION}")
        result = runner(
            argv,
            stdout_path=page_stdout,
            stderr_path=page_stderr,
            monitor_dir=inventory_dir,
            min_free_bytes=min_free_bytes,
            max_bytes=max_bytes,
            deadline=deadline,
        )
        page_record = {
            "page_index": page_index + 1,
            "request_page_token": token,
            "argv": argv,
            "command_result": result,
            "stdout_sha256": sha256(page_stdout) if page_stdout.is_file() else None,
            "stderr_sha256": sha256(page_stderr) if page_stderr.is_file() else None,
        }
        pages.append(page_record)
        if result.get("exit_code") != 0:
            return {
                "complete": False,
                "reason": "provider_files_command_failed",
                "files": list(unique_files.values()),
                "pages": pages,
            }
        if not page_stdout.is_file():
            return {
                "complete": False,
                "reason": "provider_files_stdout_missing",
                "files": list(unique_files.values()),
                "pages": pages,
            }
        try:
            page_files, next_token = parse_inventory_page(page_stdout.read_text())
        except (OSError, UnicodeError, ValueError) as error:
            page_record["parse_error"] = str(error)
            return {
                "complete": False,
                "reason": "provider_files_page_parse_failed",
                "files": list(unique_files.values()),
                "pages": pages,
            }

        for row in page_files:
            name = row["name"]
            if name in unique_files:
                page_record["duplicate_name"] = name
                return {
                    "complete": False,
                    "reason": "duplicate_provider_filename",
                    "files": list(unique_files.values()),
                    "pages": pages,
                }
            unique_files[name] = row
        page_record["row_count"] = len(page_files)
        page_record["next_page_token"] = next_token

        if next_token is None:
            files = [unique_files[name] for name in sorted(unique_files)]
            return {
                # A valid empty final page is a complete zero-file inventory; the
                # terminal collector still needs to fetch and preserve provider logs.
                "complete": True,
                "reason": None,
                "files": files,
                "pages": pages,
            }
        if next_token in request_tokens or next_token == token:
            return {
                "complete": False,
                "reason": "pagination_token_cycle",
                "files": list(unique_files.values()),
                "pages": pages,
            }
        token = next_token

    return {
        "complete": False,
        "reason": "provider_inventory_page_limit_reached",
        "files": [unique_files[name] for name in sorted(unique_files)],
        "pages": pages,
    }


def write_session_output_error_page(
    inventory_dir: Path,
    page_index: int,
    kernel: str,
    token: str | None,
    reason: str,
    file_count: int | None,
) -> dict[str, Any]:
    """Persist a safe page failure receipt without descriptor URLs or raw tokens."""
    page_path = inventory_dir / f"session_output_page_{page_index:03d}.json"
    payload = {
        "schema": "wnn.kaggle.session_output_page_error.r1",
        "page_index": page_index,
        "endpoint": "list_kernel_session_output",
        "kernel_slug": kernel,
        "request_page_token_sha256": hashlib.sha256(token.encode()).hexdigest() if token else None,
        "reason": reason,
        "file_count": file_count,
    }
    with page_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {
        "page_index": page_index,
        "receipt": page_path.name,
        "receipt_sha256": sha256(page_path),
        "reason": reason,
    }


def collect_session_output_inventory(
    *,
    kernel: str,
    inventory_dir: Path,
    deadline: float,
    fetch_page: Callable[[str | None, int], dict[str, Any]] | None = None,
    max_pages: int = MAX_INVENTORY_PAGES,
) -> dict[str, Any]:
    """List the exact output endpoint consumed by Kaggle's `kernels output` command."""
    owns_client = fetch_page is None
    if owns_client:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiListKernelSessionOutputRequest,
        )

        api = KaggleApi()
        api.authenticate()
        client = poll_provider._bounded_client(api)
        owner, slug, _version = api.parse_kernel_string(kernel)

        def fetch_page(token: str | None, page_size: int) -> dict[str, Any]:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            # The installed `kernels_output()` call uses this same session-output
            # endpoint and does not set request.version_label.
            api._set_paging(request, page_size, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            files = [
                {
                    "name": item.file_name,
                    "remote_size_bytes": None,
                }
                for item in (response.files or [])
            ]
            return {
                "files": files,
                "next_page_token": response.next_page_token or None,
                "response_log_sha256": hashlib.sha256((response.log or "").encode()).hexdigest(),
                "response_log_bytes": len((response.log or "").encode()),
            }

    pages: list[dict[str, Any]] = []
    unique_files: dict[str, dict[str, Any]] = {}
    requested_tokens: set[str] = set()
    token: str | None = None
    first_page_log: dict[str, Any] | None = None
    try:
        for page_index in range(max_pages):
            if time.monotonic() >= deadline:
                return {
                    "complete": False,
                    "reason": "session_output_inventory_deadline_reached",
                    "files": [unique_files[name] for name in sorted(unique_files)],
                    "pages": pages,
                }
            if token is not None and token in requested_tokens:
                return {
                    "complete": False,
                    "reason": "session_output_pagination_token_cycle",
                    "files": [unique_files[name] for name in sorted(unique_files)],
                    "pages": pages,
                }
            if token is not None:
                requested_tokens.add(token)
            try:
                page = fetch_page(token, PAGE_SIZE)
            except Exception as error:
                error_path = inventory_dir / f"session_output_page_{page_index + 1:03d}.json"
                payload = {
                    "page_index": page_index + 1,
                    "request_token_sha256": (
                        hashlib.sha256(token.encode()).hexdigest() if token else None
                    ),
                    "error_type": type(error).__name__,
                    "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                }
                with error_path.open("x", encoding="utf-8") as stream:
                    json.dump(payload, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                pages.append(
                    {
                        "page_index": page_index + 1,
                        "receipt": error_path.name,
                        "receipt_sha256": sha256(error_path),
                    }
                )
                return {
                    "complete": False,
                    "reason": "session_output_provider_request_failed",
                    "files": [unique_files[name] for name in sorted(unique_files)],
                    "pages": pages,
                }

            page_files = page.get("files") if isinstance(page, dict) else None
            next_token = page.get("next_page_token") if isinstance(page, dict) else None
            response_log_sha = page.get("response_log_sha256") if isinstance(page, dict) else None
            response_log_bytes = page.get("response_log_bytes") if isinstance(page, dict) else None
            if (
                not isinstance(page_files, list)
                or (next_token is not None and not isinstance(next_token, str))
                or (
                    not isinstance(response_log_sha, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", response_log_sha)
                    or isinstance(response_log_bytes, bool)
                    or not isinstance(response_log_bytes, int)
                    or response_log_bytes < 0
                )
            ):
                pages.append(
                    write_session_output_error_page(
                        inventory_dir,
                        page_index + 1,
                        kernel,
                        token,
                        "session_output_provider_schema_invalid",
                        len(page_files) if isinstance(page_files, list) else None,
                    )
                )
                return {
                    "complete": False,
                    "reason": "session_output_provider_schema_invalid",
                    "files": [unique_files[name] for name in sorted(unique_files)],
                    "pages": pages,
                }
            safe_files = []
            for index, row in enumerate(page_files):
                if (
                    not isinstance(row, dict)
                    or set(row) != {"name", "remote_size_bytes"}
                    or not isinstance(row["name"], str)
                    or not safe_provider_name(row["name"])
                    or row["remote_size_bytes"] is not None
                ):
                    reason = f"session_output_file_descriptor_invalid:{index}"
                    pages.append(
                        write_session_output_error_page(
                            inventory_dir,
                            page_index + 1,
                            kernel,
                            token,
                            reason,
                            len(page_files),
                        )
                    )
                    return {
                        "complete": False,
                        "reason": reason,
                        "files": [unique_files[name] for name in sorted(unique_files)],
                        "pages": pages,
                    }
                name = row["name"]
                if name in unique_files:
                    reason = f"duplicate_session_output_filename:{name}"
                    pages.append(
                        write_session_output_error_page(
                            inventory_dir,
                            page_index + 1,
                            kernel,
                            token,
                            "duplicate_session_output_filename",
                            len(page_files),
                        )
                    )
                    return {
                        "complete": False,
                        "reason": reason,
                        "files": [unique_files[item] for item in sorted(unique_files)],
                        "pages": pages,
                    }
                unique_files[name] = row
                safe_files.append(row)

            request_token_sha = hashlib.sha256(token.encode()).hexdigest() if token else None
            next_token_sha = hashlib.sha256(next_token.encode()).hexdigest() if next_token else None
            page_record = {
                "schema": "wnn.kaggle.session_output_page.r1",
                "page_index": page_index + 1,
                "endpoint": "list_kernel_session_output",
                "kernel_slug": kernel,
                "version_label_sent": None,
                "request_page_token_sha256": request_token_sha,
                "page_size": PAGE_SIZE,
                "files": safe_files,
                "next_page_token_sha256": next_token_sha,
                "response_log_sha256": response_log_sha,
                "response_log_bytes": response_log_bytes,
            }
            if page_index == 0:
                first_page_log = {
                    "sha256": response_log_sha,
                    "size_bytes": response_log_bytes,
                }
            page_path = inventory_dir / f"session_output_page_{page_index + 1:03d}.json"
            with page_path.open("x", encoding="utf-8") as stream:
                json.dump(page_record, stream, indent=2, sort_keys=True)
                stream.write("\n")
            pages.append(
                {
                    "page_index": page_index + 1,
                    "receipt": page_path.name,
                    "receipt_sha256": sha256(page_path),
                    "row_count": len(safe_files),
                    "request_page_token_sha256": request_token_sha,
                    "next_page_token_sha256": next_token_sha,
                }
            )
            if next_token is None:
                return {
                    "complete": True,
                    "reason": None,
                    "endpoint": "list_kernel_session_output",
                    "kernel_version_pinned_by_endpoint": False,
                    "remote_size_or_checksum_available": False,
                    "response_log": first_page_log,
                    "files": [unique_files[name] for name in sorted(unique_files)],
                    "pages": pages,
                }
            if next_token in requested_tokens or next_token == token:
                return {
                    "complete": False,
                    "reason": "session_output_pagination_token_cycle",
                    "files": [unique_files[name] for name in sorted(unique_files)],
                    "pages": pages,
                }
            token = next_token
    finally:
        if owns_client:
            client.__exit__(None, None, None)

    return {
        "complete": False,
        "reason": "session_output_page_limit_reached",
        "files": [unique_files[name] for name in sorted(unique_files)],
        "pages": pages,
    }


def compare_provider_inventory(
    provider_files: list[dict[str, Any]],
    attempt_dir: Path,
    provider_response_log: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require exact nonempty provider/local filename and byte-size agreement."""
    output_dir = attempt_dir / "outputs"
    downloaded: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    if not output_dir.is_dir():
        return {
            "matches": False,
            "reason": "downloaded_outputs_directory_missing",
            "provider_file_count": len(provider_files),
            "downloaded_file_count": 0,
            "missing_downloads": [row.get("name") for row in provider_files],
            "unexpected_downloads": [],
            "size_mismatches": [],
            "per_file_sha256": {},
        }

    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink():
            errors.append(
                f"symlink_in_downloaded_outputs:{path.relative_to(output_dir).as_posix()}"
            )
            continue
        if not path.is_file():
            continue
        name = path.relative_to(output_dir).as_posix()
        if not safe_provider_name(name):
            errors.append(f"unsafe_downloaded_filename:{name}")
            continue
        downloaded[name] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}

    provider_by_name: dict[str, int | None] = {}
    remote_sizes_available = True
    for row in provider_files:
        name = row.get("name")
        size = row.get("size", row.get("remote_size_bytes"))
        if not isinstance(name, str) or not safe_provider_name(name):
            errors.append(f"unsafe_provider_filename:{name}")
            continue
        if size is None:
            remote_sizes_available = False
        elif isinstance(size, bool) or not isinstance(size, int) or size < 0:
            errors.append(f"invalid_provider_size:{name}")
            continue
        if name in provider_by_name:
            errors.append(f"duplicate_provider_filename:{name}")
            continue
        provider_by_name[name] = size

    auxiliary_cli_log = downloaded.pop(KERNEL_OUTPUT_LOG_NAME, None)
    if auxiliary_cli_log is not None and KERNEL_OUTPUT_LOG_NAME in provider_by_name:
        errors.append(f"provider_name_conflicts_with_cli_generated_log:{KERNEL_OUTPUT_LOG_NAME}")
    provider_names = set(provider_by_name)
    downloaded_names = set(downloaded)
    missing = sorted(provider_names - downloaded_names)
    unexpected = sorted(downloaded_names - provider_names)
    size_mismatches = [
        {
            "name": name,
            "provider_size": provider_by_name[name],
            "downloaded_size": downloaded[name]["size_bytes"],
        }
        for name in sorted(provider_names & downloaded_names)
        if provider_by_name[name] is not None
        and provider_by_name[name] != downloaded[name]["size_bytes"]
    ]
    response_log_match: bool | None = None
    response_log_error = None
    if provider_response_log is None:
        response_log_match = False
        response_log_error = "provider_response_log_metadata_missing"
    else:
        expected_log_size = provider_response_log.get("size_bytes")
        expected_log_sha = provider_response_log.get("sha256")
        if (
            isinstance(expected_log_size, bool)
            or not isinstance(expected_log_size, int)
            or expected_log_size < 0
            or not isinstance(expected_log_sha, str)
            or len(expected_log_sha) != 64
        ):
            response_log_error = "provider_response_log_metadata_invalid"
        elif expected_log_size == 0:
            response_log_match = auxiliary_cli_log is None
            if not response_log_match:
                response_log_error = "unexpected_cli_response_log"
        else:
            response_log_match = (
                auxiliary_cli_log is not None
                and auxiliary_cli_log["size_bytes"] == expected_log_size
                and auxiliary_cli_log["sha256"] == expected_log_sha
            )
            if not response_log_match:
                response_log_error = "cli_response_log_bytes_do_not_match_session_output_metadata"
    if response_log_error:
        errors.append(response_log_error)
    matches = not errors and not missing and not unexpected and not size_mismatches
    return {
        "matches": matches,
        "reason": None if matches else "provider_download_inventory_mismatch",
        "provider_file_count": len(provider_names),
        "downloaded_file_count": len(downloaded_names),
        "missing_downloads": missing,
        "unexpected_downloads": unexpected,
        "size_mismatches": size_mismatches,
        "remote_sizes_available": remote_sizes_available,
        "remote_content_hashes_available": False,
        "remote_byte_identity_claim": False,
        "response_log_match": response_log_match,
        "provider_response_log": provider_response_log,
        "errors": errors,
        "per_file_sha256": downloaded,
        "kernels_output_response_log": (
            {"name": KERNEL_OUTPUT_LOG_NAME, **auxiliary_cli_log}
            if auxiliary_cli_log is not None
            else None
        ),
    }


def next_inventory_run_dir(attempt_dir: Path) -> Path:
    """Create an immutable per-invocation directory for pagination evidence."""
    existing = []
    for path in attempt_dir.glob("provider_inventory_run_*"):
        match = re.fullmatch(r"provider_inventory_run_([0-9]{3,})", path.name)
        if match:
            existing.append(int(match.group(1)))
    run_dir = attempt_dir / f"provider_inventory_run_{max(existing, default=0) + 1:03d}"
    run_dir.mkdir(mode=0o700)
    return run_dir


def next_collection_run_dir(packet_dir: Path) -> Path:
    """Create an immutable wrapper-run directory for summaries and preflight pages."""
    existing = []
    for path in packet_dir.glob("collection_run_*"):
        match = re.fullmatch(r"collection_run_([0-9]{3,})", path.name)
        if match:
            existing.append(int(match.group(1)))
    run_dir = packet_dir / f"collection_run_{max(existing, default=0) + 1:03d}"
    run_dir.mkdir(mode=0o700)
    return run_dir


def write_immutable_json(path: Path, value: dict[str, Any]) -> str:
    """Write a new receipt exactly once; never replace an earlier run's bytes."""
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)
    return hashlib.sha256(payload).hexdigest()


def inventory_log_collisions(inventory: dict[str, Any]) -> list[str]:
    """Return provider files that CLI output would overwrite with its response log."""
    if inventory.get("complete") is not True:
        return []
    return sorted(
        row["name"]
        for row in inventory.get("files", [])
        if isinstance(row, dict) and row.get("name") == KERNEL_OUTPUT_LOG_NAME
    )


def write_latest_pointer(path: Path, target_path: Path, digest: str) -> None:
    """Atomically update a convenience pointer while keeping run receipts immutable."""
    terminal.atomic_write_json(
        path,
        {
            "schema": "wnn.clean_t3_f0_profile_latest_receipt.r1",
            "target": target_path.relative_to(HERE).as_posix(),
            "sha256": digest,
        },
    )


def preserve_partial_collector_receipt(
    receipt_path: Path, artifact_dir: Path, history_dir: Path
) -> dict[str, Any] | None:
    """Archive an incomplete top-level receipt before the shared collector replaces it."""
    if not receipt_path.exists():
        return None
    if receipt_path.is_symlink():
        raise ValueError("refusing to archive a symlinked collector receipt")
    if terminal.collection_is_complete(receipt_path, artifact_dir):
        return None
    size = receipt_path.stat().st_size
    if size > 64 * 1024**2:
        raise ValueError("partial collector receipt exceeds the 64 MiB archive bound")

    history_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    existing = []
    for path in history_dir.glob("collector_receipt_*.json"):
        match = re.fullmatch(r"collector_receipt_([0-9]{3,})\.json", path.name)
        if match:
            existing.append(int(match.group(1)))
    archived_path = history_dir / f"collector_receipt_{max(existing, default=0) + 1:03d}.json"
    digest = sha256(receipt_path)
    with receipt_path.open("rb") as source, archived_path.open("xb") as destination:
        shutil.copyfileobj(source, destination, length=1 << 20)
    os.chmod(archived_path, 0o600)
    if sha256(archived_path) != digest:
        raise ValueError("archived collector receipt does not match its source bytes")
    return {
        "path": archived_path.relative_to(ROOT).as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }


def resolve_attempt_dir(
    result: dict[str, Any], receipt_path: Path, artifact_dir: Path
) -> tuple[Path | None, str | None]:
    """Resolve a fresh attempt or the exact attempt stored in an existing receipt."""
    attempt_name = result.get("attempt")
    if result.get("state") == "already_collected":
        try:
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("kernel") != KERNEL:
                return None, "existing_receipt_kernel_mismatch"
            attempt_name = receipt["terminal_collection"]["attempt_directory"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            return None, f"existing_receipt_attempt_unreadable:{type(error).__name__}"
    if not isinstance(attempt_name, str) or not re.fullmatch(r"attempt_[0-9]{3,}", attempt_name):
        return None, "collector_attempt_directory_unavailable"

    root = artifact_dir.resolve()
    candidate = artifact_dir / attempt_name
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None, "collector_attempt_directory_invalid"
    if candidate.is_symlink() or not resolved.is_dir():
        return None, "collector_attempt_directory_not_regular"
    return resolved, None


def terminal_evidence_complete(
    low_level_complete: bool,
    inventory: dict[str, Any],
    comparison: dict[str, Any] | None,
) -> bool:
    """Require successful logs/outputs plus complete matching provider identity."""
    return bool(
        low_level_complete
        and inventory.get("complete") is True
        and comparison is not None
        and comparison.get("matches") is True
    )


def main() -> int:
    """Refresh status and collect complete terminal evidence for the exact profile."""
    bindings = verify_root_bindings()
    status_target_sha256 = verify_status_target()
    source_bindings = verify_collector_sources()
    snapshot = poll_provider.collect(STATUS_TARGET_PATH)
    poll_provider.append_jsonl(HERE / "provider_quota_status_r2.jsonl", snapshot)
    status_row = terminal._target_status(snapshot, KERNEL)
    if status_row is None:
        raise ValueError("exact profile kernel is absent from provider snapshot")
    provider_status = status_row.get("status")
    if not provider_status_available(status_row):
        print(
            json.dumps(
                {
                    "state": "provider_status_unavailable",
                    "status_row": status_row,
                    "quota": snapshot["quota"],
                },
                sort_keys=True,
            )
        )
        return 1
    if not terminal.terminal_state(provider_status):
        print(
            json.dumps(
                {
                    "state": "provider_not_terminal",
                    "status": status_row,
                    "quota": snapshot["quota"],
                },
                sort_keys=True,
            )
        )
        return 0

    artifact_rel = "attempts"
    receipt_name = "TERMINAL_COLLECTION_RECEIPT_R2.json"
    target = {
        "kernel": KERNEL,
        "resource": "gpu",
        "root_admission_sha256": bindings["ROOT_PROFILE_DISPATCH_ADMISSION_R1.json"]["sha256"],
        "root_admission": "root_bindings/ROOT_PROFILE_DISPATCH_ADMISSION_R1.json",
        "push_receipt": "root_bindings/ROOT_PROFILE_PUSH_R1.json",
        "asset_ready_receipt": "root_bindings/ROOT_PRIVATE_DATASET_READY_READBACK_R1.json",
        "artifact_dir": artifact_rel,
        "terminal_receipt": receipt_name,
    }
    config = {
        "repo_root": str(ROOT),
        "config_dir": str(HERE),
        "minimum_free_bytes": 4 * 1024**3,
        "maximum_download_bytes": 8 * 1024**3,
        "max_runtime_seconds": 420,
        "max_attempts": 3,
        "kaggle_cli": str(MAIN / ".venv/bin/kaggle"),
        "expected_cli_version": "Kaggle CLI 2.2.4",
    }
    collection_run_dir = next_collection_run_dir(HERE)
    inventory_run_dir = collection_run_dir / "provider_inventory"
    inventory_run_dir.mkdir(mode=0o700)
    inventory = collect_session_output_inventory(
        kernel=KERNEL,
        inventory_dir=inventory_run_dir,
        deadline=time.monotonic() + 210,
    )
    collisions = inventory_log_collisions(inventory)
    archived_receipt = None
    result: dict[str, Any]
    if not inventory.get("complete"):
        result = {
            "kernel": KERNEL,
            "state": "provider_inventory_preflight_incomplete",
            "reason": inventory.get("reason"),
        }
    elif collisions:
        # The CLI writes its response log into the output directory after downloading
        # provider files. Refuse this call before it can overwrite a same-named artifact.
        result = {
            "kernel": KERNEL,
            "state": "provider_output_log_name_collision",
            "colliding_names": collisions,
        }
    else:
        archived_receipt = preserve_partial_collector_receipt(
            HERE / receipt_name, HERE / artifact_rel, HERE / "collector_receipt_history"
        )
        result = terminal.collect_terminal_target(
            target=target,
            status_row=status_row,
            config=config,
            repo_root=ROOT,
            config_dir=HERE,
        )
    artifact_dir = HERE / artifact_rel
    receipt_path = HERE / receipt_name
    attempt_dir, attempt_error = resolve_attempt_dir(result, receipt_path, artifact_dir)
    comparison = None
    if attempt_dir is not None and inventory["complete"]:
        comparison = compare_provider_inventory(
            inventory["files"],
            attempt_dir,
            provider_response_log=inventory.get("response_log"),
        )

    low_level_complete = terminal.collection_is_complete(receipt_path, artifact_dir)
    overall_complete = terminal_evidence_complete(low_level_complete, inventory, comparison)
    inventory_path = collection_run_dir / "PROVIDER_FILES_INVENTORY_R2.json"
    inventory_sha256 = write_immutable_json(inventory_path, inventory)
    handoff = {
        "schema": "wnn.clean_t3_f0_profile_terminal_handoff.r2",
        "kernel": KERNEL,
        "kernel_version_at_push": KERNEL_VERSION,
        "dataset": DATASET,
        "provider_status": status_row,
        "quota_snapshot": snapshot["quota"],
        "root_bindings": bindings,
        "provider_status_target_sha256": status_target_sha256,
        "collector_source_bindings": source_bindings,
        "parent_r1_collection_commit": R1_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "provider_listing_version_scope": (
            "The installed Kaggle kernels_output implementation requests list_kernel_session_output "
            "with user_name and kernel_slug and leaves version_label unset. This is current-slug output, "
            "not API-version-pinned. Bind this unique slug to root push version 1 and the actual returned "
            "runtime spec/source hashes; do not claim endpoint version pinning."
        ),
        "low_level_logs_outputs_collected": low_level_complete,
        "resolved_attempt_directory": (
            attempt_dir.relative_to(artifact_dir).as_posix() if attempt_dir else None
        ),
        "collection_run_directory": collection_run_dir.relative_to(HERE).as_posix(),
        "inventory_run_directory": inventory_run_dir.relative_to(HERE).as_posix(),
        "attempt_resolution_error": attempt_error,
        "provider_files_inventory": inventory,
        "provider_file_inventory_semantics": (
            "Actual list_kernel_session_output descriptors used by kernels output; each exposes file_name "
            "and URL only. URLs are never persisted. The provider returns no size/checksum, so local "
            "byte sizes and SHA-256 are recorded without claiming remote-byte hash verification."
        ),
        "kernels_files_diagnostic": {
            "versioned": {
                "path": "raw_terminal_capture_r1/RAW_PROVIDER_FILES_INVENTORY_R1.json",
                "sha256": sha256(
                    HERE / "raw_terminal_capture_r1/RAW_PROVIDER_FILES_INVENTORY_R1.json"
                ),
                "authoritative_for_session_outputs": False,
            },
            "unsuffixed": {
                "path": "raw_terminal_capture_r1/RAW_PROVIDER_FILES_INVENTORY_UNSUFFIXED_R1.json",
                "sha256": sha256(
                    HERE / "raw_terminal_capture_r1/RAW_PROVIDER_FILES_INVENTORY_UNSUFFIXED_R1.json"
                ),
                "authoritative_for_session_outputs": False,
            },
        },
        "provider_files_inventory_receipt": {
            "path": inventory_path.relative_to(HERE).as_posix(),
            "sha256": inventory_sha256,
        },
        "provider_download_comparison": comparison,
        "terminal_collector_receipt": {
            "path": receipt_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(receipt_path) if receipt_path.is_file() else None,
            "collection_complete": low_level_complete,
        },
        "previous_partial_receipt_archived": archived_receipt,
        "collector_receipt_history": (
            terminal.file_inventory(HERE / "collector_receipt_history")
            if (HERE / "collector_receipt_history").is_dir()
            else {}
        ),
        "download_bound": {
            "maximum_download_bytes_per_attempt": config["maximum_download_bytes"],
            "minimum_free_bytes_global_filesystem_guard": config["minimum_free_bytes"],
            "scope": "per attempt; prior attempts remain retained and the global free-space floor is checked throughout each download",
        },
        "collection_complete": overall_complete,
        "profile_success": False,
        "profile_success_note": (
            "Collection completeness is independent of scientific/profile success; "
            "provider terminal status and per-job outcomes require separate interpretation."
        ),
        "finite_gradient_counter": "not_produced_by_frozen_profile_command",
        "checkpoint_reload_evidence": "not_produced_by_frozen_profile_command",
        "fit_feasibility": "requires root interpretation of actual profile logs/resources",
        "attempt_files": terminal.file_inventory(attempt_dir) if attempt_dir else {},
    }
    handoff_path = collection_run_dir / "PROFILE_TERMINAL_HANDOFF_R2.json"
    handoff_sha256 = write_immutable_json(handoff_path, handoff)
    write_latest_pointer(
        HERE / "PROVIDER_FILES_INVENTORY_R2_LATEST.json", inventory_path, inventory_sha256
    )
    write_latest_pointer(
        HERE / "PROFILE_TERMINAL_HANDOFF_R2_LATEST.json", handoff_path, handoff_sha256
    )
    print(
        json.dumps(
            {
                "state": "terminal_evidence_collected"
                if overall_complete
                else "terminal_collection_or_inventory_incomplete",
                "provider_status": status_row.get("status"),
                "profile_success": False,
                "collection_complete": overall_complete,
                "low_level_complete": low_level_complete,
                "inventory_complete": inventory.get("complete"),
                "comparison_matches": comparison and comparison.get("matches"),
                "attempt": handoff["resolved_attempt_directory"],
                "receipt": str(handoff_path),
            },
            sort_keys=True,
        )
    )
    return 0 if overall_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
