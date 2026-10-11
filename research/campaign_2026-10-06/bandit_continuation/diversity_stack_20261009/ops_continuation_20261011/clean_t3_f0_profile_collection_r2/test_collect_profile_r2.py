from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from collect_profile_r2 import (
    collect_provider_inventory,
    collect_session_output_inventory,
    compare_provider_inventory,
    inventory_log_collisions,
    next_collection_run_dir,
    next_inventory_run_dir,
    parse_inventory_page,
    preserve_partial_collector_receipt,
    provider_status_available,
    resolve_attempt_dir,
    terminal_evidence_complete,
    write_immutable_json,
)


class InventoryPageTests(unittest.TestCase):
    def test_parses_cli_token_preamble_and_json_rows(self) -> None:
        stdout = (
            "Next Page Token = token-page-2\n"
            '[{"name":"out/profile.log","size":17,'
            '"creationDate":"2026-10-11T18:00:00Z"}]\n'
        )
        files, token = parse_inventory_page(stdout)
        self.assertEqual(token, "token-page-2")
        self.assertEqual(files[0]["name"], "out/profile.log")
        self.assertEqual(files[0]["size"], 17)

    def test_parses_final_page_without_token(self) -> None:
        files, token = parse_inventory_page(
            '[{"name":"out/result.json","size":2,"creationDate":null}]'
        )
        self.assertEqual(token, None)
        self.assertEqual(len(files), 1)

    def test_rejects_empty_truncated_and_wrong_schema_pages(self) -> None:
        invalid = [
            "",
            "No files found",
            "Next Page Token = token-only",
            '[{"name":"../escape","size":1,"creationDate":"today"}]',
            '[{"name":"out/x","size":"1","creationDate":"today"}]',
            '[{"name":"out/x","size":1}]',
            '[{"name":"out/\u0001bad","size":1,"creationDate":"today"}]',
        ]
        for stdout in invalid:
            with self.subTest(stdout=stdout), self.assertRaises(ValueError):
                parse_inventory_page(stdout)


class ProviderInventoryTests(unittest.TestCase):
    def test_poll_error_row_is_unavailable_not_nonterminal(self) -> None:
        self.assertFalse(
            provider_status_available({"monitoring_state": "poll_error_retry_next_timer_tick"})
        )
        self.assertTrue(provider_status_available({"status": "KernelWorkerStatus.ERROR"}))

    def test_collects_all_pages_and_keeps_paged_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            inventory_dir = next_inventory_run_dir(attempt)
            responses = [
                'Next Page Token = second\n[{"name":"out/a","size":3,"creationDate":"t1"}]',
                '[{"name":"out/b","size":4,"creationDate":"t2"}]',
            ]
            seen_argv: list[list[str]] = []

            def runner(command: list[str], **kwargs: Any) -> dict[str, Any]:
                seen_argv.append(command)
                kwargs["stdout_path"].write_text(responses[len(seen_argv) - 1])
                kwargs["stderr_path"].write_text("")
                return {"exit_code": 0, "elapsed_seconds": 0.01}

            result = collect_provider_inventory(
                cli_path="kaggle",
                kernel="owner/kernel",
                inventory_dir=inventory_dir,
                deadline=time.monotonic() + 100,
                min_free_bytes=0,
                max_bytes=1_000_000,
                runner=runner,
            )
            self.assertTrue(result["complete"])
            self.assertEqual([row["name"] for row in result["files"]], ["out/a", "out/b"])
            self.assertEqual(len(result["pages"]), 2)
            self.assertIn("--page-token", seen_argv[1])
            self.assertIn("second", seen_argv[1])

    def test_duplicate_page_filename_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            inventory_dir = next_inventory_run_dir(attempt)
            response = '[{"name":"out/a","size":3,"creationDate":"t"}]'
            calls = 0

            def runner(command: list[str], **kwargs: Any) -> dict[str, Any]:
                nonlocal calls
                calls += 1
                stdout = "Next Page Token = second\n" + response if calls == 1 else response
                kwargs["stdout_path"].write_text(stdout)
                kwargs["stderr_path"].write_text("")
                return {"exit_code": 0}

            result = collect_provider_inventory(
                cli_path="kaggle",
                kernel="owner/kernel",
                inventory_dir=inventory_dir,
                deadline=100,
                min_free_bytes=0,
                max_bytes=1_000_000,
                runner=runner,
            )
            self.assertFalse(result["complete"])
            self.assertEqual(result["reason"], "duplicate_provider_filename")

    def test_valid_empty_inventory_still_allows_log_collection_and_exact_zero_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory_dir = root / "inventory"
            inventory_dir.mkdir()

            def runner(command: list[str], **kwargs: Any) -> dict[str, Any]:
                kwargs["stdout_path"].write_text("[]\n")
                kwargs["stderr_path"].write_text("")
                return {"exit_code": 0}

            inventory = collect_provider_inventory(
                cli_path="kaggle",
                kernel="owner/kernel",
                inventory_dir=inventory_dir,
                deadline=100,
                min_free_bytes=0,
                max_bytes=1_000_000,
                runner=runner,
            )
            self.assertTrue(inventory["complete"])
            self.assertEqual(inventory["files"], [])

            attempt = root / "attempt_001"
            output_dir = attempt / "outputs"
            output_dir.mkdir(parents=True)
            (output_dir / "wnn-clean-t3-f0-teachers-20261011-r1-runner.log").write_text(
                "provider response log"
            )
            response_log = {
                "sha256": hashlib.sha256(b"provider response log").hexdigest(),
                "size_bytes": len(b"provider response log"),
            }
            comparison = compare_provider_inventory(
                inventory["files"], attempt, provider_response_log=response_log
            )
            self.assertTrue(comparison["matches"])
            self.assertEqual(comparison["provider_file_count"], 0)
            self.assertEqual(comparison["downloaded_file_count"], 0)

    def test_collection_run_summaries_are_unique_and_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = next_collection_run_dir(root)
            first_receipt = first / "PROFILE_TERMINAL_HANDOFF_R2.json"
            first_digest = write_immutable_json(first_receipt, {"run": 1})
            second = next_collection_run_dir(root)
            second_receipt = second / "PROFILE_TERMINAL_HANDOFF_R2.json"
            second_digest = write_immutable_json(second_receipt, {"run": 2})
            self.assertEqual(first.name, "collection_run_001")
            self.assertEqual(second.name, "collection_run_002")
            self.assertNotEqual(first_digest, second_digest)
            self.assertEqual(json.loads(first_receipt.read_text()), {"run": 1})
            self.assertEqual(json.loads(second_receipt.read_text()), {"run": 2})
            with self.assertRaises(FileExistsError):
                write_immutable_json(first_receipt, {"run": 3})
            self.assertEqual(json.loads(first_receipt.read_text()), {"run": 1})

    def test_partial_low_level_receipt_history_preserves_each_retry_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "TERMINAL_COLLECTION_RECEIPT_R2.json"
            artifacts = root / "attempts"
            artifacts.mkdir()
            history = root / "collector_receipt_history"
            first_bytes = b'{"attempt": 1, "incomplete": true}\n'
            receipt.write_bytes(first_bytes)
            with (
                patch("collect_profile_r2.ROOT", root),
                patch("collect_profile_r2.terminal.collection_is_complete", return_value=False),
            ):
                first = preserve_partial_collector_receipt(receipt, artifacts, history)
                second_bytes = b'{"attempt": 2, "incomplete": true}\n'
                receipt.write_bytes(second_bytes)
                second = preserve_partial_collector_receipt(receipt, artifacts, history)
            self.assertEqual((history / "collector_receipt_001.json").read_bytes(), first_bytes)
            self.assertEqual((history / "collector_receipt_002.json").read_bytes(), second_bytes)
            self.assertEqual(first["size_bytes"], len(first_bytes))
            self.assertEqual(second["size_bytes"], len(second_bytes))
            receipt.write_text('{"attempt": 3, "complete": true}\n')
            with (
                patch("collect_profile_r2.ROOT", root),
                patch("collect_profile_r2.terminal.collection_is_complete", return_value=True),
            ):
                self.assertIsNone(preserve_partial_collector_receipt(receipt, artifacts, history))
            self.assertEqual(len(list(history.glob("collector_receipt_*.json"))), 2)

    def test_provider_response_log_collision_is_detected_before_download(self) -> None:
        inventory = {
            "complete": True,
            "files": [
                {
                    "name": "wnn-clean-t3-f0-teachers-20261011-r1-runner.log",
                    "size": 10,
                }
            ],
        }
        self.assertEqual(
            inventory_log_collisions(inventory),
            ["wnn-clean-t3-f0-teachers-20261011-r1-runner.log"],
        )
        self.assertEqual(inventory_log_collisions({"complete": False, "files": []}), [])

    def test_downloaded_outputs_require_exact_nonempty_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            output_dir = attempt / "outputs"
            output_dir.mkdir()
            (output_dir / "a.bin").write_bytes(b"abc")
            (output_dir / "b.bin").write_bytes(b"xy")
            (output_dir / "wnn-clean-t3-f0-teachers-20261011-r1-runner.log").write_bytes(
                b"CLI response log"
            )
            exact = [
                {"name": "a.bin", "size": 3, "creationDate": "t"},
                {"name": "b.bin", "size": 2, "creationDate": "t"},
            ]
            response_log = {
                "sha256": hashlib.sha256(b"CLI response log").hexdigest(),
                "size_bytes": len(b"CLI response log"),
            }
            complete = compare_provider_inventory(
                exact, attempt, provider_response_log=response_log
            )
            self.assertTrue(complete["matches"])
            self.assertEqual(set(complete["per_file_sha256"]), {"a.bin", "b.bin"})
            self.assertEqual(
                complete["kernels_output_response_log"]["name"],
                "wnn-clean-t3-f0-teachers-20261011-r1-runner.log",
            )

            missing = compare_provider_inventory(
                exact[:1], attempt, provider_response_log=response_log
            )
            self.assertFalse(missing["matches"])
            self.assertEqual(missing["unexpected_downloads"], ["b.bin"])

            bad_size = compare_provider_inventory(
                [{"name": "a.bin", "size": 8, "creationDate": "t"}],
                attempt,
                provider_response_log=response_log,
            )
            self.assertFalse(bad_size["matches"])
            self.assertEqual(bad_size["size_mismatches"][0]["name"], "a.bin")

            collision = compare_provider_inventory(
                exact
                + [
                    {
                        "name": "wnn-clean-t3-f0-teachers-20261011-r1-runner.log",
                        "size": len(b"CLI response log"),
                        "creationDate": "t",
                    }
                ],
                attempt,
                provider_response_log=response_log,
            )
            self.assertFalse(collision["matches"])
            self.assertTrue(
                any(
                    error.startswith("provider_name_conflicts_with_cli_generated_log")
                    for error in collision["errors"]
                )
            )

            empty = compare_provider_inventory([], attempt, provider_response_log=response_log)
            self.assertFalse(empty["matches"])
            missing_log_metadata = compare_provider_inventory(exact, attempt)
            self.assertFalse(missing_log_metadata["matches"])
            self.assertIn("provider_response_log_metadata_missing", missing_log_metadata["errors"])

            self.assertTrue(terminal_evidence_complete(True, {"complete": True}, {"matches": True}))
            self.assertFalse(
                terminal_evidence_complete(False, {"complete": True}, {"matches": True})
            )
            self.assertFalse(
                terminal_evidence_complete(True, {"complete": False}, {"matches": True})
            )
            self.assertFalse(
                terminal_evidence_complete(True, {"complete": True}, {"matches": False})
            )

    def test_session_output_endpoint_paginates_without_persisting_urls_or_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inventory_dir = Path(directory)
            page_calls: list[tuple[str | None, int]] = []
            log = "safe provider log"

            def fetch_page(token: str | None, page_size: int) -> dict[str, Any]:
                page_calls.append((token, page_size))
                if token is None:
                    return {
                        "files": [
                            {"name": "experiment/nested/result.bin", "remote_size_bytes": None}
                        ],
                        "next_page_token": "secret-page-token",
                        "response_log_sha256": hashlib.sha256(log.encode()).hexdigest(),
                        "response_log_bytes": len(log.encode()),
                    }
                return {
                    "files": [{"name": "runtime_specs/profile.json", "remote_size_bytes": None}],
                    "next_page_token": None,
                    "response_log_sha256": hashlib.sha256(b"").hexdigest(),
                    "response_log_bytes": 0,
                }

            result = collect_session_output_inventory(
                kernel="owner/kernel",
                inventory_dir=inventory_dir,
                deadline=time.monotonic() + 100,
                fetch_page=fetch_page,
            )
            self.assertTrue(result["complete"])
            self.assertEqual(len(page_calls), 2)
            self.assertEqual(page_calls[0], (None, 200))
            self.assertEqual(page_calls[1], ("secret-page-token", 200))
            self.assertEqual(
                [row["name"] for row in result["files"]],
                ["experiment/nested/result.bin", "runtime_specs/profile.json"],
            )
            self.assertEqual(result["response_log"]["size_bytes"], len(log.encode()))
            serialized = "".join(path.read_text() for path in inventory_dir.glob("*.json"))
            self.assertNotIn("secret-page-token", serialized)
            self.assertNotIn("https://", serialized)
            self.assertTrue(all(row["remote_size_bytes"] is None for row in result["files"]))

    def test_empty_session_output_is_valid_and_response_log_closes_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory_dir = root / "inventory"
            inventory_dir.mkdir()
            log = "kernel log"
            inventory = collect_session_output_inventory(
                kernel="owner/kernel",
                inventory_dir=inventory_dir,
                deadline=time.monotonic() + 100,
                fetch_page=lambda _token, _size: {
                    "files": [],
                    "next_page_token": None,
                    "response_log_sha256": hashlib.sha256(log.encode()).hexdigest(),
                    "response_log_bytes": len(log.encode()),
                },
            )
            self.assertTrue(inventory["complete"])
            self.assertEqual(inventory["files"], [])
            attempt = root / "attempt_002"
            output_dir = attempt / "outputs"
            output_dir.mkdir(parents=True)
            (output_dir / "wnn-clean-t3-f0-teachers-20261011-r1-runner.log").write_text(log)
            comparison = compare_provider_inventory(
                inventory["files"], attempt, provider_response_log=inventory["response_log"]
            )
            self.assertTrue(comparison["matches"])
            self.assertEqual(comparison["provider_file_count"], 0)
            self.assertEqual(comparison["downloaded_file_count"], 0)
            self.assertTrue(comparison["response_log_match"])
            wrong_log = {
                "sha256": hashlib.sha256(b"different").hexdigest(),
                "size_bytes": len(log.encode()),
            }
            mismatch = compare_provider_inventory(
                inventory["files"], attempt, provider_response_log=wrong_log
            )
            self.assertFalse(mismatch["matches"])
            self.assertFalse(mismatch["response_log_match"])

    def test_session_output_schema_failure_writes_safe_failure_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inventory_dir = Path(directory)
            result = collect_session_output_inventory(
                kernel="owner/kernel",
                inventory_dir=inventory_dir,
                deadline=time.monotonic() + 100,
                fetch_page=lambda _token, _size: {
                    "files": [{"name": "../bad", "remote_size_bytes": None}],
                    "next_page_token": None,
                    "response_log_sha256": hashlib.sha256(b"").hexdigest(),
                    "response_log_bytes": 0,
                },
            )
            self.assertFalse(result["complete"])
            self.assertEqual(result["reason"], "session_output_file_descriptor_invalid:0")
            saved = next(inventory_dir.glob("session_output_page_*.json"))
            self.assertNotIn("../bad", saved.read_text())

    def test_duplicate_session_output_names_fail_closed_with_safe_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inventory_dir = Path(directory)
            result = collect_session_output_inventory(
                kernel="owner/kernel",
                inventory_dir=inventory_dir,
                deadline=time.monotonic() + 100,
                fetch_page=lambda _token, _size: {
                    "files": [
                        {"name": "nested/result.bin", "remote_size_bytes": None},
                        {"name": "nested/result.bin", "remote_size_bytes": None},
                    ],
                    "next_page_token": None,
                    "response_log_sha256": hashlib.sha256(b"").hexdigest(),
                    "response_log_bytes": 0,
                },
            )
            self.assertFalse(result["complete"])
            self.assertEqual(
                result["reason"], "duplicate_session_output_filename:nested/result.bin"
            )
            saved = next(inventory_dir.glob("session_output_page_*.json"))
            self.assertIn("duplicate_session_output_filename", saved.read_text())
            self.assertNotIn("nested/result.bin", saved.read_text())

    def test_already_collected_resolves_attempt_002_from_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "attempts"
            attempt = artifacts / "attempt_002"
            attempt.mkdir(parents=True)
            run_001 = attempt / "provider_inventory_run_001"
            run_001.mkdir()
            prior_page = run_001 / "provider_files_page_001.stdout.txt"
            prior_page.write_text("previous page receipt")
            receipt = root / "TERMINAL_COLLECTION_RECEIPT_R2.json"
            receipt.write_text(
                json.dumps(
                    {
                        "kernel": "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r1-runner",
                        "terminal_collection": {"attempt_directory": "attempt_002"},
                    }
                )
            )
            resolved, error = resolve_attempt_dir(
                {"state": "already_collected"}, receipt, artifacts
            )
            self.assertEqual(resolved, attempt.resolve())
            self.assertIsNone(error)
            run_002 = next_inventory_run_dir(resolved)
            self.assertEqual(run_002.name, "provider_inventory_run_002")
            self.assertEqual(prior_page.read_text(), "previous page receipt")

            receipt.write_text(
                json.dumps(
                    {
                        "kernel": "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r1-runner",
                        "terminal_collection": {"attempt_directory": "../outside"},
                    }
                )
            )
            resolved, error = resolve_attempt_dir(
                {"state": "already_collected"}, receipt, artifacts
            )
            self.assertIsNone(resolved)
            self.assertEqual(error, "collector_attempt_directory_unavailable")


if __name__ == "__main__":
    unittest.main()
