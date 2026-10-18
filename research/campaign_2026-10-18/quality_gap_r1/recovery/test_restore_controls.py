"""Exercise archive integrity and credential redirect boundaries offline."""

from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
import urllib.request
from http.client import HTTPMessage
from pathlib import Path

from restore_controls import SafeRedirect, copy_verified, relative


class RestoreTests(unittest.TestCase):
    """Protect create-only publication, exact bytes and credential scope."""

    def test_exact_bytes_publish_and_existing_target_is_not_overwritten(self) -> None:
        """A validated file publishes once; a second write cannot replace it."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.pt"
            digest = hashlib.sha256(b"abc").hexdigest()
            copy_verified(io.BytesIO(b"abc"), target, 3, digest)
            self.assertEqual(target.read_bytes(), b"abc")
            with self.assertRaises(FileExistsError):
                copy_verified(io.BytesIO(b"abc"), target, 3, digest)
            self.assertEqual(target.read_bytes(), b"abc")

    def test_invalid_size_or_hash_never_publishes(self) -> None:
        """Preserve failed partials without exposing them as a model endpoint."""
        for data in (b"ab", b"abcd", b"abd"):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "model.pt"
                with self.assertRaises(ValueError):
                    copy_verified(io.BytesIO(data), target, 3, hashlib.sha256(b"abc").hexdigest())
                self.assertFalse(target.exists())
                self.assertTrue(target.with_name("model.pt.partial").exists())

    def test_relative_paths(self) -> None:
        """Reject paths which could escape the new recovery directory."""
        for value in ("/tmp/model", "../model", "a/../../model", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                relative(value)
        self.assertEqual(str(relative("artifacts/run/model.pt")), "artifacts/run/model.pt")

    def test_redirect_strips_auth_and_rejects_plaintext(self) -> None:
        """An archive redirect must not forward the Hub credential."""
        request = urllib.request.Request(
            "https://huggingface.co/example", headers={"Authorization": "Bearer test-fixture"}
        )
        handler = SafeRedirect()
        redirected = handler.redirect_request(
            request, io.BytesIO(), 302, "Found", HTTPMessage(), "https://cdn.example/model"
        )
        self.assertIsNotNone(redirected)
        self.assertIsNone(redirected.get_header("Authorization"))
        with self.assertRaises(ValueError):
            handler.redirect_request(
                request, io.BytesIO(), 302, "Found", HTTPMessage(), "http://cdn.example/model"
            )


if __name__ == "__main__":
    unittest.main()
