"""CPU-only checks for the fit-only R4 actor and native official-primary alias."""

from __future__ import annotations

import copy
import unittest

from primary_refit_actor_fit_r4 import _primary_profile_data
from primary_refit_actor_support_r4 import PRIMARY_SHA256


def _native_data() -> dict[str, object]:
    return {
        "split_sha256": PRIMARY_SHA256,
        "primary_split_sha256": None,
        "fit_members": [["train", i] for i in range(10325)],
        "evaluation_members": [["valid", i] for i in range(1873)],
        "evaluation_role": "valid",
        "evaluation_source": "valid",
        "teacher": None,
        "initializer": None,
    }


class NativePrimaryDataTests(unittest.TestCase):
    def test_native_missing_alias_and_exact_alias_are_accepted(self) -> None:
        for alias in (None, PRIMARY_SHA256):
            data = _native_data() | {"primary_split_sha256": alias}
            self.assertTrue(_primary_profile_data(data))

    def test_wrong_identity_membership_and_ancestry_are_rejected(self) -> None:
        original = _native_data()
        for key, value in (
            ("primary_split_sha256", "b" * 64),
            ("split_sha256", "b" * 64),
            ("evaluation_role", "control"),
            ("evaluation_source", "train"),
            ("fit_members", [["train", i] for i in range(8260)]),
            ("evaluation_members", [["train", i] for i in range(1873)]),
            ("teacher", {"checkpoint": "auxiliary"}),
            ("initializer", {"checkpoint": "auxiliary"}),
        ):
            data = copy.deepcopy(original)
            data[key] = value
            with self.subTest(key=key):
                self.assertFalse(_primary_profile_data(data))
        duplicate = _native_data()
        duplicate["fit_members"][-1] = duplicate["fit_members"][0]
        self.assertFalse(_primary_profile_data(duplicate))


if __name__ == "__main__":
    unittest.main()
