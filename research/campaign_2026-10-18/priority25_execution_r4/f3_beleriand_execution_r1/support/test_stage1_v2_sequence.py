import unittest

from launch_stage1_v2 import sequence_worker_attempt

NATIVE_DEEP = "computer-vision-homework-r01-deep-f3s1001-ep01-v1"
WIDE_CONTINUATION = "computer-vision-homework-r01-wide-f3s1001-ep25-v1"
DEEP_CONTINUATION = "computer-vision-homework-r01-deep-f3s1001-ep25-v1"


def plan_fixture() -> dict[str, object]:
    return {
        "steps": [
            {},
            {
                "step_id": "stage1_deep",
                "kind": "fit_stage1",
                "arm": "deep",
                "attempt": NATIVE_DEEP,
            },
            {
                "step_id": "continuation_wide",
                "kind": "continue_fixed_six_epoch_fit",
                "arm": "wide",
                "attempt": WIDE_CONTINUATION,
            },
            {
                "step_id": "continuation_deep",
                "kind": "continue_fixed_six_epoch_fit",
                "arm": "deep",
                "attempt": DEEP_CONTINUATION,
            },
        ]
    }


class SequenceWorkerAttemptTests(unittest.TestCase):
    def test_finished_deep_stage1_with_wide_continuation_is_valid(self) -> None:
        sequence = {"index": 2, "attempt": WIDE_CONTINUATION}
        self.assertEqual(
            sequence_worker_attempt(plan_fixture(), sequence, native_attempt_id=NATIVE_DEEP),
            WIDE_CONTINUATION,
        )

    def test_finished_deep_stage1_with_deep_continuation_is_valid(self) -> None:
        sequence = {"index": 3, "attempt": DEEP_CONTINUATION}
        self.assertEqual(
            sequence_worker_attempt(plan_fixture(), sequence, native_attempt_id=NATIVE_DEEP),
            DEEP_CONTINUATION,
        )

    def test_wrong_pair_or_index_is_rejected(self) -> None:
        for sequence in (
            {"index": 2, "attempt": DEEP_CONTINUATION},
            {"index": 3, "attempt": WIDE_CONTINUATION},
            {"index": 1, "attempt": NATIVE_DEEP},
            {"index": 4, "attempt": "computer-vision-homework-r01-wide-f3s1001-export-ep5-v1"},
            {"index": True, "attempt": WIDE_CONTINUATION},
        ):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                sequence_worker_attempt(plan_fixture(), sequence, native_attempt_id=NATIVE_DEEP)

    def test_wrong_native_plan_binding_is_rejected(self) -> None:
        plan = plan_fixture()
        plan["steps"][1]["attempt"] = "different-attempt"
        with self.assertRaises(ValueError):
            sequence_worker_attempt(
                plan, {"index": 2, "attempt": WIDE_CONTINUATION}, native_attempt_id=NATIVE_DEEP
            )


if __name__ == "__main__":
    unittest.main()
