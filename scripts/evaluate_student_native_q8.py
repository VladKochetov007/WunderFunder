"""Use the existing full-fold native evaluator with an asserted Q8 implementation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import evaluate_qat_deployment as evaluation

EVALUATOR_SHA256 = "a8bb0f15d49f0a6a36ea9f37995ea5a1906814e56bed566a829fb4ff396a80ce"


def main() -> None:
    """Preserve the frozen scorer and require the selected native VNNI policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    evaluator_path = Path(evaluation.__file__)
    if hashlib.sha256(evaluator_path.read_bytes()).hexdigest() != EVALUATOR_SHA256:
        raise ValueError("native evaluator differs from the frozen reuse contract")
    original_factory = evaluation.load_case

    def asserted_factory(native: Any, directory: Path, case: str) -> Any:
        model = original_factory(native, directory, case)
        if case != "native_q8" or model.precision != "q8vnni":
            raise RuntimeError("native grade did not select the required Q8 VNNI implementation")
        if model.active_members != 1:
            raise RuntimeError("native student grade requires exactly one active member")
        return model

    evaluation.load_case = asserted_factory
    evaluation.run(argparse.Namespace(
        output=args.output.resolve(strict=True),
        manifest_sha256=args.manifest_sha256,
        profile=args.profile,
        case="native_q8",
    ))


if __name__ == "__main__":
    main()
