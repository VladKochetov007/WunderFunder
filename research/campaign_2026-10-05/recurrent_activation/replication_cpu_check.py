"""Check actual changed initialization and ordering through the E475 wrapper."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import fit_seed_replication as wrapper
import numpy as np
import torch


def fingerprint(model: torch.nn.Module) -> str:
    """Fingerprint all initialized parameter bytes without any training."""
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    """Exercise wrapper entry with a mocked fit, recording only CPU fixtures."""
    experiment = wrapper.experiment
    original_seed = experiment.SEED
    original_function = experiment.set_seed
    original_argv = sys.argv.copy()
    torch.set_num_threads(1)
    torch.manual_seed(271828)
    initial = fingerprint(experiment.CandidateGRUModel(0.0))
    receipts = []
    try:
        for seed in (271829, 271830):
            torch.manual_seed(seed)
            expected = fingerprint(experiment.CandidateGRUModel(0.0))
            assert expected != initial
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)

                def inspect_entry(
                    case_seed: int = seed, case_expected: str = expected, case_output: Path = output
                ) -> None:
                    assert case_seed == experiment.SEED
                    for alpha in (-0.1, 0.0, 0.1):
                        experiment.set_seed()
                        assert fingerprint(experiment.CandidateGRUModel(alpha)) == case_expected
                    order = list(range(24))
                    np.random.default_rng(experiment.SEED).shuffle(order)
                    expected_order = list(range(24))
                    np.random.default_rng(case_seed).shuffle(expected_order)
                    assert order == expected_order
                    assert sys.argv[1:] == ["--device", "cuda", "--output", str(case_output)]
                    (case_output / "result.json").write_text(
                        json.dumps({"seed": case_seed, "arms": [{}] * 6})
                    )

                sys.argv = [
                    str(Path(wrapper.__file__)),
                    "--seed",
                    str(seed),
                    "--output",
                    str(output),
                ]
                with mock.patch.object(experiment, "main", side_effect=inspect_entry):
                    wrapper.main()
                report = json.loads((output / "result.json").read_text())
                assert (
                    report["replication_wrapper_sha256"]
                    == hashlib.sha256(Path(wrapper.__file__).read_bytes()).hexdigest()
                )
                receipts.append(
                    {
                        "seed": seed,
                        "actual_model_initialization_sha256": expected,
                        "old_seed_initialization_sha256": initial,
                        "bound_default_and_fit_order_checked": True,
                    }
                )
    finally:
        experiment.SEED = original_seed
        experiment.set_seed = original_function
        sys.argv = original_argv
    result = {
        "kind": "CPU wrapper initialization/order fixture; mocked fit, zero scientific updates",
        "checks": receipts,
        "fixture_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = Path(__file__).parent / "REPLICATION_CPU_CHECK.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite CPU fixture receipt")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
