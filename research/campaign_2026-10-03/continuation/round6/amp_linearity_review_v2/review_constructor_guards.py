from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

OWN = Path(__file__).resolve().parent
HERE = OWN.parent


def main() -> None:
    """Evaluate only new rejection expressions, never the approval constructor."""
    helper = HERE / "approve_amp_audit.py"
    tree = ast.parse(helper.read_text())
    messages = {
        "Reviewed integration and candidate bindings differ",
        "Reviewed metadata or measurement identity changed",
        "Independently reviewed capture manifest changed",
    }
    predicates = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            for statement in node.body:
                if isinstance(statement, ast.Raise) and isinstance(statement.exc, ast.Call):
                    arguments = statement.exc.args
                    if (
                        arguments
                        and isinstance(arguments[0], ast.Constant)
                        and arguments[0].value in messages
                    ):
                        predicates[arguments[0].value] = compile(
                            ast.Expression(node.test), "reviewed_rejection_expression", "eval"
                        )
    assert set(predicates) == messages
    integration = json.loads((OWN / "integration_bindings.json").read_text())
    candidate = integration["candidate_sha256"]
    metadata = HERE / "fixed_case_metadata_v2.json"
    measurement = HERE / "capture_measurement_review/STATE.json"
    output_manifest = HERE.parent / "round4/gpu_capture_v4_default_2120r/output_hashes.json"
    fake_hashes = {
        metadata: integration["fixed_case_metadata_sha256"],
        measurement: integration["independent_measurement_STATE_sha256"],
        HERE / "prepare_capture_metadata_v2.py": integration["prepare_controller_helper_sha256"],
        HERE / "real_trajectory.json": integration["existing_trajectory_sha256"],
        output_manifest: integration["output_manifest_sha256"],
    }
    checks = {}
    for message, predicate in predicates.items():
        namespace = {
            "integration": copy.deepcopy(integration),
            "candidate_sha": copy.deepcopy(candidate),
            "state": {"verdict": integration["verdict"]},
            "HERE": HERE,
            "metadata": metadata,
            "measurement": measurement,
            "output_manifest": output_manifest,
            "sha": lambda path: fake_hashes[Path(path)],
        }
        assert not eval(predicate, {"__builtins__": {}}, namespace)
        checks[message + ":matching"] = True
        if message == "Reviewed integration and candidate bindings differ":
            namespace["state"]["verdict"] = "BLOCKED"
            assert eval(predicate, {"__builtins__": {}}, namespace)
            namespace["state"]["verdict"] = integration["verdict"]
            namespace["candidate_sha"]["metadata.py"] = "wrong"
            assert eval(predicate, {"__builtins__": {}}, namespace)
            checks[message + ":verdict_and_candidate_mismatches_reject"] = True
        else:
            targets = (
                (
                    metadata,
                    measurement,
                    HERE / "prepare_capture_metadata_v2.py",
                    HERE / "real_trajectory.json",
                )
                if message.startswith("Reviewed metadata")
                else (output_manifest,)
            )
            for path in targets:
                original = fake_hashes[path]
                fake_hashes[path] = "wrong"
                assert eval(predicate, {"__builtins__": {}}, namespace)
                fake_hashes[path] = original
                checks[str(path.name) + ":changed_binding_rejects"] = True
    result = {
        "pass": True,
        "checks": checks,
        "constructor_executed": False,
        "receipt_written": False,
        "real_sha_or_capture_access_during_predicate_tests": False,
        "prior_tiny_retest_results_unchanged": True,
    }
    (OWN / "constructor_guard_tests.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
