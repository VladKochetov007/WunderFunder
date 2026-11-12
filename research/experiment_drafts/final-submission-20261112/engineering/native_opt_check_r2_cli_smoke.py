from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

CHECKER = Path(__file__).with_name("native_opt_check_r2.py")
EXPECTED = "085dfecd1c9eecf1287606222599f25c9c89edc63d285a3ea2bc82de224032b9"
spec = importlib.util.spec_from_file_location("native_opt_check_r2", CHECKER)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def capture(args):
    assert args.child_label in ("baseline", "candidate")
    assert args.library == Path("lib.so")
    if args.child_label == "candidate":
        assert args.expected_candidate_sha == EXPECTED
    return 0

module.run_child = capture
for label in ("baseline", "candidate"):
    sys.argv = [str(CHECKER), "--child-label", label, "--library", "lib.so",
                "--package-dir", "pkg", "--inputs-dir", "inputs", "--child-out", "out"]
    if label == "candidate":
        sys.argv += ["--expected-candidate-sha", EXPECTED]
    assert module.main() == 0
print("child CLI dispatch passed for baseline/candidate; candidate SHA parsed exactly")
