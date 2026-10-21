# AllThree terminal checker V2 helper-reference audit

The checker module is `child_allthree_terminal_check_v2.py` (40,442 bytes, SHA-256 `6ff4ba93a4020a4b34ade46a2512ac77b871328e95c47f2ca4561b8dcbd435b5`). It is copied from V1 with exactly one literal replacement in `_check_native_predecessor`: strict schema `fg19.child-nll-exact-owned-stop.v1` becomes the exact frozen receipt schema `fg19-child-nll-exact-owned-stop.v1`. The V1 and V2 byte comparison confirms no other changes.

The seven-file private code closure is the checker plus these six pure helper modules. The exact byte counts and SHA-256 values are in [dependency_manifest.json](dependency_manifest.json).

| Helper module | Referenced leaf functions |
|---|---|
| `child_nll_terminal_check_v1.py` | `_eq`, `_check_ema_predecessor`, `_check_dictionary_child_predecessor` |
| `child_control_terminal_check_v3.py` | Path-compatibility, safe-global, stop-receipt, predecessor, checkpoint, representation, dictionary, normalizer, training-state, scheduler, optimizer, counter, loader/RNG, and contract leaves |
| `child_ema_terminal_check_v3.py` | Registered-data, control-predecessor, parent/dictionary-stat, and trainer-argv leaves |
| `parent_fit_terminal_check_v3.py` | Source identity, worker, cache, membership, cache provenance, and checkpoint leaves |
| `propensity_terminal_support.py` | Bounded stable-file, JSON, and hash helpers |
| `teacher_s5_terminal_check_v1.py` | Canonical launch-argv and tensor-identity leaves |

The public interface remains `assert_helper_closure()` and the same 13-path `verify_terminal(...)` API. The checker adds no `main`, CLI, copied trainer/framework, old entrypoint, or global spoof. The narrow import proof calls only `assert_helper_closure`; it does not call `verify_terminal` or exercise source/run-identity/model paths.
