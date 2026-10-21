# Helper reference audit

The checker depends on six reviewed pure helper files in addition to its own module. Exact byte counts and SHA-256 pins are in `dependency_manifest.json`. All paths below are source-qualified as `<module>.<function>`.

| Helper | Referenced leaves and constants | Use |
| --- | --- | --- |
| `child_nll_terminal_check_v1.py` | `_eq`, `_check_ema_predecessor`, `_check_dictionary_child_predecessor`; pinned predecessor acceptance/stop hashes | Non-coercing equality and two accepted-predecessor joins only. The NLL `verify_terminal` and `main` entry points are not imported or called. |
| `child_control_terminal_check_v3.py` | `_require_path_compatibility_exception`, `_reset_safe_globals`, `_check_stop_receipt`, `_check_predecessor_receipts`, `_read_pinned_json`, `_check_parent_checkpoint`, `_check_representation_payload`, `_check_dictionary_artifact`, `_check_child_normalizer`, `_load_control_training_state`, `_check_scheduler`, `_check_optimizer`, `_check_counters`, `_check_loader_and_rng`, `_check_args_and_contract`; exact receipt, cap, epoch, update, batch, checkpoint, and decoder constants | Reuse reviewed safe state and lineage checks under the exact already-granted Path-decoder authority. The decoder remains state-only and exact-path scoped. |
| `child_ema_terminal_check_v3.py` | `_check_registered_data_stats`, `_check_control_predecessor`, `_check_parent_and_dictionary_stats`, `_trainer_argv_sha` | Registered data identity stats, predecessor joins, dictionary/parent metadata, and the exact trainer-argument digest. |
| `parent_fit_terminal_check_v3.py` | `_source_identity`, `_check_worker`, `_check_cache`, `_check_membership`, `_check_cache_provenance`, `_load_checkpoint` | Source manifest, worker/cache identity, split membership, cache provenance, and capped checkpoint loads. |
| `propensity_terminal_support.py` | `_hash_file`, `_json_file`, `_stable_file` | Stable bounded metadata and artifact reads/hashes. |
| `teacher_s5_terminal_check_v1.py` | `_launch_cli_args`, `_canonical_tensor_identity` | Canonical launch argument decoding and source tensor identity. |

The AllThree checker adds strict native typed-evidence validation before any artifact access. It binds the native raw-capture path to `research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/terminal_native_capture_v1/STDOUT.raw.json`, requires a positive strict-integer byte count and a lowercase 64-character SHA-256, and enforces timezone-qualified capture/exit times. The CPU preparation wrapper independently joins this typed receipt to the captured raw bytes.

`verify_terminal` is the sole full-verification entry point. It does not call any predecessor checker entry point. The WNN input-representation modules are imported only after native, path, launch, admission, receipt, source identity, and cache checks so the accepted sparse artifact can be checked against its source implementation.
