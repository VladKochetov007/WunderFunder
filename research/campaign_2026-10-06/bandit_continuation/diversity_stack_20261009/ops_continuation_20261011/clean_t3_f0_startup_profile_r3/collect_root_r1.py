"""Collect the R3 terminal using the reviewed inventory and download primitives."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

ROOT = Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
HERE = Path(__file__).resolve().parent
KERNEL = 'vladkochetov07/wnn-clean-t3-f0-startup-profile-20261011-r3-runner'
ARCHIVE = '986304b0278ea2ea3c5b903984edd5b58fda6e04e497a6bd10f06a58a635e9e1'


def main() -> None:
    """Bind the admitted unique slug, preserve outputs, and require exact closure."""
    adapter_path = HERE.parent / 'clean_t3_f0_profile_collection_r2/collect_profile_r2.py'
    spec = importlib.util.spec_from_file_location('reviewed_r2_adapter', adapter_path)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    source_bindings = adapter.verify_collector_sources()
    admission_path = HERE / 'ROOT_PROFILE_DISPATCH_ADMISSION_R1.json'
    push_path = HERE / 'ROOT_PROFILE_PUSH_R1.json'
    ready_path = HERE / 'ROOT_PRIVATE_DATASET_READY_READBACK_R1.json'
    admission = json.loads(admission_path.read_text())
    push = json.loads(push_path.read_text())
    assert admission['archive_sha256'] == ARCHIVE
    assert push['kernel_id'] == KERNEL and push['exit_code'] == 0
    assert push['admission_sha256'] == adapter.sha256(admission_path)
    assert admission['ready_readback_sha256'] == adapter.sha256(ready_path)
    assert admission['fit_admitted'] is False
    # Runtime parameters for a new identified collection; original adapter bytes stay fixed.
    adapter.KERNEL = KERNEL
    adapter.KERNEL_OUTPUT_LOG_NAME = KERNEL.split('/')[-1] + '.log'
    snapshot = adapter.poll_provider.collect(HERE / 'ROOT_SINGLE_TARGET_R4.json')
    status = adapter.terminal._target_status(snapshot, KERNEL)
    assert status and adapter.terminal.terminal_state(status['status'])
    collection = HERE / 'terminal_collection_root_r1'
    collection.mkdir(mode=0o700)
    inventory_dir = collection / 'inventory'
    inventory_dir.mkdir(mode=0o700)
    inventory = adapter.collect_session_output_inventory(
        kernel=KERNEL, inventory_dir=inventory_dir, deadline=time.monotonic() + 120
    )
    adapter.write_immutable_json(collection / 'PROVIDER_INVENTORY_R1.json', inventory)
    assert inventory['complete'] and not adapter.inventory_log_collisions(inventory)
    target = {'kernel': KERNEL, 'resource': 'gpu', 'artifact_dir': 'attempts',
              'terminal_receipt': 'TERMINAL_COLLECTION_RECEIPT_R1.json'}
    config = {'minimum_free_bytes': 4 * 1024**3, 'maximum_download_bytes': 8 * 1024**3,
              'max_runtime_seconds': 420, 'max_attempts': 3,
              'kaggle_cli': str(ROOT / '.venv/bin/kaggle'),
              'expected_cli_version': 'Kaggle CLI 2.2.4'}
    result = adapter.terminal.collect_terminal_target(
        target=target, status_row=status, config=config, repo_root=ROOT, config_dir=collection
    )
    receipt_path = collection / target['terminal_receipt']
    attempt, error = adapter.resolve_attempt_dir(result, receipt_path, collection / 'attempts')
    assert attempt and not error
    comparison = adapter.compare_provider_inventory(
        inventory['files'], attempt, provider_response_log=inventory.get('response_log')
    )
    low = adapter.terminal.collection_is_complete(receipt_path, collection / 'attempts')
    complete = adapter.terminal_evidence_complete(low, inventory, comparison)
    handoff = {'kernel': KERNEL, 'pushed_version': 1, 'archive_sha256': ARCHIVE,
               'provider_endpoint_version_pinned': False, 'quota_snapshot': snapshot,
               'root_admission_sha256': adapter.sha256(admission_path),
               'push_sha256': adapter.sha256(push_path), 'source_bindings': source_bindings,
               'operator_sha256': adapter.sha256(Path(__file__)),
               'collector_result': result, 'comparison': comparison,
               'collection_complete': complete, 'profile_success': 'requires interpretation',
               'attempt': str(attempt), 'terminal_receipt_sha256': adapter.sha256(receipt_path)}
    adapter.write_immutable_json(collection / 'ROOT_TERMINAL_HANDOFF_R1.json', handoff)
    print(json.dumps({'collection_complete': complete, 'attempt': str(attempt),
                      'files': len(inventory['files']), 'provider_status': status['status']}))
    if not complete:
        raise RuntimeError('terminal inventory/log/download closure failed')


if __name__ == '__main__':
    main()
