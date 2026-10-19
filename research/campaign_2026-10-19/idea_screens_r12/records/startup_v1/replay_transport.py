from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path('/Users/nb/Documents/wunder/WunderFunder-maintenance')
CAMPAIGN = REPO / 'research/campaign_2026-10-19'
R10_PACKET = CAMPAIGN / 'idea_screens_r10/records/startup_v1'
R11_PACKET = CAMPAIGN / 'idea_screens_r11/records/startup_v1'
R12 = CAMPAIGN / 'idea_screens_r12'
PACKET = R12 / 'records/startup_v1'
PRIVATE = Path('/Users/nb/Documents/wunder/runtime-20261018/eda19-r12-v1-transport-private')
PYTHON = '/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python'
PINS = {
    'assembler': 'c235bc3d8a55f04b654f3adbfecdab224b069bc8d39682c17795aa82c2eabfe0',
    'inline': '3f0bd50205dc2d15673d92f1b811b9da83f6afc5710e63a417b03112d7fe8199',
    'manifest': '20d26c8adfddef62feb76edebba254751bd2f7c3cbecddfd2487606c7690ec86',
    'review': '706ac94949b722a6bbc3813700242ab62f1dd8f2a85efa026404ec89703b1b18',
    'tests': 'dc7623a70722f1a19d9c8bbfe5695e9afac19ea8982e9f36e5596b0edbc69b60',
    'smoke': 'c9a8d9e5893c62e1d67e779cbdae84c4dd243ce07a28b8d30ca499a22250df5b',
}

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def replace_exact(source: str, old: str, new: str, count: int, label: str, log: list[dict[str, Any]]) -> str:
    found = source.count(old)
    if found != count:
        raise RuntimeError(f'{label}: expected {count} occurrences, found {found}')
    log.append({'id': label, 'occurrences': found, 'source_sha256': sha(old.encode()), 'replacement_sha256': sha(new.encode())})
    return source.replace(old, new)

def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)

def main() -> None:
    manifest_bytes = (R12 / 'PREPARATION_MANIFEST.json').read_bytes()
    review_bytes = (R12 / 'IMPLEMENTATION_REVIEW.json').read_bytes()
    tests_bytes = (R12 / 'TEST_RECEIPT.json').read_bytes()
    readme_bytes = (R12 / 'README.md').read_bytes()
    smoke_bytes = (PACKET / 'LOCAL_IMPORT_SMOKE.json').read_bytes()
    for key, data in [('manifest', manifest_bytes), ('review', review_bytes), ('tests', tests_bytes), ('smoke', smoke_bytes)]:
        if sha(data) != PINS[key]:
            raise RuntimeError(f'R12 {key} bytes changed from the reviewed source freeze')
    review = json.loads(review_bytes)
    smoke = json.loads(smoke_bytes)
    if not str(review.get('status', '')).startswith('PASS_SOURCE_AND_SYNTHETIC_REVIEW'):
        raise RuntimeError('R12 source review is not PASS')
    if review.get('bindings', {}).get('preparation_manifest_sha256') != PINS['manifest'] or smoke.get('status') != 'PASS':
        raise RuntimeError('R12 review/smoke does not bind the frozen manifest or pass')

    manifest = json.loads(manifest_bytes)
    base_spec_path = R11_PACKET / 'ASSEMBLY_SPEC_ATTEMPT2.json'
    base_spec_bytes = base_spec_path.read_bytes()
    base_spec = json.loads(base_spec_bytes)
    assembler_path = R11_PACKET / 'assemble_transport_attempt2.py'
    assembler_bytes = assembler_path.read_bytes()
    inline_path = R11_PACKET / 'root_inline_attempt2.py'
    inline_bytes = inline_path.read_bytes()
    if sha(assembler_bytes) != PINS['assembler'] or sha(inline_bytes) != PINS['inline']:
        raise RuntimeError('pinned R11 assembler or inline template changed')
    for name, expected in base_spec['r10_base_pins'].items():
        if sha((R10_PACKET / name).read_bytes()) != expected:
            raise RuntimeError(f'pinned R10 helper changed: {name}')

    file_pins = manifest['files']
    if manifest.get('schema') != 'nf19.r12_preparation.v1' or len(file_pins) != 32 or len(manifest['runtime_py']) != 26:
        raise RuntimeError('R12 manifest closure differs from the reviewed 32-pin/26-module shape')
    spec = {
        'schema': 'nf19.r12_startup_assembly_spec.v1',
        'unit': 'computer-vision-homework-eda19-r12-v1.service',
        'destination': 'support-source/eda19-r12-v1',
        'output': 'research_tmp/nf19-r12-v1',
        'sample': base_spec['sample'],
        'sample_sha256': base_spec['sample_sha256'],
        'membership_sha256': base_spec['membership_sha256'],
        'source_snapshot': base_spec['source_snapshot'],
        'allowed_bundle_prefixes': ['idea_screens_r1', 'idea_screens_r2', 'idea_screens_r3', 'idea_screens_r7', 'idea_screens_r12'],
        'r7_source_allowlist': base_spec['r7_source_allowlist'],
        'extra_transport_paths': ['idea_screens_r12/PREPARATION_MANIFEST.json', 'idea_screens_r12/IMPLEMENTATION_REVIEW.json', 'idea_screens_r12/TEST_RECEIPT.json', 'idea_screens_r12/README.md'],
        'canonical_pin_names': sorted(name for name in file_pins if name.startswith('canonical:')),
        'runtime_py': manifest['runtime_py'],
        'limits': base_spec['limits'],
        'r10_base_pins': base_spec['r10_base_pins'],
        'identity_patch': {'remote_body_r10_v1_to_r12_v1': 3, 'capture_r10_v1_to_r12_v1': 3, 'stop_r10_v1_to_r12_v1': 2},
        'patches': [
            {'id': 'identity-paths', 'change': 'Bind the R12-v1 unit/source/output; retain the accepted R3 sample and fixed caps.'},
            {'id': 'runtime-map', 'change': 'Bind exact equality with the 26-module R12 runtime map.'},
            {'id': 'canonical-pins', 'change': 'Verify all six canonical WNN pins before imports and again before launch.'},
            {'id': 'prefixes', 'change': 'Allow only R1/R2/R3/R7/R12 source roots and the exact four-file R7 allowlist.'},
            {'id': 'cold-import', 'change': 'Use the R12-bound read-sentinel import gate before fresh CLI execv.'},
            {'id': 'capture-stop', 'change': 'Bind exact R12-v1 identity into pinned V2 capture and exact-stop helpers.'},
        ],
    }
    spec_bytes = (json.dumps(spec, indent=2, sort_keys=True) + '\n').encode()

    inline = inline_bytes.decode()
    inline_log: list[dict[str, Any]] = []
    for old, new, count, label in [
        ('idea_screens_r11', 'idea_screens_r12', 3, 'namespace-r11-to-r12'),
        ('r11-v2', 'r12-v1', 2, 'identity-r11-v2-to-r12-v1'),
        ('R11', 'R12', 14, 'labels-R11-to-R12'),
        ('r11', 'r12', 2, 'schema-r11-to-r12'),
        ('nf19.r12_import_smoke.attempt2.v1', 'nf19.r12_import_smoke.v1', 1, 'smoke-schema-v1'),
    ]:
        inline = replace_exact(inline, old, new, count, label, inline_log)
    if 'idea_screens_r11' in inline or 'r11-v2' in inline:
        raise RuntimeError('R12 inline source retains superseded identity')
    compile(inline, 'root_inline_r12.py', 'exec')

    assembler = assembler_bytes.decode()
    assembler_log: list[dict[str, Any]] = []
    for old, new, count, label in [
        ('idea_screens_r11', 'idea_screens_r12', 25, 'namespace-r11-to-r12'),
        ('r11-v2', 'r12-v1', 3, 'identity-r11-v2-to-r12-v1'),
        ('R11', 'R12', 29, 'labels-and-root-R11-to-R12'),
        ('r11', 'r12', 14, 'schema-and-output-r11-to-r12'),
        ('/eda19-r12-attempt2', '/eda19-r12-v1-transport-private', 1, 'private-directory'),
        ('ASSEMBLY_SPEC_ATTEMPT2.json', 'ASSEMBLY_SPEC.json', 1, 'spec-path'),
        ('ASSEMBLY_RECEIPT_ATTEMPT2.json', 'ASSEMBLY_RECEIPT.json', 1, 'receipt-path'),
        ('root_inline_attempt2.py', 'root_inline_r12.py', 2, 'inline-template-path'),
        ('r10-v2', 'r12-v1', 3, 'direct-R10-to-R12-identity'),
        ('three-file set', 'six-file set', 2, 'canonical-count-label'),
        ('nf19.r12_transport_assembly.attempt2.v1', 'nf19.r12_transport_assembly.v1', 1, 'assembly-receipt-schema'),
    ]:
        assembler = replace_exact(assembler, old, new, count, label, assembler_log)
    if any(token in assembler for token in ['idea_screens_r11', 'r11-v2', 'r10-v2']):
        raise RuntimeError('R12 assembler source retains a superseded identity')
    compile(assembler, 'assemble_transport_r12.py', 'exec')

    packet_readme = (
        '# R12 transport preparation\n\n'
        'This record replays the pinned R11 attempt-2 assembler and R10 staging/capture/stop helpers with occurrence-counted R12 namespace and `r12-v1` identity substitutions. The large remote body is not copied into this repository; generated scripts remain in the private local runtime directory.\n\n'
        'The scientific README is transported byte-for-byte. `LOCAL_IMPORT_SMOKE.json` records the cold 26-module import and zero-read sentinel check. `ASSEMBLY_SPEC.json` binds the fixed R3 sample, 32 manifest pins, 26 runtime modules, six canonical pins, four-file R7 allowlist, and unchanged resource limits. `REPLAY.json` records source hashes, substitutions, and generated script identities. This replay performs no remote action.\n\n'
        'Unit: `computer-vision-homework-eda19-r12-v1.service`; source: `support-source/eda19-r12-v1`; output: `research_tmp/nf19-r12-v1`; caps: 0.5 CPU, 1.4 GB high, 1.5 GB max, zero swap, 600 seconds.\n'
    ).encode()
    write_new(PACKET / 'ASSEMBLY_SPEC.json', spec_bytes)
    write_new(PACKET / 'root_inline_r12.py', inline.encode())
    write_new(PACKET / 'README.md', packet_readme)
    PRIVATE.mkdir(parents=True, exist_ok=False)
    private_assembler = PRIVATE / 'assemble_transport_r12.py'
    write_new(private_assembler, assembler.encode())
    result = subprocess.run([PYTHON, str(private_assembler)], cwd=REPO, text=True, capture_output=True, timeout=120, check=False)
    if result.returncode:
        raise RuntimeError('local assembly failed before remote action: ' + result.stderr[-4000:])
    assembly_receipt_path = PACKET / 'ASSEMBLY_RECEIPT.json'
    assembly_receipt_bytes = assembly_receipt_path.read_bytes()
    assembly_receipt = json.loads(assembly_receipt_bytes)
    generated = assembly_receipt['generated_scripts']
    for name, entry in generated.items():
        path = PRIVATE / name
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha(path.read_bytes()) != entry['sha256']:
            raise RuntimeError(f'private generated script failed receipt verification: {name}')

    replay = {
        'schema': 'nf19.r12_transport_replay.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'remote_actions_performed': False,
        'source_bindings': {
            'r11_attempt2_assembler_sha256': sha(assembler_bytes),
            'r11_attempt2_inline_sha256': sha(inline_bytes),
            'r11_attempt2_spec_sha256': sha(base_spec_bytes),
            'r10_helper_pins': base_spec['r10_base_pins'],
            'r12_manifest_sha256': PINS['manifest'],
            'r12_review_sha256': PINS['review'],
            'r12_test_receipt_sha256': PINS['tests'],
            'r12_readme_sha256': sha(readme_bytes),
            'local_import_smoke_sha256': PINS['smoke'],
        },
        'identity': {'unit': spec['unit'], 'source': spec['destination'], 'output': spec['output'], 'sample_sha256': spec['sample_sha256']},
        'spec_path': 'research/campaign_2026-10-19/idea_screens_r12/records/startup_v1/ASSEMBLY_SPEC.json',
        'spec_sha256': sha(spec_bytes),
        'inline_template_path': 'research/campaign_2026-10-19/idea_screens_r12/records/startup_v1/root_inline_r12.py',
        'inline_template_sha256': sha(inline.encode()),
        'private_assembler': {'path': str(private_assembler), 'bytes': private_assembler.stat().st_size, 'sha256': sha(private_assembler.read_bytes())},
        'assembler_identity_patches': assembler_log,
        'inline_identity_patches': inline_log,
        'assembly_receipt_path': 'research/campaign_2026-10-19/idea_screens_r12/records/startup_v1/ASSEMBLY_RECEIPT.json',
        'assembly_receipt_sha256': sha(assembly_receipt_bytes),
        'generated_scripts_private': {name: {**entry, 'path': str(PRIVATE / name)} for name, entry in generated.items()},
        'assembly_literal_patch_log': assembly_receipt['literal_patch_log'],
    }
    replay_bytes = (json.dumps(replay, indent=2, sort_keys=True) + '\n').encode()
    write_new(PACKET / 'REPLAY.json', replay_bytes)
    print(json.dumps({'status': 'PASS_LOCAL_ASSEMBLY_ONLY', 'spec_sha256': sha(spec_bytes), 'assembly_receipt_sha256': sha(assembly_receipt_bytes), 'replay_sha256': sha(replay_bytes), 'private_directory': str(PRIVATE), 'generated_scripts_private': replay['generated_scripts_private'], 'remote_actions_performed': False}, indent=2))

if __name__ == '__main__':
    main()
