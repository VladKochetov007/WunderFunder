from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN = 'research/campaign_2026-10-04/kaggle/E430_cadence_dualT4_v1'
RECIPES = Path('research/campaign_2026-10-04/E430_optimizer_cadence/specs')
BASE_MANIFEST = Path('research/campaign_2026-10-04/kaggle/E419_qat_dualT4_v2/bundle/bundle-manifest.json')
BASE_SPEC = Path('research/campaign_2026-10-04/kaggle/E419_qat_dualT4_v2/specs/c20261004_e419_qat_recurrent_s3_f0_t4v2.sealed.json')
DATASET = 'vladkochetov07/wnn-e430-cadence-dualt4-20261004-v1'
RUNNER = 'vladkochetov07/wnn-e430-cadence-dualt4-20261004-v1-runner'
FOLD0_ASSET_DATASET = 'vladkochetov07/wnn-c20261004-fold0-assets'


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or not value:
        raise ValueError(f'Unsafe bundle path: {value!r}')
    return str(path)


def descriptor(path: str) -> dict:
    local = ROOT / relative(path)
    if not local.is_file():
        raise FileNotFoundError(local)
    return {'path': path, 'size': local.stat().st_size, 'sha256': sha256_file(local)}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main() -> None:
    campaign_root = ROOT / CAMPAIGN
    external_manifest = json.loads((ROOT / BASE_MANIFEST).read_text())
    base_spec = json.loads((ROOT / BASE_SPEC).read_text())
    base_inputs = {item['path']: item for item in base_spec['inputs']}
    external = external_manifest['external_mounts']
    dataset_sources = {
        DATASET,
        FOLD0_ASSET_DATASET,
        'vladkochetov07/wunder-funder',
    }
    remote_names = {
        'chunk1000': 'c20261004_e430_chunk1000_s3_f0_t4v1',
        'chunk2000_connected': 'c20261004_e430_chunk2000_connected_s3_f0_t4v1',
        'accum2000_trunc1000': 'c20261004_e430_accum2000_trunc1000_s3_f0_t4v1',
    }
    full_names = {arm: name + '_fit' for arm, name in remote_names.items()}
    probe_names = {arm: name + '_probe' for arm, name in remote_names.items()}
    report_paths: dict[str, str] = {}
    plan_paths: dict[str, str] = {}
    spec_names: list[str] = []
    all_code_paths: set[str] = set()
    spec_data = {}

    for draft_path in sorted((ROOT / RECIPES).glob('*.draft.json')):
        draft = json.loads(draft_path.read_text())
        stem = draft_path.name.removesuffix('.draft.json')
        if stem.startswith('e430_chunk1000_'):
            arm = 'chunk1000'
        elif stem.startswith('e430_chunk2000_connected_'):
            arm = 'chunk2000_connected'
        elif stem.startswith('e430_accum2000_trunc1000_'):
            arm = 'accum2000_trunc1000'
        else:
            raise ValueError(f'Unexpected E430 recipe: {draft_path}')
        if arm in spec_data:
            raise ValueError(f'Duplicate E430 arm: {arm}')
        argv = list(draft['argv'])
        name_index = argv.index('--name') + 1
        argv[name_index] = full_names[arm]
        expected_equal = {
            '--train-source': 'train+valid', '--hidden': '256', '--layers': '2',
            '--batch': '32', '--epochs': '4', '--seed': '3', '--checkpoint-policy': 'fixed',
            '--checkpoint-epoch': '3', '--no-eval-during-fit': None,
        }
        for option, expected in expected_equal.items():
            if option not in argv:
                raise ValueError(f'Recipe missing required setting {option}: {arm}')
            if expected is not None and argv[argv.index(option) + 1] != expected:
                raise ValueError(f'Recipe changed fixed setting {option}: {arm}')
        if argv[argv.index('--split-manifest') + 1] != 'research/campaign_2026-09-29/artifacts/splits/fold0.json':
            raise ValueError('E430 must use fold0 split manifest')
        if '--max-batches' in argv or '--update-rows' in argv and arm == 'chunk1000':
            raise ValueError(f'Unexpected topology flags for {arm}')
        if arm == 'accum2000_trunc1000' and argv[argv.index('--update-rows') + 1] != '2000':
            raise ValueError('Accumulation macro update must remain 2,000 rows')
        if arm == 'chunk2000_connected' and argv[argv.index('--chunk') + 1] != '2000':
            raise ValueError('Connected arm must remain a connected 2,000-row chunk')
        full_artifacts = [
            f'artifacts/runs/{full_names[arm]}/model_ep3.pt',
            f'artifacts/runs/{full_names[arm]}/model.pt',
            f'artifacts/runs/{full_names[arm]}/provenance.json',
            f'artifacts/runs/{full_names[arm]}/history.json',
            f'artifacts/runs/{full_names[arm]}/training_state.pt',
        ]
        fit_report = f'{CAMPAIGN}/reports/{full_names[arm]}_steps.json'
        fit_plan = f'{CAMPAIGN}/plans/{full_names[arm]}_plan.json'
        probe_argv = list(argv)
        probe_argv[probe_argv.index('--name') + 1] = probe_names[arm]
        probe_argv.extend(['--max-batches', '1'])
        probe_artifacts = [
            f'artifacts/runs/{probe_names[arm]}/model_ep3.pt',
            f'artifacts/runs/{probe_names[arm]}/model.pt',
            f'artifacts/runs/{probe_names[arm]}/provenance.json',
            f'artifacts/runs/{probe_names[arm]}/history.json',
            f'artifacts/runs/{probe_names[arm]}/training_state.pt',
        ]
        probe_report = f'{CAMPAIGN}/reports/{probe_names[arm]}_steps.json'
        probe_plan = f'{CAMPAIGN}/plans/{probe_names[arm]}_plan.json'
        for plan_path, report_path, step_name, step_argv in (
            (fit_plan, fit_report, 'four_epoch_full_recipe_fit', argv),
            (probe_plan, probe_report, 'one_batch_per_epoch_production_shape_probe', probe_argv),
        ):
            plan = {'version': 1, 'steps': [{'name': step_name, 'argv': ['{python}', *step_argv[1:]]}], 'report': report_path}
            write_json(ROOT / plan_path, plan)
            plan_paths[f'{arm}:{step_name}'] = plan_path
            report_paths[f'{arm}:{step_name}'] = report_path
        spec_data[arm] = {
            'draft': draft,
            'fit_argv': ['{python}', 'scripts/train_rnn.py', *argv[1:]],
            'probe_argv': ['{python}', 'scripts/train_rnn.py', *probe_argv[1:]],
            'full_artifacts': full_artifacts,
            'probe_artifacts': probe_artifacts,
            'fit_report': fit_report,
            'probe_report': probe_report,
            'fit_plan': fit_plan,
            'probe_plan': probe_plan,
        }
        all_code_paths.update(draft['code'])

    if set(spec_data) != {'chunk1000', 'chunk2000_connected', 'accum2000_trunc1000'}:
        raise ValueError('Must package exactly the three preregistered cadence arms')
    all_code_paths.update({'scripts/kaggle_common.py', 'scripts/kaggle_gpu_runner.py', 'scripts/kaggle_sequence.py', f'{CAMPAIGN}/phase_runner.py'})
    all_code_paths.update(plan_paths.values())
    all_inputs = sorted(set(sum((list(item['draft']['inputs']) for item in spec_data.values()), [])))
    if any(path not in base_inputs for path in all_inputs):
        raise ValueError(f'Fold0 input descriptor unavailable in E419 map: {sorted(set(all_inputs) - set(base_inputs))}')
    if any(path not in external for path in all_inputs):
        raise ValueError(f'No fold0 external mount for inputs: {sorted(set(all_inputs) - set(external))}')
    external_mounts = {}
    for path in all_inputs:
        mapping = external[path]
        if isinstance(mapping, str):
            # Raw parquet comes from the common data dependency attached by E419.
            external_mounts[path] = {'dataset': 'vladkochetov07/wunder-funder', 'filename': mapping}
        else:
            if mapping.get('dataset') != FOLD0_ASSET_DATASET:
                raise ValueError(f'Non-fold0 asset map for {path}')
            external_mounts[path] = mapping
        dataset_sources.add(external_mounts[path]['dataset'])
    input_descriptors = [base_inputs[path] for path in all_inputs]

    for arm, data in spec_data.items():
        draft = data['draft']
        source_code_paths = sorted(set(draft['code']) | {
            'scripts/kaggle_common.py', 'scripts/kaggle_gpu_runner.py',
            'scripts/kaggle_sequence.py', f'{CAMPAIGN}/phase_runner.py',
            data['fit_plan'], data['probe_plan'],
        })
        code = [descriptor(path) for path in source_code_paths]
        code_sha = canonical_sha256(sorted(code, key=lambda item: item['path']))
        for phase, runner_name, plan_path, report_path, argv, artifacts, timeout in (
            ('probe', probe_names[arm], data['probe_plan'], data['probe_report'],
             ['{python}', 'scripts/kaggle_sequence.py', '--plan', data['probe_plan']],
             data['probe_artifacts'] + [data['probe_report']], 1200),
            ('fit', full_names[arm], data['fit_plan'], data['fit_report'],
             ['{python}', 'scripts/kaggle_sequence.py', '--plan', data['fit_plan']],
             data['full_artifacts'] + [data['fit_report']], 12600),
        ):
            spec = {
                'version': 1,
                'name': runner_name,
                'kind': 'gpu',
                'argv': argv,
                'code': code,
                'code_sha256': code_sha,
                'inputs': input_descriptors,
                'artifacts': artifacts,
                'limits': {
                    'cpus': 8,
                    'ram_bytes': 10 * 1024**3,
                    'timeout_seconds': timeout,
                    'gpu_memory_bytes': 11 * 1024**3,
                    'gpu_stop_bytes': 13 * 1024**3,
                },
                'environment': {
                    'WNN_LOADER_WORKERS': '2',
                    'WNN_LOADER_RAM_GB': '3',
                    'WNN_PORTABLE_DATA_MANIFEST': 'research/campaign_2026-10-04/portable_inputs.json',
                },
            }
            spec_path = f'kaggle_specs/{runner_name}.json'
            write_json(ROOT / CAMPAIGN / spec_path, spec)
            spec_names.append(spec_path)
    spec_names.sort()

    # Copy exact code and plan bytes into a compact deterministic source archive. Inputs stay as
    # read-only mounts to the existing fold-0 asset datasets and are never copied into this archive.
    code_files = [descriptor(path) for path in sorted(all_code_paths)]
    archive_members = code_files
    archive_path = ROOT / CAMPAIGN / 'bundle/bundle.wnn'
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as zipped:
            with tarfile.open(fileobj=zipped, mode='w') as tar:
                for item in archive_members:
                    path = item['path']
                    local = ROOT / path
                    info = tarfile.TarInfo(path)
                    info.size = item['size']
                    info.mode = 0o644
                    info.mtime = 0
                    with local.open('rb') as source:
                        tar.addfile(info, source)
                for spec_path in spec_names:
                    local = ROOT / CAMPAIGN / spec_path
                    name = f'{CAMPAIGN}/{spec_path}'
                    data = local.read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(data), 0o644, 0
                    import io
                    tar.addfile(info, io.BytesIO(data))
    archive_sha = sha256_file(archive_path)
    archive_size = archive_path.stat().st_size
    expanded = sum(item['size'] for item in code_files) + sum((ROOT / CAMPAIGN / p).stat().st_size for p in spec_names)
    manifest = {
        'version': 1,
        'dataset': DATASET,
        'private': True,
        'preparation_status': 'source-only draft; no upload or run admitted',
        'archive_bytes': archive_size,
        'archive_sha256': archive_sha,
        'expanded_bytes': expanded,
        'files': code_files,
        'spec_paths': [f'{CAMPAIGN}/{path}' for path in spec_names],
        'external_mounts': external_mounts,
    }
    write_json(ROOT / CAMPAIGN / 'bundle/bundle-manifest.json', manifest)
    write_json(ROOT / CAMPAIGN / 'bundle/dataset-metadata.json', {
        'id': DATASET, 'isPrivate': True, 'licenses': [{'name': 'other'}],
        'title': DATASET,
    })
    write_json(ROOT / CAMPAIGN / 'bundle/kernel/kernel-metadata.json', {
        'code_file': 'kernel.py', 'competition_sources': [],
        'dataset_sources': sorted(dataset_sources), 'enable_gpu': True,
        'enable_internet': False, 'enable_tpu': False, 'id': RUNNER,
        'is_private': True, 'kernel_sources': [], 'kernel_type': 'script',
        'language': 'python', 'machine_shape': 'NvidiaTeslaT4',
        'title': f'{RUNNER} runner',
    })
    kernel = '''from __future__ import annotations\nimport hashlib, json, os, shutil, subprocess, sys, tarfile\nfrom pathlib import Path, PurePosixPath\nBUNDLE_SLUG = "wnn-e430-cadence-dualt4-20261004-v1"\nEXPECTED_ARCHIVE_SHA256 = "'''+archive_sha+'''"\nEXPECTED_EXTERNAL_MOUNTS = '''+repr(external_mounts)+'''\ndef main():\n    matches = list(Path("/kaggle/input").glob(f"**/{BUNDLE_SLUG}/bundle-manifest.json"))\n    if len(matches) != 1: raise ValueError("Expected exactly one private E430 bundle")\n    manifest = json.loads(matches[0].read_text())\n    if manifest.get("private") is not True or manifest.get("external_mounts") != EXPECTED_EXTERNAL_MOUNTS: raise ValueError("Bundle privacy or fold-0 map mismatch")\n    archive = matches[0].parent / "bundle.wnn"\n    digest = hashlib.sha256()\n    with archive.open("rb") as stream:\n        for block in iter(lambda: stream.read(8 * 1024**2), b""): digest.update(block)\n    if digest.hexdigest() != EXPECTED_ARCHIVE_SHA256 or digest.hexdigest() != manifest["archive_sha256"]: raise ValueError("Source archive identity mismatch")\n    root = Path("/kaggle/working/experiment")\n    if shutil.disk_usage(root.parent).free < manifest["expanded_bytes"] + 2 * 1024**3: raise ValueError("Insufficient extraction disk reserve")\n    root.mkdir(parents=True, exist_ok=False)\n    allowed = {item["path"] for item in manifest["files"]} | set(manifest["spec_paths"]) | set(manifest["external_mounts"])\n    seen = set()\n    with tarfile.open(archive, "r:gz") as tar:\n        for member in tar:\n            path = PurePosixPath(member.name)\n            if path.is_absolute() or ".." in path.parts or "\\\\" in member.name or not member.isfile() or member.name not in allowed or member.name in seen: raise ValueError("Unsafe or undeclared archive member")\n            seen.add(member.name); target = root / member.name; target.parent.mkdir(parents=True, exist_ok=True)\n            source = tar.extractfile(member)\n            if source is None: raise ValueError("Missing archive member bytes")\n            with source, target.open("wb") as out: shutil.copyfileobj(source, out, 8 * 1024**2)\n    for logical, mapping in manifest["external_mounts"].items():\n        path = PurePosixPath(logical)\n        if path.is_absolute() or ".." in path.parts or logical not in allowed: raise ValueError("Unsafe external path")\n        slug = mapping["dataset"].split("/")[1]; filename = mapping["filename"]\n        candidates = list(Path("/kaggle/input").glob(f"**/{slug}/{filename}"))\n        if len(candidates) != 1: raise ValueError(f"Expected exactly one fold-0 input: {mapping['dataset']}/{filename}")\n        target = root / logical; target.parent.mkdir(parents=True, exist_ok=True); target.symlink_to(candidates[0])\n    command = [sys.executable, str(root / "phase_runner.py")]\n    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"\n    try: subprocess.run(command, check=True)\n    finally:\n        for logical in manifest["external_mounts"]:\n            target = root / logical\n            if target.is_symlink(): target.unlink()\nif __name__ == "__main__": main()\n'''
    (ROOT / CAMPAIGN / 'bundle/kernel/kernel.py').write_text(kernel)
    source_inventory = {
        'bundle_id': DATASET,
        'runner_id': RUNNER,
        'source_only': True,
        'archive_sha256': archive_sha,
        'builder_sha256': sha256_file(Path(__file__)),
        'phase_runner_sha256': sha256_file(ROOT / CAMPAIGN / 'phase_runner.py'),
        'archive_bytes': archive_size,
        'arm_names': {'probe': probe_names, 'fit': full_names},
        'spec_paths': spec_names,
        'plan_paths': plan_paths,
        'report_paths': report_paths,
        'input_paths': all_inputs,
        'external_dataset_sources': sorted(dataset_sources),
        'probe_protocol': 'pair chunk1000+connected first, then solitary accumulation probe; all probes required before fit',
        'fit_protocol': 'pair chunk1000+connected, require both terminal/complete, then solitary accumulation fit',
        'cardinality': {'fit_batches_per_epoch': 373, 'epochs': 4, 'fit_batches_total': 1492,
                        'updates_per_batch': {'chunk1000': 20, 'chunk2000_connected': 10,
                                              'accum2000_trunc1000': 10}},
        'resource_caps': {'remote_allocator_bytes': 11*1024**3,
                          'remote_gpu_operating_bytes': 12*1024**3,
                          'remote_gpu_stop_bytes': 13*1024**3,
                          'worker_ram_bytes': 10*1024**3,
                          'full_fit_timeout_seconds': 12600,
                          'probe_timeout_seconds': 1200},
        'time_model': {'probe_wall_upper_hours': 2/3, 'probe_allocated_gpu_hours_upper': 4/3,
                       'full_fit_wall_upper_hours': 7.0,
                       'full_fit_allocated_gpu_hours_upper': 14.0,
                       'all_stages_allocated_gpu_hours_upper': 15.34,
                       'planning_margin_wall_hours': 1.0,
                       'planning_allocated_gpu_hours_upper': 17.34,
                       'admission_reserve_gpu_hours': 6.0,
                       'quota_required_for_admission_gpu_hours': 23.34,
                       'wall_upper_hours_including_probes_and_margin': 8 + 2/3,
                       'assumption': 'Two GPUs remain allocated for solo stages; includes the idle companion GPU. Separate 3.5 h wall bounds per fit phase plus one hour total startup/input-verification margin.'},
    }
    write_json(ROOT / CAMPAIGN / 'bundle/source_inventory.json', source_inventory)
    print(json.dumps({'archive_sha256': archive_sha, 'archive_bytes': archive_size,
                      'code_files': len(code_files), 'specs': len(spec_names),
                      'inputs': len(all_inputs), 'bundle_manifest': str((ROOT / CAMPAIGN / 'bundle/bundle-manifest.json').relative_to(ROOT))}, indent=2))


if __name__ == '__main__':
    main()
