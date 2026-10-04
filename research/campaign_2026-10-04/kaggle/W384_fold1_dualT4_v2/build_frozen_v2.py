from __future__ import annotations
import gzip, hashlib, io, json, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
V1 = ROOT / 'research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v1'
HERE = ROOT / 'research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v2'
OLD = 'research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v1'
NEW = 'research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v2'
OLD_SLUG = 'wnn-w384-fold1-dualt4-20261004-v1'
SLUG = 'wnn-w384-fold1-dualt4-20261004-v2'
ASSET_DATASET = 'vladkochetov07/wnn-c20261004-fold1-assets'
RAW_DATASET = 'vladkochetov07/wunder-funder'


def sha(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def canon(v: object) -> str: return sha(json.dumps(v, sort_keys=True, separators=(',', ':')).encode())
def jdump(v: object) -> bytes: return (json.dumps(v, sort_keys=True, indent=2) + '\n').encode()
def rename(s: str) -> str:
    return s.replace(OLD, NEW).replace('t4v1', 't4v2').replace(OLD_SLUG, SLUG)


def main() -> None:
    old_manifest = json.loads((V1/'bundle/bundle-manifest.json').read_text())
    payload: dict[str, bytes] = {}
    with tarfile.open(V1/'bundle/bundle.wnn', 'r:gz') as src:
        for m in src:
            f = src.extractfile(m)
            if f is None: raise RuntimeError(f'no payload for {m.name}')
            payload[m.name] = f.read()
    # Preserve all sealed non-campaign source bytes exactly. Rewrite only the new controller,
    # screen and immutable plan/spec metadata to a distinct retry identity.
    old_phase = f'{OLD}/phase_runner.py'
    new_phase = f'{NEW}/phase_runner.py'
    phase = payload.pop(old_phase).decode().replace(OLD, NEW).replace('t4v1', 't4v2').encode()
    payload[new_phase] = phase
    old_screen = f'{OLD}/paired_fold1_screen.json'
    new_screen = f'{NEW}/paired_fold1_screen.json'
    payload[new_screen] = payload.pop(old_screen).decode().replace(OLD, NEW).replace('t4v1','t4v2').encode()
    plan_files = [n for n in list(payload) if n.startswith(f'{OLD}/plans/')]
    for name in plan_files:
        new_name = rename(name)
        data = payload.pop(name).decode().replace(OLD, NEW).replace('t4v1','t4v2').encode()
        payload[new_name] = data

    # Build four sealed specs from v1 while changing only identities, paths, plan/screen hashes,
    # and the probe wall timeout. Fixed-fit argv, hyperparameters, and fit timeout remain identical.
    v1_specs = sorted((V1/'specs').glob('*.sealed.json'))
    specs = []
    for p in v1_specs:
        spec = json.loads(p.read_text())
        old_name = spec['name']
        new_name = old_name.replace('t4v1','t4v2')
        spec['name'] = new_name
        spec['argv'] = [rename(x) if isinstance(x, str) else x for x in spec['argv']]
        spec['artifacts'] = [rename(x) for x in spec['artifacts']]
        spec['environment'] = {k: rename(v) if isinstance(v, str) else v for k, v in spec['environment'].items()}
        for group in ('code','inputs'):
            for item in spec[group]: item['path'] = rename(item['path'])
        # Update descriptor hashes/sizes for the only changed declared code/input files.
        for item in spec['code']:
            if item['path'] == new_phase:
                item.update(size=len(phase), sha256=sha(phase))
        for item in spec['inputs']:
            if item['path'] in payload:
                b = payload[item['path']]
                item.update(size=len(b), sha256=sha(b))
        spec['code_sha256'] = canon(sorted(spec['code'], key=lambda x: x['path']))
        if spec['limits']['timeout_seconds'] == 300: spec['limits']['timeout_seconds'] = 1200
        specs.append(spec)
        spec_member = f'kaggle_specs/{new_name}.json'
        payload[spec_member] = (json.dumps(spec, sort_keys=True) + '\n').encode()

    # All files declared by the four specs must have identical descriptors and resolve either to
    # an archive member or one of the exact v1 external mounts. External content is unchanged.
    files = {}
    for spec in specs:
        for item in spec['code'] + spec['inputs']:
            old = files.setdefault(item['path'], item)
            if old != item: raise RuntimeError(f'conflicting descriptor: {item["path"]}')
    external = {rename(k): v for k, v in old_manifest['external_mounts'].items()}
    if set(external) - set(files): raise RuntimeError('external mapping not declared')
    for name, item in files.items():
        if name in external: continue
        if name not in payload: raise RuntimeError(f'missing payload {name}')
        if len(payload[name]) != item['size'] or sha(payload[name]) != item['sha256']:
            raise RuntimeError(f'payload mismatch {name}')
    if len(specs) != 4: raise RuntimeError('expected four specs')

    out = HERE/'bundle'
    out.mkdir(exist_ok=False)
    archive = out/'bundle.wnn'
    with archive.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as gz, tarfile.open(fileobj=gz, mode='w|') as tar:
        for name, item in sorted(files.items()):
            if name in external: continue
            b = payload[name]
            ti = tarfile.TarInfo(name); ti.size=len(b); ti.mode=0o644
            tar.addfile(ti, io.BytesIO(b))
        for spec in specs:
            name = f'kaggle_specs/{spec["name"]}.json'; b=payload[name]
            ti=tarfile.TarInfo(name); ti.size=len(b); ti.mode=0o644
            tar.addfile(ti, io.BytesIO(b))
    manifest = {
      'version':1, 'dataset':f'vladkochetov07/{SLUG}', 'private':True,
      'archive_sha256':sha(archive.read_bytes()), 'archive_bytes':archive.stat().st_size,
      'expanded_bytes':sum(x['size'] for p,x in files.items() if p not in external),
      'files':[files[k] for k in sorted(files)], 'external_mounts':external,
      'spec_paths':[f'kaggle_specs/{s["name"]}.json' for s in specs],
    }
    (out/'bundle-manifest.json').write_bytes(jdump(manifest))
    (out/'dataset-metadata.json').write_bytes(jdump({'id':manifest['dataset'],'title':SLUG,'licenses':[{'name':'other'}],'isPrivate':True}))
    # Start with exact v1 kernel, rebinding only controller identity/path and archive hash.
    kernel = (V1/'bundle/kernel/kernel.py').read_text()
    kernel = kernel.replace(OLD, NEW).replace(OLD_SLUG, SLUG).replace(old_manifest['archive_sha256'], manifest['archive_sha256']).replace(old_phase, new_phase)
    (out/'kernel').mkdir()
    (out/'kernel/kernel.py').write_text(kernel)
    meta = json.loads((V1/'bundle/kernel/kernel-metadata.json').read_text())
    meta['id'] = f'vladkochetov07/{SLUG}-runner'; meta['title'] = f'{SLUG}-runner'
    meta['dataset_sources'][0] = manifest['dataset']
    (out/'kernel/kernel-metadata.json').write_bytes(jdump(meta))
    # Reviewable launch contract. Provider max includes the longer barrier probe plus fixed fit timeout.
    plan = {
      'kernel':meta['id'], 'dataset':manifest['dataset'], 'asset_dataset':ASSET_DATASET,
      'raw_dataset':RAW_DATASET, 'private':True, 'internet':False, 'accelerator':'2x NvidiaTeslaT4',
      'worker_fit_timeout_seconds':15600, 'worker_probe_timeout_seconds':1200,
      'provider_timeout_seconds':18000,
      'max_gpu_quota_charge_hours':10.0,
      'quota_admission': 'requires a fresh provider balance check; prior observed balance 24.27h before v1, v1 terminal runtime 958.4s (upper visible-device estimate 0.533h), so conservative modeled post-v1 balance 23.737h and post-v2 worst case 13.737h; >=6h reserve retained if provider accounting is within this bound',
      'pair_barrier':'both probes complete on distinct T4s and each below 12 GiB operating / 13 GiB stop before either fit',
      'v1_failure':'barrier timeout during source/input startup hashing; no probe forward/backward completed, no scientific fit started',
      'only_protocol_change':'fresh retry identity; probe timeout 300->1200 seconds; provider whole-job timeout 16200->18000 seconds. Scientific fit timeout/argv/source/data/model unchanged.',
      'pushed':False,
    }
    (HERE/'launch-plan.prepush.json').write_bytes(jdump(plan))
    (HERE/'provenance.prepush.json').write_bytes(jdump({
      'source_bundle_v1_sha256':old_manifest['archive_sha256'],
      'source_manifest_v1_sha256':sha((V1/'bundle/bundle-manifest.json').read_bytes()),
      'source_file_hashes_preserved_from_v1':True,
      'scientific_source_changes':[],
      'workflow_change':'fresh bundle identity; v1 probe timeout 300s -> v2 1200s; provider timeout 16200s -> 18000s; fit timeout 15600s unchanged',
      'phase_timing':'unknown within startup; verify_portable_source hashes relocated raw inputs before first model/device allocation; no phase checkpoint was recorded, so do not attribute v1 elapsed time to forward/backward',
    }))
    (HERE/'review_packet.json').write_bytes(jdump({
      'dataset':manifest['dataset'],'kernel':meta['id'],'archive_sha256':manifest['archive_sha256'],
      'manifest_sha256':sha((out/'bundle-manifest.json').read_bytes()),
      'kernel_sha256':sha((out/'kernel/kernel.py').read_bytes()),
      'kernel_metadata_sha256':sha((out/'kernel/kernel-metadata.json').read_bytes()),
      'dataset_metadata_sha256':sha((out/'dataset-metadata.json').read_bytes()),
      'specs':[{'name':s['name'],'limits':s['limits'],'code_sha256':s['code_sha256']} for s in specs],
      'file_count':len(files),'external_mount_count':len(external),
      'fit_source_same_as_v1':True,'fit_argv_same_as_v1':True,'fit_timeout_same_as_v1':True,
      'probe_timeout_seconds':1200,'provider_timeout_seconds':18000,'worst_case_device_gpu_hours':10.0,
      'pushed':False,
    }))
    print(json.dumps({'manifest':manifest,'specs':[s['name'] for s in specs]},indent=2))

if __name__ == '__main__': main()
