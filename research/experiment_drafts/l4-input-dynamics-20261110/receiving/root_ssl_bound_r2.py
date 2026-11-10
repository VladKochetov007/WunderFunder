"""Execute exactly two fresh2E-pretext -> fresh-optimizer4E-shared-regression L4 chains."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

R=Path('/data/lc-epanov_s/tabdata-next70-20261110-r1')
S=R/'source-ssl-r1'
OLD=Path('/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/source-gru-monitor-resume-r1/src')
P='/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python'
SOURCE_SHA='b917b83edd463760d83829e44f87b57af5473507302e558d36fad3d1034cd01b'
SOURCE_COMMIT='c157877d261efaa0a58cd1ae8ba5fc61c914b72d'
MANIFEST_SHA='dedc1bd13701ba1317a03f71566b0487d915db379b52c4b81f8f744dcbaafaea'
AUTHORITY_SHA='b544180c502a94344e12bc94f08f02cced5a1f399a3d46e39d7a3b62197ac103'
GPUS={203201:(6,'004cdb0f-3bf7-0003-702e-0e2fb87215e8'),203202:(7,'cd069633-c451-769e-3727-cb1fe310e407')}


def sha(path: Path) -> str:
    """Read file identity in bounded blocks."""
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''): value.update(chunk)
    return value.hexdigest()


def main() -> None:
    """Use ROOT's frozen conditional release gate, never a valid-score selection rule."""
    sys.path.insert(0,str(OLD))
    from wnn.multistage_search import atomic_write_json, file_lock
    from wnn.multistage_search_owned_command import run_owned_command
    from wnn.multistage_search_runtime import current_process_owner, is_same_process, launch_detached_worker, terminate_owned_process
    from wnn.training.sequence_observer_resources import process_table, tracked_processes

    parser=argparse.ArgumentParser()
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args()
    control=R/'control/root-ssl-r2'
    control.mkdir(parents=True,exist_ok=True)
    if args.detach:
        print(json.dumps(launch_detached_worker([P,str(Path(__file__).resolve())],cwd=R,env=dict(os.environ),receipt_path=control/'owner.json',metadata={'new_training_stages':4,'pretrain_epochs':2,'shared_child_epochs':4})))
        return
    with file_lock(control/'cohort.lock',blocking=False):
        verify=runpy.run_path(str(S/'src/wnn/training/sequence_source_capsule.py'))['verify_source_capsule']
        verify(S,S/'source-manifest.json',MANIFEST_SHA,SOURCE_COMMIT)
        auth_path=R/'ssl-inputs-r2/ROOT-conditional-admission-r2.json'
        assert sha(auth_path)==AUTHORITY_SHA
        auth=json.loads(auth_path.read_text())
        assert auth['approved'] and auth['source_sha256']==SOURCE_SHA and auth['max_new_training_stages']==4 and auth['max_profiles']==4
        cells={f'ssl_{kind}_s{seed}' for kind in ('pretrain','finetune') for seed in GPUS}
        assert set(auth['configs'])==cells
        clock=json.loads((R/'receipts/first-profile-r1.json').read_text())
        assert clock['fit_deadline_epoch']==clock['first_profile_epoch']+39600
        owner=current_process_owner()
        lock=threading.RLock()
        errors=[]
        last_check=[0.0]

        def guard() -> None:
            """Bound combined campaign RSS/disk and the original first-profile deadline."""
            with lock:
                if time.time()>=clock['fit_deadline_epoch'] and not errors: errors.append('Original fit deadline reached')
                if not errors and time.monotonic()-last_check[0]>=10:
                    last_check[0]=time.monotonic()
                    identities={}
                    for directory in (control/'owned',R/'control/root-fits-r1/owned'):
                        for path in directory.glob('*.json'):
                            child=json.loads(path.read_text()).get('child')
                            if isinstance(child,dict) and is_same_process(child): identities[int(child['pid'])]=int(child['birth'])
                    rss=sum(p.resident_bytes for p in tracked_processes(process_table(),identities).values())
                    size=0
                    for folder,dirs,files in os.walk(R,followlinks=False):
                        dirs[:]=[n for n in dirs if not (Path(folder)/n).is_symlink()]
                        for name in files:
                            path=Path(folder)/name
                            if path.is_file() and not path.is_symlink(): size+=path.stat().st_size
                    free=shutil.disk_usage(R).free
                    if rss>128*1073741824 or size>6000000000 or free<4000000000: errors.append(f'Combined resource cap RSS={rss}, outputs={size}, diskFree={free}')
                    atomic_write_json(control/'resource_status.json',{'at':time.time(),'combined_owned_rss_bytes':rss,'output_bytes':size,'free_bytes':free,'errors':errors})
                if errors: raise RuntimeError(errors[0])

        def execute(cell: str, stage: str, command: list[str], gpu: int | None, output: Path, cap: int) -> None:
            """Keep a receipt-bound leader alive until every observed child exits."""
            guard()
            seconds=min(cap,int(clock['fit_deadline_epoch']-time.time()))
            assert seconds>0
            threads='1' if gpu is None else '8'
            env=dict(os.environ,PYTHONPATH=str(S/'src'),OMP_NUM_THREADS=threads,MKL_NUM_THREADS=threads,OPENBLAS_NUM_THREADS=threads,NVIDIA_TF32_OVERRIDE='0',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',PYTHONDONTWRITEBYTECODE='1')
            for key in ('RANK','WORLD_SIZE','LOCAL_RANK','CUBLAS_WORKSPACE_CONFIG','CUBLAS_EMULATE_SINGLE_PRECISION'): env.pop(key,None)
            env['CUDA_VISIBLE_DEVICES']='' if gpu is None else str(gpu)
            observed='/data2/lc-epanov_s/tabdata/runtime/generalization-fix-20261108-r1/control/l4-wave-r1/observed_worker.py'
            boot='import sys,runpy;sys.path.insert(0,'+repr(str(OLD))+');runpy.run_path('+repr(observed)+',run_name="__main__")'
            if stage in ('profile','fit'):
                command=[P,str(S/'scripts/observe_sequence.py'),'--output',str(output),'--interval','10','--max-seconds',str(seconds),'--',*command]
            argv=[P,'-c',boot,'--',*command]
            receipt=control/'owned'/f'{cell}-{stage}.json'
            expected_owner={'controller':owner,'cell_id':cell,'stage':stage}
            failures=[]

            def progress() -> None:
                """Stop only this exact owned group when one resource guard latches."""
                try: guard()
                except Exception as error:
                    if not failures:
                        failures.append(str(error))
                        if receipt.exists():
                            record=json.loads(receipt.read_text())
                            assert record.get('command')==argv and record.get('owner')==expected_owner
                            child=record.get('child')
                            if isinstance(child,dict) and is_same_process(child): terminate_owned_process(child,grace_seconds=0.0)

            atomic_write_json(control/f'{cell}-status.json',{'cell':cell,'stage':stage,'at':time.time()})
            code,timeout,_=run_owned_command(argv,cwd=S,env=env,log_path=control/f'{cell}-{stage}.log',receipt_path=receipt,owner=expected_owner,stage=1,trial_number=0 if gpu is None else gpu,mode=stage,gpu_key='cpu' if gpu is None else str(gpu),output_dir=output,timeout_seconds=seconds,progress=progress)
            atomic_write_json(control/f'{cell}-{stage}-exit.json',{'exit_code':code,'timed_out':timeout,'errors':failures,'at':time.time()},replace=False)
            if code or timeout or failures: raise RuntimeError(f'{cell}/{stage} failed; no automatic retry')

        # One preparation pass avoids two independent complete fitting-data scans.
        execute('target-preparation','prepare',[P,str(R/'ssl-inputs-r2/prepare_ssl_targets_r2.py')],None,control/'target-preparation',1200)
        preparation=json.loads((control/'target-preparation-receipt.json').read_text())
        assert preparation['status']=='complete' and preparation['source_sha256']==SOURCE_SHA
        assert preparation['fitting_sequence_count']==10325 and preparation['validation_fitting_members']==0
        assert preparation['strict_serial_parallel_parity'] is True and preparation['strict_serial_parallel_reference_groups']==8
        assert preparation['preparation_source_sha256']==auth['preparation_source_sha256']

        def chain(seed: int) -> None:
            """Admit only a fixed2E parent followed by its same-seed shared4E child."""
            gpu,uuid=GPUS[seed]
            parent=R/'runs'/f'ssl_pretrain_s{seed}'/'fit/future_innovation_parent_receipt.json'
            for kind,epochs in (('pretrain',2),('finetune',4)):
                cell=f'ssl_{kind}_s{seed}'
                config=R/'ssl-inputs-r1/configs'/f'{cell}.json'
                expected=auth['configs'][cell]
                assert sha(config)==expected['raw_sha256']
                profile=R/'runs'/cell/'profile'
                output=R/'runs'/cell/'fit'
                assert not profile.exists() and not output.exists()
                base=[P,str(S/'scripts/train_sequence.py'),'--config',str(config),'--data-root',str(R/'project'),'--split-manifest',str(R/'project/chronological_primary.json')]
                if kind=='finetune':
                    receipt=json.loads(parent.read_text())
                    assert receipt['seed']==seed and receipt['checkpoint_sha256']==sha(parent.parent/'model.pt')
                    base+=['--initializer-receipt',str(parent)]
                execute(cell,'profile',base+['--output',str(profile),'--profile-updates','30'],gpu,profile,600)
                receipt=json.loads((profile/'profile_receipt.json').read_text())
                identity=receipt['identity']; data=identity['data']
                assert receipt['status']=='profile_complete' and receipt['approved'] is False and receipt['updates']==30 and receipt['errors']==[] and receipt['scientific_checkpoint_written'] is False
                assert identity['source_sha256']==SOURCE_SHA and identity['config_sha256']==expected['normalized_sha256'] and identity['world_size']==1
                assert identity['optimizer']['total_updates']==6460*epochs and identity['runtime']['precision']['mode']=='fp32'
                devices=identity['runtime']['devices']
                assert len(devices)==1 and devices[0]['uuid'].removeprefix('GPU-')==uuid and devices[0]['name']=='NVIDIA L4'
                assert data['split_sha256']==auth['split_sha256'] and len(data['fit_members'])==10325 and all(x[0]=='train' for x in data['fit_members']) and len({tuple(x) for x in data['fit_members']})==10325
                assert len(data['evaluation_members'])==1873 and data['normalization']['scope']=='all_fit' and data['normalization']['sequence_count']==10325 and not data.get('teacher') and not data.get('reference')
                assert data['normalization']['moments_sha256']==auth['baseline_input_moments_sha256']
                reload=receipt['engineering_reload']
                assert reload['status']=='passed' and len(reload['by_rank'])==1 and reload['by_rank'][0]['model_optimizer_scheduler_rng_equal'] is True
                assert all(math.isfinite(float(x)) for x in receipt['losses'].values()) and receipt['losses']['total']>0
                assert receipt['peak_reserved_bytes']<0.95*devices[0]['total_memory_bytes']
                if kind=='pretrain':
                    innovation=data['future_input_innovations']
                    assert innovation['validation_members_used'] is False and innovation['label_values_read'] is False
                    assert innovation['moments_sha256']==preparation['innovation']['moments_sha256'] and innovation['horizons']==[16,64,256]
                    assert receipt['pretext']['validation_quality_evaluated'] is False
                else:
                    init=data['initializer']
                    assert init['receipt_sha256']==sha(parent) and init['checkpoint_sha256']==sha(parent.parent/'model.pt') and init['optimizer_state']=='reset' and init['scheduler_state']=='reset'
                receipt['approved']=True
                receipt['approval_note']='ROOT conditional gate: frozen source/config/data/seed/parent plus actual finite-work and full-state reload; no quality selection.'
                receipt['conditional_authority_sha256']=AUTHORITY_SHA
                approved=control/'approved'/f'{cell}.json'
                atomic_write_json(approved,receipt,replace=False)
                atomic_write_json(control/f'{cell}-fit-claim.json',{'source_sha256':SOURCE_SHA,'config_sha256':expected['normalized_sha256'],'approved_profile_sha256':sha(approved),'new_stage':True},replace=False)
                execute(cell,'fit',base+['--output',str(output),'--profile-receipt',str(approved)],gpu,output,39600)
                terminal=json.loads((output/'terminal_status.json').read_text())
                assert terminal['status']=='completed' and terminal['completed_epochs']==epochs and terminal['global_updates']==6460*epochs and terminal['model_sha256']==sha(output/'model.pt')
                result={'cell':cell,'status':'complete','completed_epochs':epochs,'model_sha256':terminal['model_sha256'],'at':time.time()}
                if kind=='finetune':
                    events=[json.loads(line) for line in (output/'monitor_evaluations.jsonl').read_bytes().splitlines()]
                    assert len(events)==4 and all(e['epoch']==i+1 and e['metrics']['valid/sequence_count']==1873 and e['metrics']['valid/selected_rows']==3528273 for i,e in enumerate(events))
                    result['final_metrics']=events[-1]['metrics']
                else:
                    assert parent.exists() and not (output/'monitor_evaluations.jsonl').exists()
                    result['parent_receipt_sha256']=sha(parent)
                atomic_write_json(control/f'{cell}-result.json',result,replace=False)
                print(json.dumps(result),flush=True)

        with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(chain,GPUS))
        atomic_write_json(control/'complete.json',{'completed_new_training_stages':4,'at':time.time()},replace=False)


if __name__=='__main__':
    main()
