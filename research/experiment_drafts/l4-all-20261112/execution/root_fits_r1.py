"""Execute exactly eight ROOT-approved fresh fixed6E L4 fits, without SSH lifetime dependence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

R = Path('/data/lc-epanov_s/tabdata-all-20261112-r1')
S = Path('/data/lc-epanov_s/tabdata-stem-successors-20261111-r1/source-r1')
OLD = Path('/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/source-gru-monitor-resume-r1/src')
P = '/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python'
CELLS = ('gru256_layer1_k16_s203301', 'gru256_layer1_k16_s203302', 'gru256_layer3_k16_s203301', 'gru256_layer3_k16_s203302', 'horizon2000_k16_s203301', 'horizon2000_k16_s203302', 'horizon4000_k16_s203301', 'horizon4000_k16_s203302')
GPU_BY_CELL = {'gru256_layer1_k16_s203301': 0, 'gru256_layer1_k16_s203302': 1, 'gru256_layer3_k16_s203301': 2, 'gru256_layer3_k16_s203302': 3, 'horizon2000_k16_s203301': 4, 'horizon2000_k16_s203302': 5, 'horizon4000_k16_s203301': 6, 'horizon4000_k16_s203302': 7}
EXPECTED_ADMISSION_SHA = '776d13e9e3bb255d39c3aac7537153e1879375bb16e3d22ac6b533fc71a6caf5'


def digest(path: Path) -> str:
    """Stream a file digest without reading a model into RAM."""
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    """Own only these receipt-proven process groups and preserve every failed outcome."""
    sys.path.insert(0,str(OLD))
    from wnn.multistage_search import atomic_write_json, file_lock
    from wnn.multistage_search_owned_command import run_owned_command
    from wnn.multistage_search_runtime import current_process_owner, is_same_process, launch_detached_worker, terminate_owned_process
    from wnn.training.sequence_observer_resources import process_table, tracked_processes

    parser = argparse.ArgumentParser()
    parser.add_argument('--detach',action='store_true')
    args = parser.parse_args()
    control = R/'control/root-fits-r1'
    control.mkdir(parents=True,exist_ok=True)
    (R/'scratch').mkdir(exist_ok=True)
    if args.detach:
        print(json.dumps(launch_detached_worker([P,str(Path(__file__).resolve())],cwd=R,env=dict(os.environ),receipt_path=control/'owner.json',metadata={'new_fits':8,'epochs_each':6})))
        return
    with file_lock(control/'fits.lock',blocking=False):
        verify = runpy.run_path(str(S/'src/wnn/training/sequence_source_capsule.py'))['verify_source_capsule']
        verify(S,S/'source-manifest.json','e2220d547dc92f791509e48762f164f70a3c064d01a4136156f8cc541dfb43d0','6b10dfe71fb0ebec9b0fc76788e2de0fd13d1dba')
        assert digest(R/'l4-fit-receiver-r1/ROOT-fit-admission-r1.json')==EXPECTED_ADMISSION_SHA
        admission = json.loads((R/'l4-fit-receiver-r1/ROOT-fit-admission-r1.json').read_text())
        assert admission['approved'] and admission['max_fresh_fits']==8 and admission['source_root']==str(S)
        assert admission['schema']=='wnn.l4-first-eight-root-fit-admission.r1'
        assert admission['source_sha256']=='38d6dd71ec29fb8d23b9df801a448f23fb8b78c1e0832883dda1d6f25095643d'
        assert admission['split_sha256']=='0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c'
        assert len(admission['profiles'])==8 and set(admission['profiles'])==set(CELLS)
        assert all(admission['profiles'][cell]['gpu']==GPU_BY_CELL[cell] for cell in CELLS)
        assert len({v['gpu_uuid'] for v in admission['profiles'].values()})==8
        clock = json.loads((R/'control/CLOCK.json').read_text())
        assert clock['fit_deadline_epoch']==1794468019.060462 and clock['wave_deadline_epoch']==1794471619.060462
        owner = current_process_owner()
        lock = threading.Lock()
        resource_error: list[str] = []
        last_check = [0.0]

        def guard() -> None:
            """Latch actual disk/RSS/deadline failures and never alter foreign processes."""
            with lock:
                if resource_error:
                    raise RuntimeError(resource_error[0])
                if time.time()>=clock['fit_deadline_epoch']:
                    resource_error.append('Original fit deadline reached')
                if time.monotonic()-last_check[0]>=10:
                    last_check[0]=time.monotonic()
                    identities = {}
                    for receipt in (control/'owned').glob('*.json'):
                        child=json.loads(receipt.read_text()).get('child')
                        if isinstance(child,dict) and is_same_process(child):
                            identities[int(child['pid'])]=int(child['birth'])
                    processes=tracked_processes(process_table(),identities)
                    rss=sum(v.resident_bytes for v in processes.values())
                    output_bytes=0
                    for folder,dirs,files in os.walk(R,followlinks=False):
                        dirs[:]=[n for n in dirs if not (Path(folder)/n).is_symlink()]
                        for name in files:
                            path=Path(folder)/name
                            if path.is_file() and not path.is_symlink(): output_bytes+=path.stat().st_size
                    free=shutil.disk_usage(R).free
                    if rss>128*1073741824 or output_bytes>1100000000 or free<2500000000:
                        resource_error.append(f'Resource cap: ownRSS={rss} outputBytes={output_bytes} diskFree={free}')
                    atomic_write_json(control/'resource_status.json',{'at':time.time(),'own_rss_bytes':rss,'output_bytes':output_bytes,'free_bytes':free,'errors':resource_error})
                if resource_error:
                    raise RuntimeError(resource_error[0])

        def fit(item: tuple[str,dict[str,object]]) -> None:
            """Claim once; wrap the new-source trainer with the proven observer owner."""
            cell, spec = item
            config = R/'l4-profile-receiver-r1/configs'/f'{cell}.json'
            profile = Path(spec['receipt_path'])
            assert digest(config)==spec['raw_config_sha256'] and digest(profile)==spec['receipt_sha256']
            output = R/'runs'/cell/'fit'
            assert not output.exists()
            atomic_write_json(control/f'claim-{cell}.json',{'cell':cell,'spec':spec,'owner':owner},replace=False)
            environment=dict(os.environ,PYTHONPATH=str(S/'src'),CUDA_VISIBLE_DEVICES=str(spec['gpu_uuid']),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NVIDIA_TF32_OVERRIDE='0',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',PYTHONDONTWRITEBYTECODE='1',WNN_MONITOR_HEARTBEAT_SECONDS='10',TMPDIR=str(R/'scratch'))
            for key in ('RANK','WORLD_SIZE','LOCAL_RANK','CUBLAS_WORKSPACE_CONFIG','CUBLAS_EMULATE_SINGLE_PRECISION'):
                environment.pop(key,None)
            environment['CUDA_VISIBLE_DEVICES']='GPU-'+str(spec['gpu_uuid']).removeprefix('GPU-')
            train=[P,str(S/'scripts/train_sequence.py'),'--config',str(config),'--data-root',str(R/'project'),'--split-manifest',str(R/'project/chronological_primary.json'),'--output',str(output),'--profile-receipt',str(profile)]
            observed='/data2/lc-epanov_s/tabdata/runtime/generalization-fix-20261108-r1/control/l4-wave-r1/observed_worker.py'
            bootstrap='import sys,runpy;sys.path.insert(0,'+repr(str(OLD))+');runpy.run_path('+repr(observed)+',run_name="__main__")'
            seconds=int(clock['fit_deadline_epoch']-time.time())
            assert seconds>0
            command=[P,'-c',bootstrap,'--',P,str(S/'scripts/observe_sequence.py'),'--output',str(output),'--interval','10','--max-seconds',str(seconds),'--',*train]
            receipt=control/'owned'/f'{cell}.json'
            failures: list[str] = []

            def progress() -> None:
                """Terminate only this exact birth-bound child when a guard latches."""
                try:
                    guard()
                except Exception as error:
                    if not failures:
                        failures.append(str(error))
                        if receipt.exists():
                            record=json.loads(receipt.read_text())
                            child=record.get('child')
                            assert record.get('command')==command and record.get('owner')=={'controller':owner,'cell_id':cell}
                            if isinstance(child,dict) and is_same_process(child): terminate_owned_process(child,grace_seconds=0.0)

            code,timed_out,_=run_owned_command(command,cwd=S,env=environment,log_path=control/f'{cell}.log',receipt_path=receipt,owner={'controller':owner,'cell_id':cell},stage=1,trial_number=int(spec['gpu']),mode='fit',gpu_key=str(spec['gpu']),output_dir=output,timeout_seconds=seconds,progress=progress)
            result={'cell':cell,'exit_code':code,'timed_out':timed_out,'resource_errors':failures,'at':time.time()}
            if code==0 and not timed_out and not failures:
                terminal=json.loads((output/'terminal_status.json').read_text())
                events=[json.loads(line) for line in (output/'monitor_evaluations.jsonl').read_bytes().splitlines()]
                assert terminal['status']=='completed' and terminal['completed_epochs']==6 and terminal['global_updates']==spec['total_updates']
                assert terminal['model_sha256']==digest(output/'model.pt')
                assert len(events)==6 and all(e['epoch']==i+1 and e['metrics']['valid/sequence_count']==1873 and e['metrics']['valid/selected_rows']==3528273 for i,e in enumerate(events))
                result.update(status='fit_complete',model_sha256=terminal['model_sha256'],final_metrics=events[-1]['metrics'])
            else:
                result['status']='failed_or_partial'
            atomic_write_json(control/f'{cell}-result.json',result,replace=False)
            print(json.dumps(result),flush=True)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fit,admission['profiles'].items()))
        atomic_write_json(control/'complete.json',{'worker_exits_collected':8,'at':time.time()},replace=False)


if __name__=='__main__':
    main()
