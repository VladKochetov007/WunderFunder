"""Allocation-gated, target-free 39.4M fixed-four callback audit of the newly packaged ZIP."""
import time
WORKER_START = time.perf_counter()
import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import signal
import struct
import sys
import tempfile
from types import SimpleNamespace
from protocol import ROW_GROUPS, CALLBACKS, EMITTED, callback_identity, extract_exact

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CPU = SMT = None
THREAD_KEYS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def counters():
    return {p[0]:list(map(int,p[1:9])) for p in
            (line.split() for line in Path('/proc/stat').read_text().splitlines())
            if p and p[0] in (f'cpu{CPU}',f'cpu{SMT}')}


def activity(before,after):
    result = {}
    for key in before:
        d = [e-b for b,e in zip(before[key],after[key])]
        assert min(d)>=0
        total=sum(d);busy=total-d[3]-d[4]
        result[key]={'total_ticks':total,'busy_ticks':busy,
                     'busy_fraction':busy/total if total else 0.0}
    return result


def summed_activity(rows):
    result={}
    for key in (f'cpu{CPU}',f'cpu{SMT}'):
        total=sum(row['callback_activity'][key]['total_ticks'] for row in rows)
        busy=sum(row['callback_activity'][key]['busy_ticks'] for row in rows)
        assert total>0
        result[key]={'total_ticks':total,'busy_ticks':busy,'busy_fraction':busy/total}
    return result


def process_status():
    fields=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    tasks={}
    for path in Path('/proc/self/task').iterdir():
        # stat fields after the last ')' begin with state (field3); utime/stime are14/15.
        tail=(path/'stat').read_text().rsplit(')',1)[1].split()
        affinity=sorted(os.sched_getaffinity(int(path.name)))
        assert affinity==[CPU],('thread escaped one-CPU affinity',path.name,affinity)
        tasks[path.name]={'user_ticks':int(tail[11]),'system_ticks':int(tail[12]),'affinity':affinity}
    return {'threads':int(fields['Threads']),
            'vm_rss_KiB':int(fields.get('VmRSS','0 kB').split()[0]),
            'vm_hwm_KiB':int(fields.get('VmHWM','0 kB').split()[0]),'tasks':tasks}


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def stat_identity(path):
    st=path.stat()
    return {'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'resolved_path':str(path.resolve())}


def cgroup_limits():
    relative=Path('/proc/self/cgroup').read_text().strip().split(':',2)[2]
    base=Path('/sys/fs/cgroup')/relative.lstrip('/')
    result={'path':relative,**{name:(base/name).read_text().strip() for name in ('cpu.max','memory.max','memory.swap.max')}}
    assert result['cpu.max']=='100000 100000'
    assert int(result['memory.max'])==16*(1<<30) and int(result['memory.swap.max'])==0
    return result


def main():
    global CPU,SMT
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--allocation',type=Path,required=True);ap.add_argument('--allocation-sha256',required=True)
    args=ap.parse_args();plan_path=args.plan.resolve();allocation_path=args.allocation.resolve()
    assert sha(allocation_path)==args.allocation_sha256
    allocation=json.loads(allocation_path.read_text());plan=json.loads(plan_path.read_text())
    assert allocation['midpoint_runtime_audit_authorized'] is True
    CPU=int(plan['resources']['CPU']);SMT=int(plan['resources']['SMT_sibling'])
    assert CPU!=SMT
    assert allocation['plan_sha256']==sha(plan_path) and allocation['attempts']==1
    assert plan['id']=='MIDPOINT-FINAL-INT8-RUNTIME-01' and plan['execution_authorized'] is False
    output=HERE/'execution';assert output.is_dir() and not (output/'worker-result.json').exists()
    rows=[];record={'status':'STARTED','utc':now(),'pid':os.getpid(),
                    'plan_sha256':sha(plan_path),'allocation_sha256':args.allocation_sha256,
                    'targets_decoded':False,'scoring_masks_read':False,'WP_queries':0,
                    'callback_count':0,'emitted_count':0,'completed_sequences':0}
    signal.signal(signal.SIGTERM,lambda signum,frame:(_ for _ in ()).throw(TimeoutError('external wall cutoff')))
    try:
        assert sys.version_info[:2]==(3,11)
        assert sorted(os.sched_getaffinity(0))==[CPU]
        assert all(os.environ.get(k)=='1' for k in THREAD_KEYS)
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        assert all(k not in os.environ for k in ('WNN_NATIVE_LEVEL','WNN_EXPECTED_ROWS','WNN_BUDGET_S'))
        record['cgroup_limits']=cgroup_limits()
        cap=16*(1<<30);resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
        phase=time.perf_counter()
        for rel,expected in plan['source_sha256'].items(): assert sha(ROOT/rel)==expected,rel
        for name,entry in plan['environment_sha256'].items(): assert sha(Path(plan['scorer_environment'])/name)==entry['sha256'],name
        record['source_guard_seconds']=time.perf_counter()-phase
        package=ROOT/plan['artifact']['path'];assert package.stat().st_size==plan['artifact']['bytes'];assert sha(package)==plan['artifact']['sha256']
        data_path=ROOT/plan['input']['path'];expected_stat={k:plan['input'][k] for k in ('bytes','mtime_ns','resolved_path')}
        assert stat_identity(data_path)==expected_stat
        phase=time.perf_counter();before_hash=sha(data_path);record['input_hash_before_seconds']=time.perf_counter()-phase
        assert before_hash==plan['input']['sha256'];record['input_identity_before']={**stat_identity(data_path),'sha256':before_hash}
        b=counters();time.sleep(2.0);record['quiet_precheck']=activity(b,counters())
        record['quiet_precheck_below5percent']=record['quiet_precheck'][f'cpu{SMT}']['busy_fraction']<.05
        class NoTorch:
            def find_spec(self,fullname,path=None,target=None):
                if fullname=='torch' or fullname.startswith('torch.'): raise AssertionError('Torch runtime forbidden')
        sys.meta_path.insert(0,NoTorch())
        phase=time.perf_counter()
        import numpy as np
        import pyarrow as pa
        pa.set_cpu_count(1);pa.set_io_thread_count(1)
        sys.path.insert(0,str(ROOT/'src'))
        from wnn.data import SequenceFile,FEATURE_COLUMNS
        assert FEATURE_COLUMNS==plan['input']['feature_columns']
        record['dependency_import_seconds']=time.perf_counter()-phase
        record['library_versions']={'numpy':np.__version__,'pyarrow':pa.__version__}
        assert record['library_versions']==plan['library_versions']
        record['process_status_after_dependencies']=process_status()
        with tempfile.TemporaryDirectory(prefix='fixed4-int8-runtime-') as temp:
            extracted=Path(temp)/'package';extracted.mkdir();empty=Path(temp)/'empty-cwd';empty.mkdir()
            phase=time.perf_counter()
            extract_exact(package,extracted,plan['artifact']['files'])
            record['extraction_seconds']=time.perf_counter()-phase
            os.chdir(empty);sys.path.insert(0,str(extracted));phase=time.perf_counter()
            module=importlib.import_module('solution');model=module.PredictionModel()
            record['solution_import_model_seconds']=time.perf_counter()-phase
            assert Path(module.__file__).resolve()==extracted/'solution.py'
            assert module.LEVEL=='v3' and model.member_count==4 and model.quant_backend==2
            assert json.loads((extracted/'members.json').read_text())==plan['artifact']['member_spec']
            assert model.seq_ix is None
            native_file=module.LEVEL+'/wnn_cpp.cpython-311-x86_64-linux-gnu.so'
            assert Path(module.wnn_cpp.__file__).resolve()==extracted/native_file
            assert sha(module.wnn_cpp.__file__)==plan['artifact']['files'][native_file]['sha256']
            record['natural_native_level']=module.LEVEL;record['integer_backend']=model.quant_backend
            record['fixed_members']=plan['artifact']['members'];record['fixed_coefficients']=[.25]*4
            phase=time.perf_counter();data=SequenceFile(data_path);record['parquet_open_seconds']=time.perf_counter()-phase
            assert len(data)==1873 and data._pf.metadata.num_rows==37460000
            native_reader=data._pf;allowed=['seq_ix',*FEATURE_COLUMNS];read_calls=[]
            class InputOnlyReader:
                def read_row_group(self,index,columns,use_threads):
                    assert columns==allowed and use_threads is False
                    table=native_reader.read_row_group(index,columns=columns,use_threads=False)
                    ids=table.column('seq_ix').to_numpy()
                    assert len(ids)==20000 and np.all(ids==ids[0])
                    read_calls.append({'row_group':int(index),'seq_ix':int(ids[0]),'rows':len(ids)})
                    return table
            data._pf=InputOnlyReader()
            seen=set();raw_ids=[];next_fresh=0;raw_digest=hashlib.sha256();pred_digest=hashlib.sha256()
            schedule=list(ROW_GROUPS)
            assert CALLBACKS==39400000 and EMITTED==39204970
            callback_wall=0.;callback_cpu=0.;io_wall=0.;validation_wall=0.
            first_callback_elapsed=None;last_callback_elapsed=None;whole_before=counters();trace_start=time.perf_counter()
            log=output/'sequences.jsonl'
            assert not log.exists()
            for index,row_group in enumerate(schedule):
                seq_start=time.perf_counter();sample_before=process_status()
                assert sample_before['vm_rss_KiB']<=16*(1<<20)
                io_start=time.perf_counter();sequence=data.read(row_group,targets=False,mask=False);io_seconds=time.perf_counter()-io_start
                assert sequence.targets is None and sequence.scored is None
                assert sequence.features.shape==(20000,112) and sequence.features.dtype==np.float32
                callback_id,next_fresh=callback_identity(index,row_group,sequence.seq_ix,raw_ids,seen,next_fresh)
                if index<1873:raw_ids.append(sequence.seq_ix)
                seen.add(callback_id)
                check_start=time.perf_counter();assert np.isfinite(sequence.features).all()
                raw_digest.update(struct.pack('<qqqq',index,row_group,sequence.seq_ix,callback_id));raw_digest.update(sequence.features.astype('<f4',copy=False).tobytes(order='C'))
                validation_seconds=time.perf_counter()-check_start
                values=np.zeros((20000,2),np.float32)
                point=SimpleNamespace(seq_ix=callback_id,step_in_seq=0,need_prediction=False,state=None)
                assert model.seq_ix is None or model.seq_ix!=callback_id
                assert model.member_count==4
                b=counters();cpu0=time.process_time();t0=time.perf_counter()
                if first_callback_elapsed is None:first_callback_elapsed=t0-WORKER_START
                for step,row in enumerate(sequence.features):
                    point.step_in_seq=step;point.need_prediction=step>=99;point.state=row
                    value=model.predict(point)
                    if step<99: assert value is None
                    else:
                        assert value is not None and value.shape==(2,) and value.dtype==np.float32
                        values[step]=value
                wall=time.perf_counter()-t0;cpu=time.process_time()-cpu0;e=counters()
                last_callback_elapsed=time.perf_counter()-WORKER_START
                assert model.seq_ix==callback_id and model.member_count==4
                check_start=time.perf_counter();assert np.isfinite(values).all()
                body=values[99:].astype('<f4',copy=False).tobytes(order='C')
                pred_digest.update(struct.pack('<qqqq',index,row_group,sequence.seq_ix,callback_id));pred_digest.update(body)
                pred_sha=hashlib.sha256(body).hexdigest();validation_seconds+=time.perf_counter()-check_start
                sample_after=process_status()
                item={'schedule_index':index,'row_group':row_group,'raw_seq_ix':sequence.seq_ix,'seq_ix':callback_id,
                      'repeated_raw_sequence':index>=1873,
                      'fixed_member_count':model.member_count,'callbacks':20000,'emitted':19901,
                      'io_decode_conversion_seconds':io_seconds,'callback_loop_wall_seconds':wall,
                      'callback_loop_cpu_seconds':cpu,'validation_checksum_seconds':validation_seconds,
                      'sequence_total_seconds':time.perf_counter()-seq_start,'callback_activity':activity(b,e),
                      'status_before':sample_before,'status_after':sample_after,'sample_utc':now(),
                      'emitted_f32_sha256':pred_sha,'wrapper_seq_ix_after':model.seq_ix}
                rows.append(item)
                with log.open('a') as stream:stream.write(json.dumps(item,allow_nan=False)+'\n');stream.flush()
                callback_wall+=wall;callback_cpu+=cpu;io_wall+=io_seconds;validation_wall+=validation_seconds
                record.update(callback_count=(index+1)*20000,emitted_count=(index+1)*19901,completed_sequences=index+1)
                if index%25==0:
                    atomic_json(output/'progress.json',{**record,'callback_seconds':callback_wall,'worker_elapsed_seconds':time.perf_counter()-WORKER_START})
                    print(json.dumps({'completed_sequences':index+1,'total_sequences':1970,'elapsed_seconds':time.perf_counter()-WORKER_START}),flush=True)
            whole_activity=activity(whole_before,counters());callback_activity=summed_activity(rows)
            record.update(trace_seconds=time.perf_counter()-trace_start,callback_loop_wall_seconds=callback_wall,
                          callback_loop_cpu_seconds=callback_cpu,io_decode_conversion_seconds=io_wall,
                          validation_checksum_seconds=validation_wall,callback_average_us=callback_wall/39400000*1e6,
                          actual_callback_minutes_39_4M=callback_wall/60,
                          callback_activity=callback_activity,whole_trace_activity=whole_activity,
                          callback_smt_clean=callback_activity[f'cpu{SMT}']['busy_fraction']<.05,
                          whole_trace_smt_clean=whole_activity[f'cpu{SMT}']['busy_fraction']<.05,
                          first_callback_worker_elapsed_seconds=first_callback_elapsed,
                          last_callback_worker_elapsed_seconds=last_callback_elapsed,
                          raw_ordered_sha256=raw_digest.hexdigest(),emitted_ordered_sha256=pred_digest.hexdigest(),
                          raw_read_calls=read_calls,unique_sequence_ids=len(seen),sequence_resets_observed=1970,unique_raw_sequences=len(raw_ids),repeated_raw_rows=1940000,
                          final_member_count=model.member_count,
                          opaque_native_hidden_state_serialized=False,
                          max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          final_process_status=process_status(),native_sha256=sha(module.wnn_cpp.__file__))
            assert record['callback_count']==39400000 and record['emitted_count']==39204970
            assert len(read_calls)==1970 and [r['row_group'] for r in read_calls]==schedule
            assert len(seen)==1970 and len(set(raw_ids))==1873
            record['warmup_none_count']=1970*99
            record['member_count_transitions']=[]
            record['max_observed_os_threads']=max([record['process_status_after_dependencies']['threads'],
                record['final_process_status']['threads'],*[r[k]['threads'] for r in rows for k in ('status_before','status_after')]])
            assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
            for name,entry in plan['artifact']['files'].items():
                if entry['is_dir']:assert (extracted/name).is_dir()
                else:assert sha(extracted/name)==entry['sha256']
        phase=time.perf_counter();after_hash=sha(data_path);record['input_hash_after_seconds']=time.perf_counter()-phase
        assert after_hash==before_hash and stat_identity(data_path)==expected_stat
        record['input_identity_after']={**stat_identity(data_path),'sha256':after_hash}
        for rel,expected in plan['source_sha256'].items():assert sha(ROOT/rel)==expected,rel
        assert sha(package)==plan['artifact']['sha256'] and sha(plan_path)==allocation['plan_sha256']
        assert cgroup_limits()==record['cgroup_limits']
        record['quiet_runtime_qualified']=(record['callback_smt_clean'] and record['whole_trace_smt_clean'])
        record['status']='COMPLETE'
    except BaseException as exc:
        record.update(status='FAILED_OR_INTERRUPTED',exception_type=type(exc).__name__,exception=str(exc),
                      preserved_sequence_receipts=len(rows))
        raise
    finally:
        record.update(ended_utc=now(),worker_total_seconds=time.perf_counter()-WORKER_START,
                      max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        try:final_status=process_status()
        except BaseException as exc:
            final_status={'threads':0,'vm_rss_KiB':0,'vm_hwm_KiB':0,'tasks':{},'error':repr(exc)}
            record.update(status='FAILED_OR_INTERRUPTED',postguard_status_error=repr(exc))
        record['postguard_process_status']=final_status
        samples=[final_status,record.get('process_status_after_dependencies',final_status),record.get('final_process_status',final_status),*[r[k] for r in rows for k in ('status_before','status_after')]]
        record['max_observed_rss_KiB']=max([record['max_rss_KiB'],*[s['vm_rss_KiB'] for s in samples],*[s['vm_hwm_KiB'] for s in samples]])
        record['max_observed_os_threads']=max(s['threads'] for s in samples)
        atomic_json(output/'worker-result.json',record)


if __name__=='__main__':main()
