import os,sys,json,time,hashlib,shutil,pathlib,traceback
B=pathlib.Path('/data2/lc-epanov_s/tabdata');R=B/'runtime/l4-teacher-capacity-20261104-r1';C=R/'control';S=R/'source-delivery-r5';N=B/'runtime/l4-objective-20261104-r1'
sys.path[:0]=[str(S),str(S/'src')]
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search import atomic_write_json
from wnn.multistage_search_runtime import current_process_owner
from scripts.l4_teacher_capacity_20261104 import continue_students_recovery_r5 as r5
h=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
panelp=C/'STUDENT_CONTINUATION_PANEL_R1.json';adp=C/'STUDENT_RECOVERY_SOURCE_ADMISSION_R5.json';amp=C/'ROOT_STORAGE_AMENDMENT_R1.json';receipt=R/'receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R5_CONTROLLER.json';entry=S/'scripts/l4_teacher_capacity_20261104/continue_students_recovery_r5.py'
assert not receipt.exists()
panel,ph=r5.cs.validate_panel(panelp);ad=json.loads(adp.read_text());am=json.loads(amp.read_text());binding=json.loads((C/'SOURCE_BINDING_RECOVERY_R5.json').read_text())
assert h(C/'SOURCE_BINDING_RECOVERY_R5.json')==ad['source_binding_sha256'] and h(entry)==binding['files_sha256'][entry.relative_to(S).as_posix()]
assert json.loads((C/'RECEIVING_PREFLIGHT_R5.json').read_text())['status']=='passed'
argv=[shutil.which('taskset'),'-c',','.join(map(str,range(64))),panel['paths']['python'],str(entry),'--controller-mode','continue','--panel',str(panelp),'--recovery-admission',str(adp),'--storage-amendment',str(amp),'--controller-receipt',str(receipt),'--gpu2-release-receipt',str(N/'control/GPU2_RELEASE_R1.json')]
owner={**panel['identity'],'continuation_panel_sha256':ph,'recovery_source_binding_sha256':ad['source_binding_sha256'],'recovery_admission_receipt_path':str(adp),'recovery_admission_receipt_sha256':h(adp),'storage_amendment_receipt_path':str(amp),'storage_amendment_receipt_sha256':h(amp),'r4_stop_receipt_path':ad['r4_stop_receipt_path'],'r4_stop_receipt_sha256':ad['r4_stop_receipt_sha256'],'recovery_entrypoint_path':str(entry),'recovery_entrypoint_sha256':h(entry),'cell_id':r5.CONTROLLER_ID,'run_name':r5.CONTROLLER_ID,'family':'continuation','command_sha256':hashlib.sha256(json.dumps(argv,separators=(',',':')).encode()).hexdigest()}
e=os.environ.copy();e.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(S)+':'+str(S/'src'))
finish=C/'STUDENT_RECOVERY_R5_LAUNCHER_TERMINAL.json'
try:
 code,timed,record=run_owned_command(argv,cwd=S,env=e,log_path=R/'logs/continuation/controller_recovery_r5.log',receipt_path=receipt,owner=owner,stage=2,trial_number=0,mode='continuation-r5',gpu_key='cpu-controller',output_dir=R,timeout_seconds=r5.cs._remaining(r5.cs._deadline(panel)))
 atomic_write_json(finish,{'schema':'wnn.r5-launcher-terminal.v1','status':'complete' if code==0 else 'failed','exit_code':code,'timed_out':timed,'controller_receipt_path':str(receipt),'controller_receipt_sha256':h(receipt),'wrapper':current_process_owner(),'finished_epoch':time.time()},replace=False)
except BaseException as error:
 atomic_write_json(finish,{'schema':'wnn.r5-launcher-terminal.v1','status':'failed','error_type':type(error).__name__,'error':str(error),'finished_epoch':time.time()},replace=False);raise
