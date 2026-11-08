import os,sys,json,time,pathlib,hashlib,shutil
B=pathlib.Path('/data2/lc-epanov_s/tabdata');R=B/'runtime/l4-teacher-capacity-20261104-r1';C=R/'control';S=R/'source-observer-r5-r2';sys.path[:0]=[str(S),str(S/'src')]
from scripts.l4_teacher_capacity_20261104 import continue_students as cs
from wnn.multistage_search import atomic_write_json
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import current_process_owner
h=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
panelp=C/'STUDENT_CONTINUATION_PANEL_R1.json';panel,ph=cs.validate_panel(panelp);deadline=cs._deadline(panel);statep=C/'CACHE_REVIEW_WATCHER_R6_R2_STATE.json';summary=R/'receipts/stage2/causal_transformer_metadata_recovery_r5.json';worker=R/'receipts/stage2/causal_transformer_stage2_compact_finalize_r5.json';failed=R/'receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R5_FAILED.json'
entry=S/'scripts/l4_teacher_capacity_20261104/grade_stage2_cache_r6.py';bp=C/'SOURCE_BINDING_OBSERVER_R5_R2.json';binding=json.loads(bp.read_text());assert h(entry)==binding['files_sha256'][entry.relative_to(S).as_posix()]
adp=C/'STAGE2_CACHE_REVIEW_SOURCE_ADMISSION_R6_R2.json'
def state(status,**values):atomic_write_json(statep,{'schema':'wnn.cache-review-watcher-r6-r2.v1','status':status,'updated_epoch':time.time(),'owner':current_process_owner(),'source_binding_sha256':h(bp),'deadline_epoch':deadline,**values})
state('waiting_for_actual_tf_metadata')
try:
 while not summary.exists():
  if failed.exists():state('stopped_after_controller_failure');sys.exit(1)
  cs._remaining(deadline);time.sleep(10)
 wr=json.loads(worker.read_text());assert wr['status']=='completed' and wr['exit_code']==0 and wr['timed_out'] is False
 argv=[shutil.which('taskset'),'-c',','.join(map(str,range(40,48))),panel['paths']['python'],str(entry),'--continuation-panel',str(panelp),'--family','causal_transformer','--recovery-admission',str(C/'STUDENT_RECOVERY_SOURCE_ADMISSION_R5.json'),'--storage-amendment',str(C/'ROOT_STORAGE_AMENDMENT_R1.json'),'--grader-source-admission',str(adp)]
 e=os.environ.copy();e.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONPATH=str(S)+':'+str(S/'src'),PYTHONUNBUFFERED='1')
 rp=R/'receipts/stage2/causal_transformer_cache_review_r6_r2.json';assert not rp.exists();state('grading_actual_cache',summary_sha256=h(summary))
 code,timed,rec=run_owned_command(argv,cwd=S,env=e,log_path=R/'logs/stage2/cache_review_r6_r2.log',receipt_path=rp,owner={**panel['identity'],'continuation_panel_sha256':ph,'grader_source_binding_sha256':h(bp),'grader_source_admission_path':str(adp),'grader_source_admission_sha256':h(adp),'cell_id':'causal_transformer_cache_review_r6_r2','run_name':'causal_transformer_cache_review_r6_r2','family':'causal_transformer'},stage=2,trial_number=0,mode='independent-cache-review-r6-r2',gpu_key='cpu-reviewer',output_dir=pathlib.Path(panel['families']['causal_transformer']['target_cache']),timeout_seconds=min(1800,cs._remaining(deadline)))
 reviewp=pathlib.Path(panel['families']['causal_transformer']['cache_admission_receipt'])
 if code!=0 or timed:
  if not reviewp.exists():atomic_write_json(reviewp,{'schema':'wnn.stage2-cache-review-rejected.v1','status':'rejected_checker_failure','family':'causal_transformer','continuation_panel_sha256':ph,'checker_receipt_path':str(rp),'checker_receipt_sha256':h(rp),'exit_code':code,'timed_out':timed,'created_epoch':time.time()},replace=False)
  state('failed',checker_receipt_sha256=h(rp));sys.exit(1)
 review=json.loads(reviewp.read_text());assert str(review['status']).startswith('approved');state('complete',cache_admission_receipt_path=str(reviewp),cache_admission_receipt_sha256=h(reviewp),checker_receipt_sha256=h(rp))
except Exception as error:state('failed',error_type=type(error).__name__,error=str(error));raise
