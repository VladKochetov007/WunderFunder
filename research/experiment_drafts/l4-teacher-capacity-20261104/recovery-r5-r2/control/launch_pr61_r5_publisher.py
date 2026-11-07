import os,sys,json,time,pathlib,hashlib,shutil
B=pathlib.Path('/data2/lc-epanov_s/tabdata');R=B/'runtime/l4-teacher-capacity-20261104-r1';C=R/'control';S=R/'source-observer-r5-r2';token=B/'private/tabdata-pr61-r5-final.token';ident=token.stat();token_id=(ident.st_dev,ident.st_ino);result={}
try:
 sys.path[:0]=[str(S),str(S/'src')]
 from scripts.l4_teacher_capacity_20261104 import continue_students as cs
 from wnn.multistage_search import atomic_write_json
 from wnn.multistage_search_owned_command import run_owned_command
 from wnn.multistage_search_runtime import current_process_owner
 h=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
 panelp=C/'STUDENT_CONTINUATION_PANEL_R1.json';panel,ph=cs.validate_panel(panelp);shared=json.loads(pathlib.Path(panel['paths']['shared_state']).read_text());deadline=shared['hard_deadline_epoch'];entry=S/'scripts/l4_teacher_capacity_20261104/publish_r5_results.py';bp=C/'SOURCE_BINDING_OBSERVER_R5_R2.json';binding=json.loads(bp.read_text());assert h(entry)==binding['files_sha256'][entry.relative_to(S).as_posix()]
 argv=[shutil.which('taskset'),'-c','8',panel['paths']['python'],str(entry),'--panel',str(panelp),'--r5-complete',str(R/'receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R5_COMPLETE.json'),'--token-file',str(token),'--deadline',str(deadline),'--poll-seconds','30']
 e=os.environ.copy();e.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(S)+':'+str(S/'src'),PYTHONUNBUFFERED='1')
 rp=R/'receipts/publication/PR61_R5_PUBLISHER_PROCESS_R1.json';assert not rp.exists()
 code,timed,rec=run_owned_command(argv,cwd=S,env=e,log_path=R/'logs/publication/publisher_r5.log',receipt_path=rp,owner={**panel['identity'],'continuation_panel_sha256':ph,'publisher_source_binding_sha256':h(bp),'entrypoint_path':str(entry),'entrypoint_sha256':h(entry),'cell_id':'pr61_r5_results_publisher','run_name':'pr61_r5_results_publisher','family':'publication'},stage=2,trial_number=0,mode='pr61-r5-results-publication',gpu_key='cpu-publisher',output_dir=R/'publication-r5',timeout_seconds=max(1,int(deadline-time.time())))
 result={'status':'complete' if code==0 else 'failed','exit_code':code,'timed_out':timed,'process_receipt_path':str(rp),'process_receipt_sha256':h(rp)}
except BaseException as error:
 result={'status':'failed','error_type':type(error).__name__,'error':str(error)};raise
finally:
 removed=False
 try:
  st=token.lstat()
  if (st.st_dev,st.st_ino)==token_id and not token.is_symlink():token.unlink();removed=True
 except FileNotFoundError:removed=True
 terminal=C/'PR61_R5_PUBLISHER_WRAPPER_TERMINAL_R1.json';terminal.write_text(json.dumps({'schema':'wnn.pr61-r5-publication-wrapper.v1',**result,'temporary_credential_removed':removed,'finished_epoch':time.time()},sort_keys=True,indent=2)+'\n')
