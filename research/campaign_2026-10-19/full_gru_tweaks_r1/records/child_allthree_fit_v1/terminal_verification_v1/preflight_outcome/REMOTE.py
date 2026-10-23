from pathlib import Path
import datetime,json,subprocess,stat
B=Path('/home/grinchenkoeg/computer-vision-homework-20261018')
K=('LoadState','ActiveState','SubState','MainPID','InvocationID','Result','ExecMainCode','ExecMainStatus','ControlGroup')
def show(unit):
 r=subprocess.run(['systemctl','--user','show','--all',unit,*sum((['-p',k] for k in K),[])],capture_output=True,timeout=15)
 if len(r.stdout)+len(r.stderr)>131072:raise RuntimeError('bounded unit query output exceeded')
 return {'returncode':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode(),'properties':dict(x.split('=',1) for x in r.stdout.decode().splitlines() if '=' in x)}
def info(p):
 try:s=p.lstat()
 except FileNotFoundError:return {'exists':False}
 return {'exists':True,'symlink':stat.S_ISLNK(s.st_mode),'mode':s.st_mode,'bytes':s.st_size,'device':s.st_dev,'inode':s.st_ino}
F=B/'research_tmp/fg19_child_allthree_fit_v1';D=B/'research_tmp/fg19_child_allthree_verify_v1'
x={'schema':'fg19.child-allthree-verifier-preflight-outcome-capture.v1','captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'support_unit':show('computer-vision-homework-fg19-child-allthree-verify-v1.service'),'native_unit':show('computer-vision-homework-fg19-child-allthree-fit-v1.service'),'support_directory_stat_only':info(D),'native_receipt_stat_only':{n:info(F/n) for n in ('ADMISSION.json','LAUNCH.json','LAUNCH_ACCEPTANCE.json','ALLTHREE_LAUNCH_STDOUT.raw.json','TRANSPORT.json')},'scope':'One readonly exact-outcome query after failed preflight; no unit action, mutation, payload read/hash, source/checker execution or admission.'}
print(json.dumps(x,sort_keys=True))
