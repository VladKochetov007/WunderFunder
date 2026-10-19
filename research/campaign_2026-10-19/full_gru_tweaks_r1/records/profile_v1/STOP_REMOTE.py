import subprocess,json,hashlib,pathlib,datetime
u='computer-vision-homework-fg19-propensity-profile-v1.service'
p=subprocess.run(['systemctl','--user','show','--all',u,'-p','InvocationID','-p','ExecMainCode','-p','ExecMainStatus','-p','SubState'],capture_output=True,text=True,timeout=10);assert p.returncode==0
v=dict(x.split('=',1) for x in p.stdout.splitlines());assert v['InvocationID']=='0698555969d348eebb9b51d7dc39f545' and v['ExecMainCode']=='1' and v['ExecMainStatus']=='0' and v['SubState']=='exited'
f=pathlib.Path('/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_propensity_profile_v1/PROFILE_RESULT.json');assert hashlib.sha256(f.read_bytes()).hexdigest()=='cfafd5b90b480e22513b6b13924e2aadd3921154362bfd38620b04bc2ead43da'
s=subprocess.run(['systemctl','--user','stop',u],capture_output=True,text=True,timeout=20);assert s.returncode==0
z=subprocess.run(['systemctl','--user','show','--all',u,'-p','LoadState','-p','ActiveState','-p','SubState'],capture_output=True,text=True,timeout=10)
print(json.dumps({'created_utc':datetime.datetime.now(datetime.UTC).isoformat(),'unit':u,'expected_invocation':v['InvocationID'],'pre_stop':p.stdout,'stop_returncode':s.returncode,'stop_stdout':s.stdout,'stop_stderr':s.stderr,'post_show_returncode':z.returncode,'post_show_stdout':z.stdout,'post_show_stderr':z.stderr,'raw_receipt_validated_before_stop':True}))
