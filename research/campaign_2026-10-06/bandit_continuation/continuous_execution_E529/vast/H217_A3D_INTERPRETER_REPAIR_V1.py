import os,pathlib,hashlib,json,subprocess,datetime
root=pathlib.Path(ROOT_VALUE)
work=pathlib.Path(WORK_VALUE)/'work'
for entry in root.iterdir():
 if entry.name not in {'worker.log','dependencies.log','.worker'}:
  raise RuntimeError('unexpected prior output; refuse repair: '+entry.name)
if hashlib.sha256((work/'run.sh').read_bytes()).hexdigest()!='5f47096901aba9a7bed33f512828b60c5fe583a6a097a8f529f8093f31c16074':
 raise RuntimeError('frozen run.sh hash changed')
if not pathlib.Path('/venv/main/bin/python').is_file():
 raise RuntimeError('image interpreter missing')
# Refuse a second live scientific worker in this exclusively owned container.
for proc in pathlib.Path('/proc').iterdir():
 if proc.name.isdigit() and int(proc.name)!=os.getpid():
  try: args=(proc/'cmdline').read_bytes().split(b'\0')
  except (OSError,PermissionError): continue
  if any(x in [b'./run.sh',b'worker.py',b'scripts/train_rnn.py',b'scripts/predict_run.py'] for x in args):
   raise RuntimeError('existing worker detected; repair refuses duplicate launch')
repair=root/'interpreter_retry1_20261008'
repair.mkdir(exist_ok=False)
for name in ['worker.log','dependencies.log']:
 p=root/name
 if p.exists(): p.rename(repair/name)
env=os.environ.copy();env.update({'WNN_ATTEMPT_ID': 'e529-vast-h217-pair-v2-20261008-a3d', 'WNN_PACKET_SHA256': '401e3a7e5315ad851f3b7aae3135fcf81e9dda71420d1fd26a6b595f197745b6', 'WNN_OUTPUT_DIR': '/workspace/results', 'WNN_INPUT_DIR': '/workspace/results/.worker/wnn-e529-h217-pair-20261008-a3d/inputs', 'WNN_EXPECTED_ARTIFACTS': '["artifact_bundle.tar","result.json","worker.log"]', 'WNN_INPUT_MANIFEST': '[{"path":"cache/chain_folds.npz","sha256":"a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f","size_bytes":550072},{"path":"cache/chains.npz","sha256":"39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766","size_bytes":650142},{"path":"cache/train_targets.npz","sha256":"ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9","size_bytes":1909335790},{"path":"cache/valid_targets.npz","sha256":"dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955","size_bytes":337186198},{"path":"datasets/train.parquet","sha256":"4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81","size_bytes":29240219011},{"path":"datasets/valid.parquet","sha256":"2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782","size_bytes":5615509128}]', 'WNN_MAX_RUNTIME_SECONDS': '17666', 'WNN_EXECUTION_RETRY_ID': 'h217-a3d-interpreter-retry1-20261008'})
hard=datetime.datetime.fromisoformat('2026-10-08T05:53:17.694698+00:00')
remaining=int((hard-datetime.datetime.now(datetime.timezone.utc)).total_seconds())
env['WNN_MAX_RUNTIME_SECONDS']=str(remaining-750-450)
if int(env['WNN_MAX_RUNTIME_SECONDS'])<=90:
 raise RuntimeError('insufficient repair runway')
env['PATH']='/venv/main/bin:'+env.get('PATH','/usr/bin:/bin')
with (root/'launcher_retry1.log').open('xb') as out:
 child=subprocess.Popen(['bash','./run.sh'],cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=out,start_new_session=True)
receipt={'execution_retry_id':env['WNN_EXECUTION_RETRY_ID'],'attempt':env['WNN_ATTEMPT_ID'],'pid':child.pid,'launched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repair':'prepend existing image /venv/main/bin to worker PATH','run_sha256':'5f47096901aba9a7bed33f512828b60c5fe583a6a097a8f529f8093f31c16074','packet_sha256':env['WNN_PACKET_SHA256'],'worker_runtime_seconds':int(env['WNN_MAX_RUNTIME_SECONDS']),'source_or_recipe_changed':False,'prior_logs_directory':str(repair)}
(root/'interpreter_retry1_launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
