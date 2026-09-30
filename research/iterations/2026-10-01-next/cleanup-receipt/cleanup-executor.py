import os, sys, stat, json, hashlib, subprocess, datetime, pathlib, traceback, time

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path, obj):
    with open(path, 'x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
def fp(s): return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino, device=s.st_dev, mode=s.st_mode, nlink=s.st_nlink)
def run(cmd):
    p=subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    return dict(argv=cmd, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr, observed_utc=utc())
def hash_stable(path, expected=None):
    fd=os.open(path, os.O_RDONLY|os.O_NOFOLLOW)
    try:
        initial=os.fstat(fd)
        if not stat.S_ISREG(initial.st_mode): raise RuntimeError('not regular '+path)
        if expected is not None and fp(initial)!=expected: raise RuntimeError('fingerprint changed '+path)
        h=hashlib.sha256()
        while True:
            chunk=os.read(fd, 8*1024*1024)
            if not chunk: break
            h.update(chunk)
        if fp(os.fstat(fd))!=fp(initial): raise RuntimeError('changed during hash '+path)
        if fp(os.lstat(path))!=fp(initial): raise RuntimeError('path changed during hash '+path)
        return h.hexdigest(),fp(initial)
    finally: os.close(fd)
def inventory(root):
    entries=[]
    for n in sorted(os.listdir(root)):
        s=os.lstat(os.path.join(root,n)); e=dict(name=n,fingerprint=fp(s),allocated_bytes=s.st_blocks*512)
        if stat.S_ISLNK(s.st_mode): e['symlink_target']=os.readlink(os.path.join(root,n))
        entries.append(e)
    return dict(observed_utc=utc(),root=root,entries=entries,regular_files=sum(stat.S_ISREG(e['fingerprint']['mode']) for e in entries),symlinks=sum(stat.S_ISLNK(e['fingerprint']['mode']) for e in entries),regular_bytes=sum(e['fingerprint']['bytes'] for e in entries if stat.S_ISREG(e['fingerprint']['mode'])))
def space(root):
    s=os.statvfs(root)
    return dict(available_bytes=s.f_bavail*s.f_frsize,free_bytes=s.f_bfree*s.f_frsize,total_bytes=s.f_blocks*s.f_frsize)

def main(plan_path):
    raw=pathlib.Path(plan_path).read_bytes(); plan=json.loads(raw); plan_sha=hashlib.sha256(raw).hexdigest()
    expected_sha='0191250902acba03a4f28bf592a1ce91c40fd399718369ea18b00e418d0d7a14'
    assert plan_sha==expected_sha, 'frozen plan hash mismatch'
    root='/home/grinchenkoeg/WunderFunder/cache'; repo='/home/grinchenkoeg/WunderFunder'
    assert plan['root']==root
    regs=[e for e in plan['files'] if e['action']=='propose_delete']; links=[e for e in plan['files'] if e['action']=='propose_delete_symlink']
    assert len(regs)==154 and len(links)==24 and sum(e['fingerprint']['bytes'] for e in regs)==54534419712
    names=set()
    for e in regs+links:
        n=e['name']; assert n not in names; names.add(n)
        assert n==os.path.basename(n) and n.startswith('pred_') and n.endswith(('_valid.npy','_train.npy')) and '\n' not in n
        assert e['path']==os.path.join(root,n) and e['fingerprint']['nlink']==1
    regnames={e['name'] for e in regs}
    for e in links: assert e['target'] in regnames and e['target']==os.path.basename(e['target'])
    receipt='/home/grinchenkoeg/WunderFunder-cleanup-20261001-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    os.mkdir(receipt,0o700)
    pathlib.Path(receipt,'cleanup-plan.json').write_bytes(raw)
    pathlib.Path(receipt,'cleanup-executor.py').write_bytes(pathlib.Path(__file__).read_bytes())
    started=utc(); start_space=space(root); before=inventory(root); dump(receipt+'/before-inventory.json',before)
    dump(receipt+'/execution-context.json',dict(started_utc=started,plan_sha256=plan_sha,receipt_dir=receipt,method='Serial SHA256; nice=10, ionice best-effort7; open O_NOFOLLOW, dir_fd unlink; no train/predict processes launched.',limitations='lsof/fuser are point-in-time checks; container visibility may be incomplete. Concurrent noncooperating opens between check and unlink cannot be excluded. No scheduled new training/prediction is permitted by orchestration.',own_processes=run(['ps','-u',str(os.getuid()),'-o','pid,ppid,lstart,args']),space_before=start_space))
    print(json.dumps(dict(phase='receipt_created',receipt_dir=receipt,plan_sha256=plan_sha)),flush=True)
    hashes=open(receipt+'/file-hashes.jsonl','x'); events=open(receipt+'/events.jsonl','x')
    def append(f,obj): f.write(json.dumps(obj,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
    results=[]; validation=[]; lsof_result=None; fuser_initial=None; fatal=None
    cachefd=None
    try:
        cachefd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        assert os.path.realpath(root)==root, 'cache root path contains symlink'
        for rel, expected in plan['generation_script_sha256'].items():
            assert rel.startswith('scripts/') and '..' not in pathlib.PurePosixPath(rel).parts
            actual,f=hash_stable(os.path.join(repo,rel)); assert actual==expected,'generator SHA mismatch: '+rel
            validation.append(dict(kind='generator',path=rel,sha256=actual,fingerprint=f))
        for c in plan['regeneration_checkpoints']:
            rel=c['path']; assert rel.startswith('artifacts/runs/') and '..' not in pathlib.PurePosixPath(rel).parts
            actual,f=hash_stable(os.path.join(repo,rel),{k:c[k] for k in ('bytes','mtime_ns','inode','device','mode','nlink')})
            assert actual==c['sha256'],'checkpoint SHA mismatch: '+rel
            validation.append(dict(kind='checkpoint',path=rel,sha256=actual,fingerprint=f))
        dump(receipt+'/regeneration-validation.json',dict(completed_utc=utc(),passed=True,items=validation))
        paths=[e['path'] for e in regs+links]
        lsof_result=run(['lsof','-nP','-Fpn',*paths]); fuser_initial=run(['fuser','-v',*paths])
        dump(receipt+'/target-usage-before.json',dict(lsof=lsof_result,fuser=fuser_initial))
        assert lsof_result['returncode'] in (0,1),'lsof command failed'
        assert fuser_initial['returncode'] in (0,1),'fuser command failed'
        initially_open=set()
        for line in lsof_result['stdout'].splitlines():
            if line.startswith('n') and line[1:] in paths: initially_open.add(line[1:])
        if lsof_result['returncode']==0 and not initially_open: raise RuntimeError('lsof detected use without exact target mapping')
        print(json.dumps(dict(phase='validated',generators=7,checkpoints=220,initially_open=sorted(initially_open))),flush=True)
        deleted=set()
        for i,e in enumerate(regs,1):
            result=dict(name=e['name'],path=e['path'],kind='regular',expected_fingerprint=e['fingerprint'],started_utc=utc())
            try:
                if e['path'] in initially_open: raise RuntimeError('target was open in pre-batch lsof')
                s=os.stat(e['name'],dir_fd=cachefd,follow_symlinks=False)
                if not stat.S_ISREG(s.st_mode) or fp(s)!=e['fingerprint']: raise RuntimeError('lstat fingerprint changed before hash')
                usage_pre=run(['fuser','-v',e['path']])
                if usage_pre['returncode']!=1 or usage_pre['stdout'].strip() or usage_pre['stderr'].strip():
                    result['usage_before_hash']=usage_pre; raise RuntimeError('target open or fuser uncertain before hash')
                fd=os.open(e['name'],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=cachefd)
                try:
                    if fp(os.fstat(fd))!=e['fingerprint']: raise RuntimeError('fstat fingerprint changed before hash')
                    h=hashlib.sha256()
                    while True:
                        b=os.read(fd,8*1024*1024)
                        if not b: break
                        h.update(b)
                    if fp(os.fstat(fd))!=e['fingerprint']: raise RuntimeError('fstat fingerprint changed during hash')
                    digest=h.hexdigest()
                finally: os.close(fd)
                result['sha256']=digest
                append(hashes,dict(hashed_utc=utc(),plan_sha256=plan_sha,name=e['name'],path=e['path'],fingerprint=e['fingerprint'],sha256=digest,allocated_bytes=s.st_blocks*512))
                usage=run(['fuser','-v',e['path']])
                if usage['returncode']!=1 or usage['stdout'].strip() or usage['stderr'].strip():
                    result['usage_before_unlink']=usage; raise RuntimeError('target open or fuser uncertain immediately before unlink')
                s2=os.stat(e['name'],dir_fd=cachefd,follow_symlinks=False)
                if fp(s2)!=e['fingerprint'] or not stat.S_ISREG(s2.st_mode): raise RuntimeError('lstat fingerprint changed immediately before unlink')
                os.unlink(e['name'],dir_fd=cachefd)
                os.fsync(cachefd)
                result.update(status='deleted',logical_bytes=s.st_size,allocated_bytes=s.st_blocks*512,completed_utc=utc()); deleted.add(e['name'])
            except Exception as ex:
                result.update(status='skipped',reason=str(ex),completed_utc=utc())
            append(events,result); results.append(result)
            if i%10==0 or i==len(regs): print(json.dumps(dict(phase='regular_progress',processed=i,total=len(regs),deleted=sum(r['status']=='deleted' for r in results),deleted_logical_bytes=sum(r.get('logical_bytes',0) for r in results),skipped=sum(r['status']=='skipped' for r in results))),flush=True)
        for e in links:
            result=dict(name=e['name'],path=e['path'],kind='symlink',expected_fingerprint=e['fingerprint'],target=e['target'],started_utc=utc())
            try:
                if e['target'] not in deleted: raise RuntimeError('target regular file was not deleted; retain alias')
                s=os.stat(e['name'],dir_fd=cachefd,follow_symlinks=False)
                if not stat.S_ISLNK(s.st_mode) or fp(s)!=e['fingerprint']: raise RuntimeError('symlink lstat fingerprint changed')
                if os.readlink(e['name'],dir_fd=cachefd)!=e['target']: raise RuntimeError('symlink target changed')
                append(hashes,dict(hashed_utc=utc(),plan_sha256=plan_sha,name=e['name'],path=e['path'],fingerprint=e['fingerprint'],symlink_target=e['target'],target_string_sha256=hashlib.sha256(os.fsencode(e['target'])).hexdigest(),no_dereference=True))
                if fp(os.stat(e['name'],dir_fd=cachefd,follow_symlinks=False))!=e['fingerprint'] or os.readlink(e['name'],dir_fd=cachefd)!=e['target']: raise RuntimeError('symlink changed immediately before unlink')
                os.unlink(e['name'],dir_fd=cachefd); os.fsync(cachefd)
                result.update(status='deleted',logical_bytes=s.st_size,allocated_bytes=s.st_blocks*512,completed_utc=utc())
            except Exception as ex: result.update(status='skipped',reason=str(ex),completed_utc=utc())
            append(events,result); results.append(result)
    except Exception as ex:
        fatal=dict(error=str(ex),traceback=traceback.format_exc(),utc=utc())
        print(json.dumps(dict(phase='fatal',error=str(ex))),flush=True)
    finally:
        if cachefd is not None: os.close(cachefd)
        hashes.close(); events.close()
        after=inventory(root); dump(receipt+'/after-inventory.json',after); end_space=space(root)
        aftermap={e['name']:e for e in after['entries']}
        retain_check=[dict(name=e['name'],unchanged=(e['name'] in aftermap and aftermap[e['name']]['fingerprint']==e['fingerprint'])) for e in plan['files'] if e['action']=='retain']
        summary=dict(status='complete' if fatal is None else 'aborted',started_utc=started,completed_utc=utc(),plan_sha256=plan_sha,receipt_dir=receipt,requested_regular=154,requested_symlinks=24,deleted_regular=sum(r['kind']=='regular' and r['status']=='deleted' for r in results),deleted_symlinks=sum(r['kind']=='symlink' and r['status']=='deleted' for r in results),deleted_regular_logical_bytes=sum(r.get('logical_bytes',0) for r in results if r['kind']=='regular'),deleted_allocated_bytes=sum(r.get('allocated_bytes',0) for r in results),skipped=[r for r in results if r['status']=='skipped'],unprocessed_names=sorted(names-{r['name'] for r in results}),retained_all_unchanged=all(x['unchanged'] for x in retain_check),retained_checks=retain_check,regeneration_validated_count=len(validation),space_before=start_space,space_after=end_space,filesystem_available_byte_delta=end_space['available_bytes']-start_space['available_bytes'],before_inventory_summary={k:before[k] for k in ('regular_files','symlinks','regular_bytes')},after_inventory_summary={k:after[k] for k in ('regular_files','symlinks','regular_bytes')},fatal=fatal,receipt_note='Append-only hash rows fsynced before unlink; final receipt files chmod0444 and directory0555. Regular file digests correspond to the exact frozen lstat fingerprint. Deletion events are separately append-only. No retained file, checkpoint, log, dataset, or source file was modified.')
        dump(receipt+'/cleanup-execution.json',summary)
        sha_lines=[]
        for p in sorted(pathlib.Path(receipt).iterdir()):
            if p.is_file(): sha_lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
        pathlib.Path(receipt,'SHA256SUMS').write_text(''.join(sha_lines))
        for p in pathlib.Path(receipt).iterdir(): os.chmod(p,0o444)
        os.chmod(receipt,0o555)
        print(json.dumps({k:summary[k] for k in ('status','receipt_dir','deleted_regular','deleted_symlinks','deleted_regular_logical_bytes','deleted_allocated_bytes','skipped','retained_all_unchanged','after_inventory_summary','fatal')}),flush=True)
    return 0 if fatal is None else 1
if __name__=='__main__': sys.exit(main(sys.argv[1]))
