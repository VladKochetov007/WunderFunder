import json,subprocess,os
from pathlib import Path
b=Path('/home/grinchenkoeg/computer-vision-homework-20261018');reg=json.loads((b/'support-source/terminal-v1/PORTABLE_SOURCES.json').read_bytes())
rows=[]
for rel,entry in reg['sources'].items():
 p=b/'data'/rel;s=p.stat()
 rows.append(dict(path=str(p),resolved=str(p.resolve()),islink=p.is_symlink(),device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,bytes=s.st_size,registered=entry))
u='computer-vision-homework-fg19-propensity-f0-e4-v2.service'
print(json.dumps(dict(sources=rows,unit=subprocess.run(['systemctl','--user','show','--all',u,'-p','LoadState','-p','ActiveState','-p','SubState'],capture_output=True,text=True).stdout,output_exists=(b/'research_tmp/fg19_propensity_clean_f0_admission_v2').exists(),run_exists=(b/'data/artifacts/runs/fg19_propensity_clean_f0_seed0_e4').exists())))
