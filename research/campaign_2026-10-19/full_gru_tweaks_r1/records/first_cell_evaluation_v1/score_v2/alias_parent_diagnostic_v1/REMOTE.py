import json,stat
from pathlib import Path
from datetime import datetime,timezone
B=Path("/home/grinchenkoeg/computer-vision-homework-20261018/data")
def observe(p):
 l=p.lstat();s=p.stat();return {"path":str(p),"resolved":str(p.resolve(strict=True)),"islink":stat.S_ISLNK(l.st_mode),"mode":s.st_mode,"bytes":s.st_size,"device":s.st_dev,"inode":s.st_ino,"mtime_ns":s.st_mtime_ns}
parents={k:observe(B/k)for k in ("cache","datasets")}
leaves={k:observe(B/k)for k in ("cache/chain_folds.npz","cache/chains.npz","cache/train_targets.npz","cache/valid_targets.npz","datasets/train.parquet","datasets/valid.parquet")}
parents_after={k:observe(B/k)for k in ("cache","datasets")}
if parents!=parents_after:raise RuntimeError("parent metadata changed within bounded observation")
print(json.dumps({"schema":"fg19.first-cell-score-registered-parent-stat-observation.v1","captured_utc":datetime.now(timezone.utc).isoformat(),"parents":parents,"parents_after":parents_after,"leaves":leaves,"read_only":True,"unit_queries":0,"payload_bytes_read_or_hashed":0}))
