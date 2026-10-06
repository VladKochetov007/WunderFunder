from pathlib import Path
import json
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave13';P=R/'research/iterations/2026-10-05-wave42';moved=[]
for p in list((R/'cache').glob('pred_w13native_*_train.npy'))+[D/'precision-report.json',D/'precision-moments-train.npz',D/'native-replay-train.json']:
 if p.exists():
  dst=Path(str(p)+'.invalid-schema');assert not dst.exists();p.rename(dst);moved.append([str(p),str(dst)])
(P/'preserved-invalid-outputs.json').write_text(json.dumps(moved,indent=2))
