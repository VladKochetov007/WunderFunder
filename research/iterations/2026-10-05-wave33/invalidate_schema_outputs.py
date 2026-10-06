from pathlib import Path
import json,datetime
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave33';moved=[]
for p in list((R/'cache').glob('pred_w33native_*_train.npy'))+[D/'native-quality-report.json',D/'native-moments-train.npz',D/'native-student-parity.json']:
 if p.exists():
  dst=Path(str(p)+'.invalid-schema');assert not dst.exists();p.rename(dst);moved.append([str(p),str(dst)])
for p in (R/'cache').glob('pred_w36native_comp_*_*.npy.partial'):
 dst=Path(str(p)+'.cancelled-schema-audit');assert not dst.exists();p.rename(dst);moved.append([str(p),str(dst)])
r={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cause':'train has3 metadata columns; valid has4. Native train quality harness used positional[4:116], missing i0_p0 and includingtargett0. Invalidtrainreport quarantined. Valid nativecaches andallGPUquality remain correctlyloaded. Error is in offlineparquet reader, notdeployed112-feature Predictor API.','corrective_action':'Explicitcanonical feature names; bothsplit byteparity withSequenceFile plusbothsplit Torch-vs-native modelparity; trainreplayrecomputed.','moved':moved}
(D/'schema-incident.json').write_text(json.dumps(r,indent=2));print(json.dumps({'preserved_files':len(moved)}),flush=True)
