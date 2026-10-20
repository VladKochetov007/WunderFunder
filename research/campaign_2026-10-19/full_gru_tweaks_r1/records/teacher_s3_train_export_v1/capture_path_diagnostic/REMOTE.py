from pathlib import Path
import json,datetime
b=Path('/home/grinchenkoeg/computer-vision-homework-20261018');d=b/'research_tmp/fg19_teacher_s3_train_export_v1';p=b/'research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1'
paths=[d/'ADMISSION.json',d/'LAUNCH.json',d/'unit.log',p/'pred_fg19_t3_h384x2_e3_s3_v1_train.json',b/'splits/fold0.json',b/'data/artifacts/runs/fg19_t3_h384x2_e3_s3_v1/provenance.json']
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':[{'path':str(x),'exists':x.exists(),'is_file':x.is_file(),'symlink':x.is_symlink(),'bytes':x.stat().st_size if x.exists() else None,'resolved':str(x.resolve())} for x in paths]}))
