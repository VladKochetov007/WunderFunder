from pathlib import Path
import json
D=Path.home()/'WunderFunder-creative/research/iterations/2026-10-05-wave45'
full=json.loads((D/'full-report.json').read_text())['results']['valid_clean']
comp=json.loads((D/'comp-report.json').read_text())['results']
res={}
for arm in ['short_pair', 'four_mean']:
 deltas=[full['scores'][arm]-full['scores']['old_pair'],comp['valid_clean']['scores'][arm]-comp['valid_clean']['scores']['old_pair'],comp['train_clean']['scores'][arm]-comp['train_clean']['scores']['old_pair']]
 res[arm]={'deltas_full_valid_comp_valid_train':deltas,'pass':deltas[0]>=.0003 and deltas[1]>=.0003 and deltas[2]>=0}
report={'arms':res,'proceed_native':any(v['pass'] for v in res.values()),'scope':'Preregistered GPU screen for further native validation only; no final promotion. At least +.0003 on each true-mask panel and nonnegative train diagnostic. Fixed models/weights, reused panels.'}
(D/'native-gate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
