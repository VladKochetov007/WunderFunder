from pathlib import Path
import json
r=Path('delivery-2026-10-05')
for wave,seeds,newarm in [(43,(13,14),'new_pair'),(45,(3,4),'short_pair')]:
 d=r/f'wave{wave}'
 s=(r/'wave36/export_oof.py').read_text().replace('wave36',f'wave{wave}').replace('w36_',f'w{wave}_').replace('(3,4)',repr(seeds)).replace(";assert ck['args']['family_class_mode']=='off'",'').replace("'unused_class_weights_omitted':True","'unused_family_weights_omitted':True")
 (d/'export_oof.py').write_text(s)
 s=(r/'wave36/export_train_probes.py').read_text().replace('wave36',f'wave{wave}').replace('w36_',f'w{wave}_').replace('(3,4)',repr(seeds));(d/'export_train_probes.py').write_text(s)
 s=(r/'wave36/native_replay.py').read_text().replace('wave36',f'wave{wave}').replace('w36native_',f'w{wave}native_').replace('(3,4)',repr(seeds)).replace('wave26/native-class','wave16/native-sparse').replace(';assert wnn_cpp.CLASS_HEAD_SUPPORTED','')
 (d/'native_replay.py').write_text(s)
 for kind in ['full','comp']:
  s=(d/f'evaluate_{kind}.py').read_text();s=s.replace(f"'{kind}-report",f"'native-{kind}-report").replace(f"'{kind}-moments",f"'native-{kind}-moments")
  a=s.index('specs=');b=s.index('\ndef wp',a);ns={};exec(s[a:b],{},ns);spec=ns['specs'];pairs=ns['comparisons']
  for name,sp in list(spec.items()):
   spec[name]={k.replace(f'w{wave}_{kind}',f'w{wave}native_{kind}').replace('w12compress256_distilled','w18native_distilled').replace('w18compcompress_distilled','w33native_old'):v for k,v in sp.items()}
  for name in [newarm,'four_mean']:
   spec[name+'_gpu']=ns['specs'][name] if False else {k.replace(f'w{wave}native_{kind}',f'w{wave}_{kind}').replace('w18native_distilled','w12compress256_distilled').replace('w33native_old','w18compcompress_distilled'):v for k,v in spec[name].items()}
   pairs.append((name,name+'_gpu'))
  s=s[:a]+f'specs={spec!r}\ncomparisons={pairs!r}\n'+s[b:]
  marker="        report['results'][split+'_clean']=result"
  s=s.replace(marker,"        for name in ["+repr(newarm)+",'four_mean']:\n            assert abs(result['scores'][name]-result['scores'][name+'_gpu'])<.005, (split,name,'native/GPU mismatch')\n"+marker)
  (d/f'evaluate_native_{kind}.py').write_text(s)
 arms=['four_mean'] if wave==43 else ['short_pair','four_mean']
 (d/'native_gate.py').write_text(f'''from pathlib import Path
import json
D=Path.home()/'WunderFunder-creative/research/iterations/2026-10-05-wave{wave}'
full=json.loads((D/'full-report.json').read_text())['results']['valid_clean']
comp=json.loads((D/'comp-report.json').read_text())['results']
res={{}}
for arm in {arms!r}:
 deltas=[full['scores'][arm]-full['scores']['old_pair'],comp['valid_clean']['scores'][arm]-comp['valid_clean']['scores']['old_pair'],comp['train_clean']['scores'][arm]-comp['train_clean']['scores']['old_pair']]
 res[arm]={{'deltas_full_valid_comp_valid_train':deltas,'pass':deltas[0]>=.0003 and deltas[1]>=.0003 and deltas[2]>=0}}
report={{'arms':res,'proceed_native':any(v['pass'] for v in res.values()),'scope':'Preregistered GPU screen for further native validation only; no final promotion. At least +.0003 on each true-mask panel and nonnegative train diagnostic. Fixed models/weights, reused panels.'}}
(D/'native-gate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
''')
 (d/'native_quality.sh').write_text(f'''#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave{wave}"
for kind in full comp; do
 while systemctl --user is-active --quiet wunder-wave{wave}-evaluate-$kind;do sleep 10;done
 test -f "$D/$kind-report.json"
done
.venv/bin/python "$D/native_gate.py" > "$D/native-gate.log" 2>&1
if ! .venv/bin/python -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["proceed_native"])' "$D/native-gate.json";then exit 0;fi
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src" WNN_CPU_PCT=6 WNN_MEM_MAX=24G
for kind in full comp;do
 scripts/limited .venv/bin/python "$D/export_oof.py" "$kind" > "$D/export-$kind.log" 2>&1
 scripts/limited .venv/bin/python "$D/export_train_probes.py" "$kind" > "$D/train-probes-$kind.log" 2>&1
 scripts/limited "$HOME/venv311/bin/python" "$D/native_replay.py" "$kind" > "$D/native-replay-$kind.log" 2>&1
 scripts/limited .venv/bin/python "$D/evaluate_native_$kind.py" > "$D/evaluate-native-$kind.log" 2>&1
done
''')
 p=d/'protocol.json';j=json.loads(p.read_text());j['native_screen_gate']={'arms':arms,'minimum_delta_each_true_mask_panel':.0003,'minimum_delta_train_diagnostic':0,'scope':'Gate to spend native validation, not automatic final promotion; set before results.'};p.write_text(json.dumps(j,indent=2))
