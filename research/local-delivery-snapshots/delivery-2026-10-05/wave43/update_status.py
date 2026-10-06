from pathlib import Path
import json,datetime,subprocess,shutil
r=Path('delivery-2026-10-05');p=r/'AUTONOMOUS_STATUS.json';a=json.loads(p.read_text());a['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
a['baseline_recheck']['finding']='Corrected baseline FT still fails GPU train diagnostic. Native int8 blend after wave42 named-feature repair: valid delta +0.00000473, train diagnostic +0.00000646 versus oldhalf, both conditional intervals include zero. Previous train delta about -0.000020 was invalidated.'
a['native_schema_incident']={'detected_utc':'2026-10-05T22:46:00Z','cause':'Positional [4:116] selection only correct for valid, not train (train has 3 metadata columns). Wrong train replay dropped i0_p0 and included t0.','affected':'Wave33/36 CPU train replay, historical wave13 precision train replay. Deployed Predictor API and GPU training/teachers unaffected.','repair':'Explicit 112 feature names; byte equality against independent SequenceFile and Torch/native parity on both schemas; native/GPU score-gap guards for new students. Invalid outputs preserved; affected train panels recomputed.','evidence':['wave33/schema-incident.json','wave33/schema-repair-complete.json','wave36/native-comp-report.json','wave42/scope.json','wave42/input-model-parity.json','wave42/corrected-precision-report.json'],'status':'repaired_and_recomputed'}
a['latest_completed']={}
for w,n in [(32,'wave-report'),(33,'native-quality-report'),(35,'native-quality-report'),(36,'native-full-report'),(41,'full-report')]:
 report=json.loads((r/f'wave{w}/{n}.json').read_text());a['latest_completed'][str(w)]={'evidence':f'wave{w}/{n}.json','results':report['results'],'decision':'not_promoted'}
a['latest_completed']['36']['composite_native']=json.loads((r/'wave36/native-comp-report.json').read_text())['results']
a['latest_completed']['41']['composite_gpu']=json.loads((r/'wave41/comp-report.json').read_text())['results']
a['latest_completed']['41']['decision']='Rejected: head-only and subsequent full FT lose both panels and seeds; no native/final fit.'
a['original_stage_headwidth']['v1_native_delta']=8.866236797855365e-06
a['class_head_replication']['decision']='Rejected after GPU and repaired native composite replication; exact-shuffle control complete, no final fit.'
a['active_experiments']={str(w):json.loads((r/f'wave{w}/protocol.json').read_text()) for w in [38,39,40,43]}
u=subprocess.check_output(['ssh','beleriand','systemctl --user list-units --state=active --no-legend --no-pager "wunder-wave*"'],text=True)
a['active_units']=[l.split()[0].removesuffix('.service') for l in u.splitlines() if l.strip()]
p.write_text(json.dumps(a,indent=2,ensure_ascii=False))
p=r/'wave31/native-quality-report.json';x=json.loads(p.read_text());x['initialization_audit']=json.loads((r/'wave31/initialization-context-audit.json').read_text());x['scope']+=' V1 initializer omitted preceding PropensityNet RNG consumption: auxiliary family-head initialization differs from control. This is width plus auxiliary-init treatment. Corrected repeat is wave39.';p.write_text(json.dumps(x,indent=2))
p=r/'wave13/precision-report.json'
if p.exists() and not p.with_name(p.name+'.invalid-schema').exists():p.rename(p.with_name(p.name+'.invalid-schema'))
shutil.copy(r/'wave42/corrected-precision-report.json',p)
ep=r/'wave23/ledger_events.json';e=json.loads(ep.read_text())
for w in [41,43]:e['hypotheses'].append({'id':f'H-20261005-W{w}','date':'2026-10-05','kind':f'Wave{w} preregistered protocol','protocol':json.loads((r/f'wave{w}/protocol.json').read_text())})
for w in [32,33,35,36,41]:e['experiments'].append({'id':f'E-20261005-W{w}-CLOSURE','date':'2026-10-05','kind':f'Wave{w} validated closure','result':a['latest_completed'][str(w)]})
e['experiments'].append({'id':'E-20261005-W42-SCHEMA-REPAIR','date':'2026-10-05','kind':'Native train feature schema correction and recomputation','result':a['native_schema_incident'],'baseline_precision':json.loads((r/'wave42/corrected-precision-report.json').read_text())['results']})
for k in e:e[k]=list({v['id']:v for v in e[k]}.values())
ep.write_text(json.dumps(e,indent=2));print('Status and ledger updated',len(a['active_units']),'active units')
