from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET,ast
root=Path('/home/vlad/development/wunder/worktrees/sol-diversity-delivery-20261009')
base=root/'research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/tpu_gpu_benchmark'
files=[base/'jax_step.py',base/'tpu_worker.py',base/'run_device_parity.py',base/'tests/test_gru256_feature_adapter_parity.py']
for p in files: ast.parse(p.read_text())
xml=ET.parse(base/'CPU_LIFECYCLE_TESTS_R3.xml').getroot()
suites=list(xml.iter('testsuite')); total=sum(int(s.attrib['tests']) for s in suites)
assert total==8 and all(int(s.attrib.get(k,0))==0 for s in suites for k in ['errors','failures','skipped'])
receipt={'status':'cpu_lifecycle_pass_device_gate_pending','tests_passed':total,'tests_skipped':0,'junit_sha256':hashlib.sha256((base/'CPU_LIFECYCLE_TESTS_R3.xml').read_bytes()).hexdigest(),'elapsed_test_seconds':sum(float(s.attrib['time']) for s in suites),'jax_version':'0.6.2','jaxlib_version':'0.6.2','backend':'cpu','temporary_dependency_target':'/tmp/wnn-sol-jax-deps','temporary_installed_bytes':510983134,'main_venv_mutated':False,'limits':{'cpu_cores':1,'memory_bytes':2000000000,'runtime_seconds':120},'code_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'checks':['211-feature parity','forward and chunk state','checkpoint reload','explicit reset','prefix causality','weighted loss','wmse/huber gradients','two scheduled AdamW updates','global gradient clipping','writer-reader resume checkpoint'],'not_established':['TPU numerical parity','TPU replica reduction','actual producer schema compatibility','full B32/C2000 memory/timing','model quality or multiplier']}
(base/'CPU_LIFECYCLE_RECEIPT_R3.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':receipt['status'],'tests_passed':total,'test_seconds':receipt['elapsed_test_seconds']}))
