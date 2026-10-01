"""Small source-only checks. Does not import Torch or execute the probe."""
from pathlib import Path
import ast, hashlib, json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

trainer=ROOT/'research/iterations/2026-10-01-overnight/training/train_aux.py'
assert sha(trainer)=='f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f'
source=trainer.read_text()
classes=[n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='CudaPrefetcher']
assert len(classes)==1
node=classes[0]
extracted='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'
assert extracted==(HERE/'extracted-CudaPrefetcher.py.txt').read_text()
probe=(HERE/'probe.py').read_text();tree=ast.parse(probe)
for p in [HERE/'probe.py',HERE/'driver.py',HERE/'compare.py',HERE/'check_preparation.py']:
    compile(p.read_text(),str(p),'exec')
calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
syncs=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='synchronize']
assert len(syncs)==2 # one initial, one inside the four-epoch loop
assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ['empty_cache','set_per_process_memory_fraction'] for n in calls)
assert 'from wnn' not in probe and 'import trainer' not in probe
assert 'current = CudaPrefetcher(loader, device)' in probe
assert "if a.arm == 'per_epoch' or current is None:" in probe
fixture_bytes=2*(32*20000*211*4+32*20000*2*4+32*20000+32*20000*4)
assert fixture_bytes==1096960000
receipt={'status':'PASS_SOURCE_ONLY_NO_PROBE_EXECUTION','trainer_sha256':sha(trainer),
         'extracted_class_sha256':hashlib.sha256(extracted.encode()).hexdigest(),
         'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [HERE/'probe.py',HERE/'driver.py',HERE/'compare.py',HERE/'check_preparation.py']},
         'generated_fixture_bytes':fixture_bytes,'full_fixture_extra_copy_required':False,
         'epochs':4,'batches_per_epoch':32,'explicit_initial_sync_count':1,'explicit_epoch_sync_count':4,
         'allocator_configuration_changed':False,'Torch_imported':False,'GPU_executed':False,'competition_inputs_read':False}
(HERE/'preparation-check.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
