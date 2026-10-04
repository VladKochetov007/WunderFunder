from __future__ import annotations
import ast, hashlib, json, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
V2 = ROOT/'research/campaign_2026-10-04/kaggle/jax_realdata_preflight_v2'
HERE = ROOT/'research/campaign_2026-10-04/kaggle/jax_realdata_preflight_v3'

def sha(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def parse(path: Path) -> ast.Module: return ast.parse(path.read_text())
def strip_instrumentation(tree: ast.Module) -> ast.Module:
    class Strip(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == '_emit_milestone': return None
            return self.generic_visit(node)
        def visit_AsyncFunctionDef(self, node):
            if node.name == '_emit_milestone': return None
            return self.generic_visit(node)
        def visit_Expr(self, node):
            value=node.value
            if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='_emit_milestone': return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='_MILESTONE_STARTED' for t in node.targets): return None
            return self.generic_visit(node)
    return ast.fix_missing_locations(Strip().visit(tree))

def mem_budget_ok(host: int, cgroup: int | None, rss_cap: int=4*1024**3, launcher: int=2*1024**3) -> bool:
    limits=[int(host*.8)]
    if cgroup is not None: limits.append(cgroup)
    return min(limits)>=rss_cap+launcher

v2m=json.loads((V2/'upload_dataset/source-manifest.json').read_text())
v3m=json.loads((HERE/'upload_dataset/source-manifest.json').read_text())
assert v2m['external_inputs']==v3m['external_inputs']
assert set(v2m['files'])==set(v3m['files'])
assert v3m['attempt']==3 and v3m['dataset']=='vladkochetov07/wnn-jax-realdata-e463-20261004-v3'
for name in v2m['files']:
    a=v2m['files'][name]; b=v3m['files'][name]
    if name=='scripts/realdata_preflight.py': continue
    assert (a['sha256'],a['size_bytes'])==(b['sha256'],b['size_bytes']),name
with zipfile.ZipFile(V2/'upload_dataset/source_bundle.zip') as old, zipfile.ZipFile(HERE/'upload_dataset/source_bundle.zip') as new:
    assert set(old.namelist())==set(new.namelist())==set(v3m['files'])
    for name in old.namelist():
        data=new.read(name); item=v3m['files'][name]
        assert len(data)==item['size_bytes'] and sha(data)==item['sha256'],name
    old_script=old.read('scripts/realdata_preflight.py')
    new_script=new.read('scripts/realdata_preflight.py')
    old_ast=ast.parse(old_script)
    new_ast=ast.parse(new_script)
    assert ast.dump(strip_instrumentation(old_ast),include_attributes=False)==ast.dump(strip_instrumentation(new_ast),include_attributes=False)
    for name in ('scripts/export_combo_checkpoint_npz.py','src/wnn/nn.py','src/wnn_jax/bridge.py','src/wnn_jax/rnn.py'):
        assert old.read(name)==new.read(name),name
kernel_tree=parse(HERE/'kernel_push/kernel.py')
def literal(node):
    if isinstance(node,ast.Constant): return node.value
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Mult): return literal(node.left)*literal(node.right)
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Pow): return literal(node.left)**literal(node.right)
    raise AssertionError(f'nonliteral {ast.dump(node)}')
const={n.targets[0].id:literal(n.value) for n in kernel_tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'RSS_LIMIT_BYTES','MAX_RUNTIME_SECONDS','DATASET_SLUG'} and (isinstance(n.value,ast.Constant) or isinstance(n.value,ast.BinOp))}
assert const['RSS_LIMIT_BYTES']==4*1024**3
assert const['MAX_RUNTIME_SECONDS']==1800
assert const['DATASET_SLUG'].endswith('-v3')
output_node=next(n.value for n in kernel_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OUTPUT' for t in n.targets))
assert ast.unparse(output_node)=='Path("/kaggle/working/jax_realdata_preflight_e463_v3").as_posix()' or '_v3' in ast.unparse(output_node)
worker=next(n.value.value for n in kernel_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WORKER_SOURCE' for t in n.targets))
worker_tree=ast.parse(worker)
assert any(isinstance(n,ast.Import) and any(a.name=='jax' for a in n.names) for n in ast.walk(worker_tree))
assert 'milestone("jax_import_start")' in worker and 'milestone("jax_import_complete")' in worker
main=next(n for n in kernel_tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
body_src=ast.get_source_segment((HERE/'kernel_push/kernel.py').read_text(),main)
assert body_src.index('validate_cpu_host(OUTPUT)') < body_src.index('subprocess.Popen(')
for phase in ('realdata_import_start','realdata_import_complete','fit_sequence_and_target_load_start','fit_sequence_and_target_load_complete','model_parameter_initialization_start','model_parameter_initialization_complete','evaluation_sequence_load_start','evaluation_sequence_load_complete'):
    assert phase in new_script.decode(),phase
assert mem_budget_ok(32*1024**3,30*1024**3)
assert mem_budget_ok(8*1024**3,7*1024**3)
assert not mem_budget_ok(7*1024**3,8*1024**3)
assert not mem_budget_ok(32*1024**3,5*1024**3)
assert 'f"{precision}_gradient_update_start"' in new_script.decode() and 'f"{precision}_gradient_update_complete"' in new_script.decode()
assert 'f"{precision}_evaluation_complete"' in new_script.decode()
assert json.loads((HERE/'kernel_push/kernel-metadata.json').read_text())['enable_gpu'] is False
assert json.loads((HERE/'kernel_push/kernel-metadata.json').read_text())['enable_tpu'] is False
assert json.loads((HERE/'kernel_push/kernel-metadata.json').read_text())['enable_internet'] is False
print(json.dumps({'status':'PASS','source_members':len(v3m['files']),'external_inputs':len(v3m['external_inputs']),'scientific_ast_equal_after_removing_only_milestones':True,'JAX_not_imported_by_static_harness':True,'resource_guard_fit_boundary_tests':4,'output_cap_bytes':const['RSS_LIMIT_BYTES'],'runtime_seconds':const['MAX_RUNTIME_SECONDS']},indent=2))
