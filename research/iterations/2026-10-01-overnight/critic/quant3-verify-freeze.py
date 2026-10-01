"""Preparation provenance only: hash frozen code/artifacts/receipts, no raw fixtures."""
from pathlib import Path
import hashlib,json,zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];P=HERE.parent/'quant3'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
freeze_path=P/'FREEZE.json';expected='c01fd5698e338a2b514b617bdc4f17074f26093e1358d7725515c1c47b425519'
assert sha(freeze_path)==expected;f=json.loads(freeze_path.read_text());checked={}
for group in ['source_and_binaries','actor_receipts','independent_receipts']:
    for name,digest in f[group].items():assert sha(ROOT/name)==digest,(group,name)
    checked[group]=len(f[group])
package=ROOT/f['zip'];assert sha(package)==f['zip_sha256'];manifest=json.loads(package.with_suffix('.manifest.json').read_text())
assert manifest['zip_sha256']==f['zip_sha256'] and manifest['zip_bytes']==f['zip_bytes']==package.stat().st_size
with zipfile.ZipFile(package) as z:
    assert z.testzip() is None and set(z.namelist())==set(manifest['files'])
    for name,meta in manifest['files'].items():
        data=z.read(name);assert hashlib.sha256(data).hexdigest()==meta['sha256'] and len(data)==meta['bytes']
    for level in ['v2','avx2','v3']:
        result=json.loads((HERE/f'quant3-independent-model-{level}.json').read_text());assert result['status']=='PASS'
        entry=next(n for n in z.namelist() if n.startswith(level+'/'))
        assert result['native_sha256']==hashlib.sha256(z.read(entry)).hexdigest()
        assert result['empirical_rows']==result['checkpoint_reads']==result['WP_queries']==0
out={'status':'PREPARATION_PASS_AWAIT_ROOT_EMPIRICAL_ALLOCATION','freeze_sha256':expected,'zip_sha256':f['zip_sha256'],'zip_bytes':f['zip_bytes'],'verified_counts':checked,'all_zip_files_verified':len(manifest['files']),'native_independent_checks':'all 3 ISA builds, scalar/AVX2/VNNI integer backends, generated weights/rows, original-order arithmetic, resets/prefixes; normal-scale/environment guard','preserved_failures':['quant3-independent-kernel-result.json','quant3-subnormal-boundary-result.jsonl'],'protocol_review':{'fixed_composition_equal_thirds':True,'unchanged_valid_domain_quantizer':True,'runtime_first_all_3_short_plus_1_long_under_50min':True,'quiet_SMT_timed_regions_and_bounded_contamination_retries':True,'AVX2_extension_conditional_and_both_dispatches_forced':True,'AVX2_fullfold_byte_identity_separately_allocated':True,'two_fixed_WP_gates_unchanged':True,'composite_manifest_truthful_and_child_lineage_bound':True},'limitations':['No empirical runtime or WP pass established by this receipt.','Cross-ISA numerical screens alone do not establish WP equivalence.','AVX2 held-fold quality requires full-array byte identity; v2 remains compatibility-only unless separately qualified.','All folds are exposed exploratory data; no top-rank guarantee.'],'empirical_fixture_read':False,'targets_read':False,'forecast_fits':0,'WP_queries':0}
dest=HERE/'quant3-preparation-review.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
