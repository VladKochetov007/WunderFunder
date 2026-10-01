"""Independent static disassembly only; no model import, inference, data, or timing."""
from pathlib import Path
import argparse,collections,hashlib,json,os,re,subprocess

EXPECTED={
 'v2':'b64bb8ab124337fe9e3539226dc8fc000d099ab089ee509e25a479040d80b0cd',
 'avx2':'1e7a120dfd250a7b3f468be4a6385d6624cc0094051f775bc58637f3f33f89a3',
 'v3':'75c631b26ee8f989340603b50bfc3959c2778cce91df6350b440ef2b49ca77b3'}
NAME='wnn::ScalarEmbedding::aggregate(float const*, float*) const'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 os.sched_setaffinity(0,{37})
 source=a.root/'research/iterations/2026-10-01-overnight/scalar/native_v2'
 result={'scope':'Static source/disassembly only; no imports of candidate modules, models, data, inference or timing.','CPU_affinity':[37],
         'compiler':subprocess.run(['g++','--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0],
         'build_script_sha256':sha(source/'build.sh'),'aggregate_source_sha256':sha(source/'include/wnn/scalar_embedding.hpp'),
         'reviewer_source_sha256':sha(__file__),'modules':{}}
 build=(source/'build.sh').read_text();header=(source/'include/wnn/scalar_embedding.hpp').read_text()
 assert '-O3' in build and '-mtune=generic' in build and '-fno-trapping-math' in build and '-fno-math-errno' in build
 assert '-ffast-math' not in build and '-fassociative-math' not in build and '-flto' not in build
 assert sha(source/'include/wnn/scalar_embedding.hpp')=='34a63687a069c568e70c33609b7d4e3ef0e8a302339f2ee3b3abe3c1a1e82b80'
 assert 'float* __restrict__ out' in header
 assert '__attribute__((optimize("fp-contract=off")))' in header and '#error "SCALARPLE native_v2 requires GCC' in header
 for level,expected in EXPECTED.items():
  path=source/'build'/level/'wnn_cpp.cpython-311-x86_64-linux-gnu.so';assert sha(path)==expected
  text=subprocess.run(['objdump','-d','-C','--no-show-raw-insn',str(path)],capture_output=True,text=True,check=True).stdout
  blocks={};current=None
  for line in text.splitlines():
   match=re.match(r'^([0-9a-f]+) <(.*)>:$',line)
   if match:current=match.group(2);blocks[current]=[line]
   elif current is not None:blocks[current].append(line)
  matches=[name for name in blocks if 'ScalarEmbedding::aggregate' in name]
  assert matches==[NAME],matches
  body=blocks[NAME];instructions=[]
  for line in body:
   match=re.match(r'^\s*([0-9a-f]+):\s+([a-z0-9]+)\s*(.*)$',line)
   if match:instructions.append({'address':match.group(1),'opcode':match.group(2),'operands':match.group(3)})
  counts=collections.Counter(i['opcode'] for i in instructions)
  fused=[i for i in instructions if re.match(r'v?f(n?madd|n?msub|maddsub|msubadd)',i['opcode'])]
  assert not fused
  sub_scalar=counts['subss']+counts['vsubss'];mul_scalar=counts['mulss']+counts['vmulss'];add_scalar=counts['addss']+counts['vaddss']
  sub_packed=counts['subps']+counts['vsubps'];add_packed=counts['addps']+counts['vaddps'];mul_packed=counts['mulps']+counts['vmulps']
  assert (sub_scalar,mul_scalar,add_scalar)==(0,0,0)
  lanes=4 if level=='v2' else 8
  groups=16//lanes
  assert sub_packed==groups and add_packed==2*groups
  assert mul_packed==2*groups
  assert counts['subsd']+counts['vsubsd']==2 and counts['divsd']+counts['vdivsd']==1
  assert counts['cvtsd2ss']+counts['vcvtsd2ss']==1
  callers=[]
  for name,lines in blocks.items():
   if name==NAME:continue
   calls=[line.strip() for line in lines if re.search(r'\bcall\s+',line) and '<'+NAME+'>' in line]
   if calls:callers.append({'function':name,'calls':calls})
  assert len(callers)==2,callers
  assert any('wnn::Predictor::step' in c['function'] for c in callers)
  arithmetic=[i for i in instructions if i['opcode'] in {'subss','vsubss','mulss','vmulss','addss','vaddss','mulps','vmulps','subps','vsubps','addps','vaddps','subsd','vsubsd','divsd','vdivsd','cvtsd2ss','vcvtsd2ss'}]
  result['modules'][level]={'sha256':expected,'function_header':body[0],
    'aggregate_disassembly_sha256':hashlib.sha256(('\n'.join(body)+'\n').encode()).hexdigest(),
    'opcodes':dict(sorted(counts.items())),'fused_multiply_add_inside_aggregate':False,
    'interpolation_and_ordered_sum_simd':True,'final_scale_simd_lanes':4 if level=='v2' else 8,
    'scalar_core_operations':{'subtract':sub_scalar,'multiply':mul_scalar,'add':add_scalar},
    'packed_core_subtract_add':{'subtract':sub_packed,'add':add_packed},
    'packed_interpolation_multiply_instructions':groups,'packed_final_scale_multiply_instructions':groups,'out_of_line_callers':callers,
    'arithmetic_instructions':arithmetic}
  assert sha(path)==expected
 result['status']='PASS_STATIC_SIMD_AND_SEPARATE_ROUNDING_BOUNDARY'
 result['conclusion']='The exact final binaries preserve a non-inlined aggregate boundary with no fused multiply-add. Each ordered feature iteration uses four SSE4-lane groups (v2) or two AVX8-lane groups (avx2/v3), with separate subtract, multiply, interpolation add and sequential feature-accumulation add. FP64 fraction arithmetic and its single FP32 conversion remain. Final scaling is separately packed. This proves the inspected instruction pattern, not a runtime gain.'
 result['limits']=['No machine-code theorem for every FP environment; no empirical numeric fixtures or performance measurements repeated.','No proof of performance gain from static instruction counts.','Projection/trunk outside aggregate retain existing compiler policy; FMA outside aggregate is not rejected.','Output-only restrict is sound for the two supported Python-extension call paths, as documented in the separately preserved static note; direct third-party C++ callers inherit its nonaliasing precondition.']
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'output_sha256':sha(a.output),'module_count':len(result['modules'])}))
if __name__=='__main__':main()
