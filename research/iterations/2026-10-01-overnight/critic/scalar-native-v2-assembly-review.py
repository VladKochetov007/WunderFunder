"""Independent static disassembly only; no model import, inference, data, or timing."""
from pathlib import Path
import argparse,collections,hashlib,json,os,re,subprocess

EXPECTED={
 'v2':'03d7a8e32aa159c82b1af40a6376f553055729e32ed9e478c559c9d1e4310d88',
 'avx2':'b0fe37fff1d3599e521d765ee798a1e104181bf33769aca2744497360b8f5d95',
 'v3':'d1492ca1f5a0537e00548ec6e3b81b7ef407ae988cbe001b0c354efa4ebf9a51'}
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
  assert (sub_scalar,mul_scalar,add_scalar)==(16,16,32)
  assert sub_packed==0 and add_packed==0
  assert mul_packed==(4 if level=='v2' else 2)
  assert counts['subsd']+counts['vsubsd']==2 and counts['divsd']+counts['vdivsd']==1
  assert counts['cvtsd2ss']+counts['vcvtsd2ss']==1
  callers=[]
  for name,lines in blocks.items():
   if name==NAME:continue
   calls=[line.strip() for line in lines if re.search(r'\bcall\s+',line) and '<'+NAME+'>' in line]
   if calls:callers.append({'function':name,'calls':calls})
  assert callers
  arithmetic=[i for i in instructions if i['opcode'] in {'subss','vsubss','mulss','vmulss','addss','vaddss','mulps','vmulps','subsd','vsubsd','divsd','vdivsd','cvtsd2ss','vcvtsd2ss'}]
  result['modules'][level]={'sha256':expected,'function_header':body[0],
    'aggregate_disassembly_sha256':hashlib.sha256(('\n'.join(body)+'\n').encode()).hexdigest(),
    'opcodes':dict(sorted(counts.items())),'fused_multiply_add_inside_aggregate':False,
    'interpolation_and_ordered_sum_simd':False,'final_scale_simd_lanes':4 if level=='v2' else 8,
    'scalar_core_operations':{'subtract':sub_scalar,'multiply':mul_scalar,'add':add_scalar},
    'packed_core_subtract_add':{'subtract':sub_packed,'add':add_packed},
    'packed_final_scale_multiply_instructions':mul_packed,'out_of_line_callers':callers,
    'arithmetic_instructions':arithmetic}
  assert sha(path)==expected
 result['status']='STATIC_ROUNDING_BOUNDARY_CONFIRMED_CORE_INTERPOLATION_STILL_SCALAR'
 result['conclusion']='The GNU function boundary prevents contraction in these exact binaries; all44 feature iterations use16 separately rounded scalar interpolation lanes and sequential output additions. Only final scaling is packed SIMD. Removing volatile does not by itself establish SIMD interpolation or any runtime improvement.'
 result['limits']=['No machine-code theorem for every FP environment; no empirical numeric fixtures or performance measurements repeated.','No proof of performance gain from static instruction counts.','Projection/trunk outside aggregate retain existing compiler policy; FMA outside aggregate is not rejected.','Possible aliasing as the missed-vectorization cause is an inference, not established by this inspection.']
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'output_sha256':sha(a.output),'module_count':len(result['modules'])}))
if __name__=='__main__':main()
