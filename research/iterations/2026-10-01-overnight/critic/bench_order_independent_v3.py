"""Independent fixed-three full-callback replication using the original benchmark worker."""
from pathlib import Path
import argparse,shutil,json
from bench_fixed_v3 import run
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--condition',choices=['baseline3','order_v2_3']);p.add_argument('--repeat',type=int,default=0);a=p.parse_args();dest=HERE/'order-runtime'
 if a.mode=='prepare':
  assert not dest.exists();dest.mkdir();(dest/'raw.npy').symlink_to(HERE/'runtime-probe/raw.npy')
  for name in ['baseline3','order_v2_3']:
   shutil.copytree(HERE/'runtime-probe/fixed3h128',dest/name)
   if name=='order_v2_3':
    for isa in ['v2','avx2','v3']:
     for old in (dest/name/isa).glob('*.so'):old.unlink()
     bins=list((HERE/'native_order_v2/build'/isa).glob('*.cpython-311-*.so'));assert len(bins)==1
     shutil.copy(bins[0],dest/name/isa/bins[0].name)
  print('Prepared two raw-only fixed-three replicas; same weights, coefficients and wrapper.')
 else:
  assert not (dest/f'{a.condition}-r{a.repeat}.json').exists()
  run(dest,a.condition,a.repeat)
if __name__=='__main__':main()
