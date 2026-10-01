// Independent static-contract smoke: no Python actor implementation and no empirical rows.
#include "critic/native_transition/include/wnn/transition_features.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
int main() {
  wnn::TransitionConfig c; c.enabled=true; c.conditional=true;
  for(int j=0;j<16;++j) {c.pi[j]=1./16; for(int k=0;k<16;++k) c.prior[16*j+k]=(j==k ? .25 : .05);}
  wnn::TransitionFeatures f(c);
  float neutral[4]={0,0,0,0}; float a[4]={1,-1,-1,-1}; float b[4]={-1,1,-1,-1};
  auto p=f.update(neutral); assert(p[34]==0 && p[35]==1.f/16);
  p=f.update(a); assert(p[34]==1 && p[1]==1 && p[35+1]==1.f/16 && p[16+1]==0);
  p=f.update(b); assert(p[34]==1 && p[2]==1 && p[16+1]==1 && p[35+2]==.05f);
  p=f.update(a); assert(p[34]==1 && p[35+2]==.25f); // predicts from context2 prior
  p=f.update(b); // one observed1->2 transition, pre-observation posterior(1+.8)/(1+16)
  assert(std::abs(p[35+2]-float(1.8/17))<1e-8);
  assert(std::abs(p[51+2]-float(1.-1.8/17))<1e-8);
  float bad[4]={std::numeric_limits<float>::quiet_NaN(),1,2,3};
  p=f.update(bad); assert(p[34]==0);
  p=f.update(a); assert(p[34]==0); // invalid immediate-previous suppresses this edge
  p=f.update(b); assert(p[34]==1 && p[16+2]==1); // previous emitted mark remains2 across suppressed rows
  f.reset(); p=f.update(a); assert(p[34]==0 && p[1]==0 && p[32]==0);
  p=f.update(a); assert(p[34]==0);
  float samecategory[4]={1.1f,-1,-1,-1}; p=f.update(samecategory); assert(p[34]==1 && p[1]==1);
  wnn::TransitionConfig invalid=c; invalid.pi[0]=0;
  bool caught=false; try { wnn::TransitionFeatures broken(invalid); } catch(const std::invalid_argument&) {caught=true;}
  assert(caught);
  std::cout << "PASS: independent literal pre-observation posterior, held last mark, invalid bridge, reset, same-category event, invalid config\n";
}
