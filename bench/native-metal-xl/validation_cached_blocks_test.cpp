// Build: clang++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off -pthread validation_cached_blocks_test.cpp -o validation_cached_blocks_test
#include "validation_cached_blocks.hpp"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>

using xl_validation::Check;
using xl_validation::CachedBlockValidator;
static double edges64[25];
// Independent historical serial oracle; do not rewrite to follow the cache.
static Check baseline(const float*out,size_t count,const std::vector<double>&ref,const std::vector<int32_t>&layer){
 Check c;c.expected=3*layer.size();c.actual=count;if(count!=c.expected||ref.size()!=c.expected)return c;
 for(size_t i=0;i<count;i++){if(!std::isfinite(out[i])||!std::isfinite(ref[i])){c.nonfinite++;continue;}double e=std::abs(double(out[i])-ref[i]);c.max_error=std::max(c.max_error,e);c.error_count+=e>5e-4;}
 for(size_t i=0;i<layer.size();i++){double x=out[3*i],y=out[3*i+1],z=out[3*i+2],r=std::hypot(x,y,z),h=z/(r?r:1);int l=layer[i];if(!std::isfinite(r)||r==0||l<0||l>=24||!(h<=edges64[l]&&h>=edges64[l+1]))c.outside++;if(std::isfinite(r))c.max_radius_error=std::max(c.max_radius_error,std::abs(r-1.));}
 c.pass=c.nonfinite==0&&c.error_count==0&&c.outside==0;return c;
}
static void demand(bool b,const std::string&what){if(!b)throw std::runtime_error(what);}
static bool same(const Check&a,const Check&b){
 return a.expected==b.expected&&a.actual==b.actual&&a.nonfinite==b.nonfinite&&a.error_count==b.error_count&&a.outside==b.outside&&a.pass==b.pass&&
 std::memcmp(&a.max_error,&b.max_error,sizeof(double))==0&&std::memcmp(&a.max_radius_error,&b.max_radius_error,sizeof(double))==0;
}
struct Fixture {std::vector<float> out;std::vector<double> ref;std::vector<int32_t> layer;};
static Fixture fixture(size_t n){
 Fixture f;f.out.resize(3*n);f.ref.resize(3*n);f.layer.resize(n);
 for(size_t i=0;i<n;i++){int l=i%24;double z=(edges64[l]+edges64[l+1])/2,angle=.013*i;double v[3]={std::sqrt(1-z*z)*std::cos(angle),std::sqrt(1-z*z)*std::sin(angle),z};f.layer[i]=l;for(int k=0;k<3;k++){f.ref[3*i+k]=v[k];f.out[3*i+k]=float(v[k]);}}
 return f;
}
static size_t cases=0;
static void equivalent(const xl_validation::CachedBlockResult&r,const Fixture&f,size_t count=SIZE_MAX){
 if(count==SIZE_MAX)count=f.out.size();
 demand(same(r.check,baseline(f.out.data(),count,f.ref,f.layer)),"summary mismatch at case "+std::to_string(cases));
 cases++;
}
template<class F>static void mustThrow(F f){bool threw=false;try{f();}catch(const std::exception&){threw=true;}demand(threw,"expected rejection");cases++;}

int main(){try{
 for(int l=0;l<=24;l++)edges64[l]=.9*(1.-2.*l/24.);
 mustThrow([]{CachedBlockValidator c(0);});
 for(unsigned workers:{1u,2u,7u,18u}){
  CachedBlockValidator c(workers);
  mustThrow([&]{c.validate(nullptr,0,nullptr,0);});
  uint64_t lastGeneration=0,lastContext=0;
  for(size_t n:{0u,1u,4095u,4096u,4097u,8193u}){
   auto f=fixture(n);const size_t blocks=(n+4095)/4096;
   c.reset(f.ref,f.layer,edges64);
   auto r=c.validate(f.out.data(),f.out.size(),nullptr,0);equivalent(r,f);
   demand(r.cache_valid&&r.generation>lastGeneration&&r.context_generation>lastContext,"initial generation");
   lastGeneration=r.generation;lastContext=r.context_generation;
   demand(r.rechecked_markers==n&&r.reused_markers==0&&r.rechecked_blocks==blocks,"initial complete check");
   std::vector<uint32_t> flags(blocks,0);
   r=c.validate(f.out.data(),f.out.size(),flags);equivalent(r,f);
   demand(r.cache_valid&&r.generation>lastGeneration&&r.rechecked_markers==0&&r.reused_markers==n,"full reuse");
   lastGeneration=r.generation;
   c.invalidate();r=c.validate(f.out.data(),f.out.size(),flags);equivalent(r,f);
   demand(r.rechecked_markers==n&&r.reused_markers==0,"invalidation forces all");
   lastGeneration=r.generation;
   std::fill(flags.begin(),flags.end(),1);r=c.validate(f.out.data(),f.out.size(),flags);equivalent(r,f);
   demand(r.rechecked_markers==n&&r.reused_markers==0,"all changed");lastGeneration=r.generation;
   if(!n)continue;
   // A short coordinate count neither reads the buffer nor permits reuse.
   r=c.validate(nullptr,f.out.size()-1,flags);equivalent(r,f,f.out.size()-1);
   demand(!r.cache_valid&&r.generation==0&&!r.check.pass,"short output invalidation");
   std::fill(flags.begin(),flags.end(),0);r=c.validate(f.out.data(),f.out.size(),flags);equivalent(r,f);
   demand(r.rechecked_markers==n,"full scan after short output");
   mustThrow([&]{c.validate(f.out.data(),f.out.size(),nullptr,0);});
   r=c.validate(f.out.data(),f.out.size(),nullptr,0);equivalent(r,f);
   demand(r.rechecked_markers==n,"full scan after flag shape rejection");
   lastGeneration=r.generation;
  }
  auto original=fixture(8193), f=original;
  c.reset(f.ref,f.layer,edges64);equivalent(c.validate(f.out.data(),f.out.size(),nullptr,0),f);
  // One block changes; other summaries survive, including the one-marker tail.
  for(size_t at:{0u,4096u,8192u})for(float v:{NAN,INFINITY,-INFINITY,0.f,-0.f,std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::max()}){
   std::vector<uint32_t> flags(3,0);flags[at/4096]=7;
   f.out[3*at]=v;auto r=c.validate(f.out.data(),f.out.size(),flags);equivalent(r,f);
   const size_t changed=at==8192?1:4096;
   demand(r.rechecked_blocks==1&&r.rechecked_markers==changed&&r.reused_markers==8193-changed,"single-block coverage");
   flags.assign(3,0);auto reused=c.validate(f.out.data(),f.out.size(),flags);equivalent(reused,f);
   demand(reused.cache_valid&&reused.reused_markers==8193,"reuse failed gate summaries");
   f.out[3*at]=original.out[3*at];flags[at/4096]=1;
   equivalent(c.validate(f.out.data(),f.out.size(),flags),f);
  }
  // Exact counts and maxima must decrease when an offending block is repaired.
  std::vector<uint32_t> all(3,1), zero(3,0);
  f.out[3]=NAN;f.out[3*4097]=INFINITY;f.out[3*8192]=100.f;
  equivalent(c.validate(f.out.data(),f.out.size(),all),f);
  f=original;auto fixed=c.validate(f.out.data(),f.out.size(),all);equivalent(fixed,f);demand(fixed.check.pass,"repair clears cached failures");
  // Caller mutations cannot alter the owned reference, layer, or edge context.
  auto changed=original;c.reset(changed.ref,changed.layer,edges64);
  changed.ref[0]=NAN;changed.layer[0]=-1;
  const double edge=edges64[0];edges64[0]=-100;
  auto owned=c.validate(original.out.data(),original.out.size(),nullptr,0);
  edges64[0]=edge;equivalent(owned,original);
  // Explicit resets invalidate numerical context even if the marker count stays fixed.
  for(int kind=0;kind<4;kind++){
   f=original;if(kind==0)f.ref[9]=NAN;if(kind==1)f.layer[7]=-1;if(kind==2)f.ref[17]+=.001;if(kind==3)f.layer[1]=24;
   c.reset(f.ref,f.layer,edges64);auto r=c.validate(f.out.data(),f.out.size(),zero);equivalent(r,f);demand(r.rechecked_markers==8193,"context reset forces full check");
  }
  // Random raw float patterns and arbitrary changed-block combinations.
  std::mt19937 rng(20261114);f=original;c.reset(f.ref,f.layer,edges64);
  equivalent(c.validate(f.out.data(),f.out.size(),nullptr,0),f);
  for(int round=0;round<80;round++){
   std::vector<uint32_t> flags(3,0);
   for(int k=0;k<8;k++){size_t at=rng()%f.out.size();uint32_t bits=rng();std::memcpy(&f.out[at],&bits,4);flags[(at/3)/4096]=1;}
   equivalent(c.validate(f.out.data(),f.out.size(),flags),f);
  }
  // A malformed context is rejected and cannot leave the old cache reusable.
  auto bad=original;bad.ref.pop_back();mustThrow([&]{c.reset(bad.ref,bad.layer,edges64);});
  auto r=c.validate(f.out.data(),f.out.size(),zero);equivalent(r,f);demand(r.rechecked_markers==8193,"bad reset invalidates");
 }
 std::cout<<"{\"controls_passed\":"<<cases<<",\"all_summary_fields_bit_identical\":true,\"block_markers\":4096,\"workers_tested\":[1,2,7,18]}\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
