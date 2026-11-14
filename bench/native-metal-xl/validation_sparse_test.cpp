// Build: xcrun clang++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off -pthread validation_sparse_test.cpp -o validation_sparse_test
// Tiny CPU-only controls; no GPU work and no benchmark input files are read.
#include "validation_sparse.hpp"
#include <cstring>
#include <iostream>
#include <random>
#include <string>

using xl_validation::Check;
using xl_validation::SparseOptions;
using xl_validation::ValidationMode;
static double edges64[25];
static size_t assertions=0,summary_cases=0;
static void demand(bool value,const std::string& why) {
  ++assertions;
  if(!value) throw std::runtime_error(why);
}
template<class F> static void invalid(F body,const char* why) {
  bool caught=false;
  try { body(); } catch(const std::invalid_argument&) { caught=true; }
  demand(caught,why);
}
static bool same(const Check& a,const Check& b) {
  return a.expected==b.expected && a.actual==b.actual && a.nonfinite==b.nonfinite &&
    a.error_count==b.error_count && a.outside==b.outside && a.pass==b.pass &&
    std::memcmp(&a.max_error,&b.max_error,sizeof(double))==0 &&
    std::memcmp(&a.max_radius_error,&b.max_radius_error,sizeof(double))==0;
}
// Independent original serial checker; sparse expectations compact the selected
// points and call this body, rather than duplicating the new point-check lambda.
static Check baseline(const float*out,size_t count,const std::vector<double>&ref,const std::vector<int32_t>&layer){
 Check c;c.expected=3*layer.size();c.actual=count;if(count!=c.expected||ref.size()!=c.expected)return c;
 for(size_t i=0;i<count;i++){if(!std::isfinite(out[i])||!std::isfinite(ref[i])){c.nonfinite++;continue;}double e=std::abs(double(out[i])-ref[i]);c.max_error=std::max(c.max_error,e);c.error_count+=e>5e-4;}
 for(size_t i=0;i<layer.size();i++){double x=out[3*i],y=out[3*i+1],z=out[3*i+2],r=std::hypot(x,y,z),h=z/(r?r:1);int l=layer[i];if(!std::isfinite(r)||r==0||l<0||l>=24||!(h<=edges64[l]&&h>=edges64[l+1]))c.outside++;if(std::isfinite(r))c.max_radius_error=std::max(c.max_radius_error,std::abs(r-1.));}
 c.pass=c.nonfinite==0&&c.error_count==0&&c.outside==0;return c;
}
struct Fixture { std::vector<float> out; std::vector<double> ref; std::vector<int32_t> layer; };
static Fixture fixture(size_t n) {
  Fixture f; f.out.resize(3*n); f.ref.resize(3*n); f.layer.resize(n);
  for(size_t i=0;i<n;i++) {
    const int l=int(i%24); const double z=(edges64[l]+edges64[l+1])/2,angle=.013*i;
    const double p[3]={std::sqrt(1-z*z)*std::cos(angle),std::sqrt(1-z*z)*std::sin(angle),z};
    f.layer[i]=l;
    for(size_t k=0;k<3;k++) { f.out[3*i+k]=float(p[k]); f.ref[3*i+k]=p[k]; }
  }
  return f;
}
static Check expected(const Fixture& f,size_t actual,const xl_validation::SparsePlan& plan) {
  if(actual!=3*f.layer.size() || f.ref.size()!=3*f.layer.size() || plan.mode()==ValidationMode::Full)
    return baseline(f.out.data(),actual,f.ref,f.layer);
  Fixture subset;
  for(size_t i:plan.indices()) {
    subset.layer.push_back(f.layer[i]);
    for(size_t k=0;k<3;k++) { subset.out.push_back(f.out[3*i+k]); subset.ref.push_back(f.ref[3*i+k]); }
  }
  Check c=baseline(subset.out.data(),subset.out.size(),subset.ref,subset.layer);
  c.expected=3*f.layer.size(); c.actual=actual; return c;
}
static xl_validation::SparseCheck compare(const Fixture& f,const xl_validation::SparsePlan& plan,size_t actual=SIZE_MAX) {
  if(actual==SIZE_MAX) actual=f.out.size();
  auto result=xl_validation::validateSparse(f.out.data(),actual,f.ref,f.layer,edges64,plan);
  demand(same(result.check,expected(f,actual,plan)),"original summary differs");
  demand(result.mode==plan.mode(),"mode reported");
  demand(result.total_markers==f.layer.size(),"total reported");
  demand(result.seed==plan.seed() && result.iteration==plan.iteration(),"seed/iteration reported");
  demand(result.requested_markers==plan.requested_markers(),"request reported");
  const bool counts=actual==3*f.layer.size() && f.ref.size()==3*f.layer.size();
  demand(result.counts_valid==counts,"whole-buffer count checks");
  demand(result.checked_markers==(counts?plan.checked_markers():0),"actual marker coverage");
  demand(result.checked_coordinates==3*result.checked_markers,"coordinate coverage");
  demand(result.covers_all_markers==(counts && result.checked_markers==f.layer.size()),"full coverage flag");
  demand(result.coverage==(counts?(f.layer.empty()?1.:double(plan.checked_markers())/double(f.layer.size())):0.),"coverage fraction");
  ++summary_cases;
  return result;
}
static bool selected(const xl_validation::SparsePlan& p,size_t i) {
  return p.mode()==ValidationMode::Full || std::binary_search(p.indices().begin(),p.indices().end(),i);
}
static void checkPlan(const xl_validation::SparsePlan& p) {
  if(p.mode()==ValidationMode::Full) { demand(p.indices().empty(),"full implicit index range"); return; }
  demand(std::is_sorted(p.indices().begin(),p.indices().end()),"sorted sample");
  demand(std::adjacent_find(p.indices().begin(),p.indices().end())==p.indices().end(),"no replacement");
  demand(p.indices().back()<p.total_markers(),"indices in range");
  demand(p.checked_markers()==std::max(p.requested_markers(),p.boundary_markers()),"exact actual count");
  demand(p.checked_markers()<p.total_markers(),"sparse mode excludes full coverage");
}
static void planControls() {
  SparseOptions options; options.sample_markers=127; options.full_below_markers=0; options.seed=123;
  for(size_t n:{0u,1u,127u,4095u,4096u}) {
    auto p=xl_validation::makeSparsePlan(n,options);
    demand(p.mode()==ValidationMode::Full,"hard small-N full fallback");
    compare(fixture(n),p);
  }
  demand(xl_validation::makeSparsePlan(65536,SparseOptions{}).mode()==ValidationMode::Full,"default cutoff full");
  for(size_t n:{4097u,8193u,10000u,1310721u,25000009u}) {
    for(size_t count:{1u,13u,127u,257u}) {
      options.sample_markers=count;
      auto p=xl_validation::makeSparsePlan(n,options); checkPlan(p);
      auto repeat=xl_validation::makeSparsePlan(n,options);
      demand(p.indices()==repeat.indices(),"seed reproducible");
      demand(selected(p,0) && selected(p,1) && selected(p,n-2) && selected(p,n-1),"end boundaries covered");
      for(size_t stride:{262144u,25000000u}) for(size_t i=stride;i<n;i+=stride)
        demand(selected(p,i-1) && selected(p,i) && (i+1==n || selected(p,i+1)),"tile/chunk neighbors covered");
    }
  }
  options.sample_markers=257;
  auto a=xl_validation::makeSparsePlan(10000,options);
  options.iteration=1; auto b=xl_validation::makeSparsePlan(10000,options);
  demand(a.indices()!=b.indices(),"iteration rotates random subset");
  options.iteration=0; options.seed=124; auto c=xl_validation::makeSparsePlan(10000,options);
  demand(a.indices()!=c.indices(),"seed rotates random subset");
  options.sample_markers=9999; checkPlan(xl_validation::makeSparsePlan(10000,options));
  options.sample_markers=10000;
  demand(xl_validation::makeSparsePlan(10000,options).mode()==ValidationMode::Full,"request covers all");
  options.sample_markers=10001;
  demand(xl_validation::makeSparsePlan(10000,options).mode()==ValidationMode::Full,"oversized request full");
  options.sample_markers=1;
  auto enormous=xl_validation::makeSparsePlan(std::numeric_limits<size_t>::max()/3,options);
  checkPlan(enormous);
  demand(enormous.boundary_markers()<=6157,"boundary anchor cap bounds storage");
  options.sample_markers=0;
  invalid([&]{xl_validation::makeSparsePlan(10000,options);},"zero sample rejected");
  options.sample_markers=1;
  invalid([&]{xl_validation::makeSparsePlan(std::numeric_limits<size_t>::max()/3+1,options);},"count overflow rejected");
}
static void numericalControls() {
  SparseOptions options; options.sample_markers=257; options.full_below_markers=4096;
  for(size_t n:{43u,10000u}) {
    const auto plan=xl_validation::makeSparsePlan(n,options);
    const auto original=fixture(n);
    demand(compare(original,plan).check.pass,"valid fixture passes");
    const size_t at=plan.mode()==ValidationMode::Full ? 12 : plan.indices()[plan.indices().size()/2];
    for(float value:{NAN,INFINITY,-INFINITY,0.f,-0.f,std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::min(),std::numeric_limits<float>::max(),-std::numeric_limits<float>::max()})
      for(size_t axis:{0u,1u,2u}) { auto f=original; f.out[3*at+axis]=value; compare(f,plan); }
    for(double value:{double(NAN),double(INFINITY),double(-INFINITY),std::numeric_limits<double>::max(),-std::numeric_limits<double>::max()}) {
      auto f=original; f.ref[3*at]=value; compare(f,plan);
    }
    for(int layer:{-1,24,INT32_MIN,INT32_MAX}) { auto f=original; f.layer[at]=layer; compare(f,plan); }
    { auto f=original; f.out[3*at]=f.out[3*at+1]=f.out[3*at+2]=0; demand(!compare(f,plan).check.pass,"zero norm caught"); }
    { auto f=original; f.out[3*at]+=1; demand(!compare(f,plan).check.pass,"sampled corruption caught"); }
    for(double error:{0.,-0.,std::nextafter(5e-4,0.),5e-4,std::nextafter(5e-4,1.),-5e-4}) {
      auto f=original; f.ref[1]=error;
      demand(compare(f,plan).check.error_count==size_t(std::abs(error)>5e-4),"strict unchanged tolerance");
    }
    for(size_t edge=0;edge<25;edge++) for(float z:{float(edges64[edge]),std::nextafter(float(edges64[edge]),-INFINITY),std::nextafter(float(edges64[edge]),INFINITY)}) {
      auto f=original; f.out[3*at+2]=z; f.out[3*at]=std::sqrt(1.f-z*z); f.out[3*at+1]=0;
      f.layer[at]=int(std::min(edge,size_t(23))); compare(f,plan);
    }
    { auto f=original; f.ref.pop_back(); demand(!compare(f,plan).check.pass,"truncated reference fails"); }
    { auto f=original; f.ref.push_back(0); demand(!compare(f,plan).check.pass,"extra reference coordinate fails"); }
    demand(!compare(original,plan,original.out.size()-1).check.pass,"truncated output count fails");
    demand(!compare(original,plan,original.out.size()+1).check.pass,"extra output count fails");
    std::mt19937 random(470331);
    for(size_t round=0;round<50;round++) {
      auto f=original;
      for(size_t j=0;j<12;j++) {
        const size_t i=plan.mode()==ValidationMode::Full?random()%n:plan.indices()[random()%plan.indices().size()];
        const uint32_t bits=random(); float value; std::memcpy(&value,&bits,sizeof(value));
        f.out[3*i+random()%3]=value;
      }
      compare(f,plan);
    }
    invalid([&]{xl_validation::validateSparse(original.out.data(),original.out.size(),original.ref.data(),original.ref.size(),original.layer.data(),n-1,edges64,plan);},"plan/input count mismatch rejected");
    invalid([&]{xl_validation::validateSparse(nullptr,original.out.size(),original.ref,original.layer,edges64,plan);},"null output rejected");
    if(plan.mode()==ValidationMode::Sparse) {
      size_t missing=0; while(selected(plan,missing)) ++missing;
      auto corrupt=original; corrupt.out[3*missing]+=1;
      const auto sampled=compare(corrupt,plan);
      demand(sampled.check.pass,"unsampled corruption deliberately not detected");
      demand(!sampled.covers_all_markers && sampled.coverage<1.,"sparse result explicitly incomplete");
      demand(!baseline(corrupt.out.data(),corrupt.out.size(),corrupt.ref,corrupt.layer).pass,"full baseline detects unsampled corruption");
      demand(same(sampled.check,compare(original,plan).check),"unsampled corruption leaves sampled aggregates unchanged");
    }
  }
}
int main() {
  try {
    for(size_t l=0;l<=24;l++) edges64[l]=.9*(1.-2.*double(l)/24.);
    planControls(); numericalControls();
    std::cout<<"{\"assertions_passed\":"<<assertions<<",\"summary_cases_bit_identical\":"<<summary_cases
      <<",\"sampled_corruption_detected\":true,\"unsampled_corruption_deliberately_undetected\":true,\"gpu_work_performed\":false}\n";
    return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
