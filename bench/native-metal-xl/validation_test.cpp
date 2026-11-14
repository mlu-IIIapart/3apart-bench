// Build: clang++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off -pthread validation_test.cpp -o validation_test
// Small controls: validation_test
// CPU-only 100M comparison: validation_test OUTPUT.f32 INPUT.3ain REFERENCE.f64 [REPEATS]
#include "validation_parallel.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>

using xl_validation::Check;
static double edges64[25];
// Verbatim numerical body of baseline native_bench.mm validate(), kept as an
// independent oracle. Do not change it to match the optimized implementation.
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
static void controls(){
 size_t cases=0;
 for(unsigned workers:{1u,2u,7u,18u}) {
   xl_validation::ParallelValidator validator(workers,7);
   auto check=[&](Fixture f,size_t count=SIZE_MAX){if(count==SIZE_MAX)count=f.out.size();auto a=baseline(f.out.data(),count,f.ref,f.layer),b=validator.validate(f.out.data(),count,f.ref,f.layer,edges64);demand(same(a,b),"summary mismatch, workers="+std::to_string(workers)+" case="+std::to_string(cases));cases++;};
   for(size_t n:{0u,1u,6u,7u,8u,23u,127u,8193u})check(fixture(n));
   auto original=fixture(43);
   for(float v:{NAN,INFINITY,-INFINITY,0.f,-0.f,std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::min(),std::numeric_limits<float>::max(),-std::numeric_limits<float>::max()})for(size_t k:{0u,1u,2u,21u,128u}){auto f=original;f.out[k]=v;check(f);}
   for(double v:{double(NAN),double(INFINITY),double(-INFINITY),std::numeric_limits<double>::max(),-std::numeric_limits<double>::max()}){auto f=original;f.ref[29]=v;check(f);}
   for(int l:{-1,24,INT32_MIN,INT32_MAX}){auto f=original;f.layer[12]=l;check(f);}
   {auto f=original;f.ref.pop_back();check(f);}
   check(original,original.out.size()-1);check(original,original.out.size()+1);
   {auto f=original;f.out[0]=f.out[1]=f.out[2]=0;check(f);}
   // Include both sides of the strict tolerance threshold and signed zero.
   for(double e:{0.,-0.,std::nextafter(5e-4,0.),5e-4,std::nextafter(5e-4,1.),-5e-4}){auto f=original;f.ref[0]=double(f.out[0])+e;check(f);}
   for(size_t edge=0;edge<25;edge++)for(float z:{float(edges64[edge]),std::nextafter(float(edges64[edge]),-INFINITY),std::nextafter(float(edges64[edge]),INFINITY)}){auto f=original;f.out[2]=z;f.out[0]=std::sqrt(1.f-z*z);f.out[1]=0;f.layer[0]=int(std::min(edge,size_t(23)));check(f);}
   std::mt19937 rng(470331);for(int round=0;round<50;round++){auto f=original;for(int k=0;k<12;k++){size_t at=rng()%f.out.size();float x;uint32_t bits=rng();std::memcpy(&x,&bits,4);f.out[at]=x;}check(f);}
 }
 std::cout<<"{\"controls_passed\":"<<cases<<",\"all_summary_fields_bit_identical\":true}\n";
}
template<class T>static std::vector<T> load(const std::string&name,size_t count,size_t offset=0){std::ifstream f(name,std::ios::binary);demand(bool(f),"open "+name);f.seekg(offset);std::vector<T>v(count);f.read(reinterpret_cast<char*>(v.data()),count*sizeof(T));demand(bool(f),"read "+name);return v;}
static double median(std::vector<double>v){std::sort(v.begin(),v.end());return v.size()%2?v[v.size()/2]:(v[v.size()/2-1]+v[v.size()/2])/2;}
static void jsonArray(const std::vector<double>&v){std::cout<<'[';for(size_t i=0;i<v.size();i++){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']';}
int main(int argc,char**argv){try{
 for(int l=0;l<=24;l++)edges64[l]=.9*(1.-2.*l/24.);
 if(argc==1){controls();return 0;}
 demand(argc==4||argc==5,"usage: validation_test OUTPUT.f32 INPUT.3ain REFERENCE.f64 [REPEATS]");
 auto header=load<uint32_t>(argv[2],4);demand(header[0]==0x4e494133&&header[1]==1,"3AIN header");size_t n=header[2];
 Fixture f;f.layer=load<int32_t>(argv[2],n,16);f.out=load<float>(argv[1],3*n);f.ref=load<double>(argv[3],3*n);int repeats=argc==5?std::stoi(argv[4]):10;demand(repeats>0,"positive repeats");
 using Clock=std::chrono::steady_clock;const auto oracle=baseline(f.out.data(),f.out.size(),f.ref,f.layer);demand(oracle.pass,"saved GPU output does not pass baseline");
 std::cout.precision(12);std::cout<<"{\"markers\":"<<n<<",\"coordinates\":"<<3*n<<",\"repeats\":"<<repeats<<",\"max_abs_error\":"<<oracle.max_error<<",\"series\":[";
 std::vector<double> serial;for(int i=0;i<repeats;i++){auto start=Clock::now();auto c=baseline(f.out.data(),f.out.size(),f.ref,f.layer);auto ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();demand(same(c,oracle),"serial repeat mismatch");serial.push_back(ms);}
 std::cout<<"{\"validator\":\"original_serial\",\"workers\":1,\"median_ms\":"<<median(serial)<<",\"raw_ms\":";jsonArray(serial);std::cout<<'}';
 unsigned bestWorkers=1;double bestMedian=std::numeric_limits<double>::infinity();
 for(unsigned workers:{1u,4u,8u,12u,18u}){
  xl_validation::ParallelValidator validator(workers);demand(same(validator.validate(f.out.data(),f.out.size(),f.ref,f.layer,edges64),oracle),"parallel warmup mismatch");std::vector<double> samples;
  for(int i=0;i<repeats;i++){auto start=Clock::now();auto c=validator.validate(f.out.data(),f.out.size(),f.ref,f.layer,edges64);auto ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();demand(same(c,oracle),"parallel repeat mismatch");samples.push_back(ms);}
  const double med=median(samples);if(med<bestMedian){bestMedian=med;bestWorkers=workers;}
  std::cout<<",{\"validator\":\"parallel_exact\",\"workers\":"<<workers<<",\"median_ms\":"<<med<<",\"raw_ms\":";jsonArray(samples);std::cout<<'}';
 }
 xl_validation::ParallelValidator best(bestWorkers);std::vector<double>pairedSerial,pairedParallel;
 auto serialRun=[&](){auto t=Clock::now();auto c=baseline(f.out.data(),f.out.size(),f.ref,f.layer);auto ms=std::chrono::duration<double,std::milli>(Clock::now()-t).count();demand(same(c,oracle),"paired serial summary mismatch");pairedSerial.push_back(ms);};
 auto parallelRun=[&](){auto t=Clock::now();auto c=best.validate(f.out.data(),f.out.size(),f.ref,f.layer,edges64);auto ms=std::chrono::duration<double,std::milli>(Clock::now()-t).count();demand(same(c,oracle),"paired parallel summary mismatch");pairedParallel.push_back(ms);};
 for(int i=0;i<repeats;i++){if(i%2){parallelRun();serialRun();}else{serialRun();parallelRun();}}
 std::cout<<"],\"paired_confirmation\":{\"order\":\"AB then BA alternating by pair\",\"pairs\":"<<repeats<<",\"workers_selected_from_sweep\":"<<bestWorkers<<",\"serial_median_ms\":"<<median(pairedSerial)<<",\"parallel_median_ms\":"<<median(pairedParallel)<<",\"serial_raw_ms\":";jsonArray(pairedSerial);std::cout<<",\"parallel_raw_ms\":";jsonArray(pairedParallel);
 std::cout<<"},\"all_summaries_bit_identical\":true,\"file_loading_and_pool_creation_excluded\":true,\"gpu_work_performed\":false}\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
