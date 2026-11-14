// Additive experiment; historical benchmark and validation remain intact.
#define main historicalBenchmarkMain
#include "native_bench.mm"
#undef main
#include <random>
#include "validation_parallel.hpp"

struct Variant { const char* name; MTLMathMode mode; MTLMathFloatingPointFunctions functions; bool algebra; unsigned threads; bool directParallel; id<MTLComputePipelineState> pipeline; NSMutableArray* trials; };
int main(int argc,char**argv){@autoreleasepool{try{
 demand(argc==4,"usage: optimization_bench INPUT_PREFIX SHADER OUTPUT_JSON");
 demand(!std::filesystem::exists(argv[3]),"preserve existing result");edges();Input d=input(argv[1]);
 id<MTLDevice>device=MTLCreateSystemDefaultDevice();demand(device!=nil,"Metal device");
 demand(24ull*d.n<device.recommendedMaxWorkingSetSize/4,"GPU memory budget");
 id<MTLCommandQueue>queue=[device newCommandQueue];NSError*error=nil;
 NSString*original=[NSString stringWithContentsOfFile:ns(argv[2]) encoding:NSUTF8StringEncoding error:&error];demand(original!=nil,"read shader");
 NSString*simplified=[original stringByReplacingOccurrencesOfString:@"float th=acos(clamp(z,-1.0f,1.0f)),st=sin(th);" withString:@"z=clamp(z,-1.0f,1.0f); float st=sqrt(max(0.0f,1.0f-z*z));"];
 simplified=[simplified stringByReplacingOccurrencesOfString:@"out[3*i+2]=cos(th);" withString:@"out[3*i+2]=z;"];
 std::vector<Variant> variants={
  {"baseline_copy_serial",MTLMathModeSafe,MTLMathFloatingPointFunctionsPrecise,false,256,false},
  {"fast_intrinsics_copy_serial",MTLMathModeSafe,MTLMathFloatingPointFunctionsFast,false,256,false},
  {"baseline_direct_parallel",MTLMathModeSafe,MTLMathFloatingPointFunctionsPrecise,false,256,true},
  {"fast_intrinsics_direct_parallel",MTLMathModeSafe,MTLMathFloatingPointFunctionsFast,false,256,true}};
 xl_validation::ParallelValidator parallel(18);
 for(auto&v:variants){MTLCompileOptions*o=[MTLCompileOptions new];o.mathMode=v.mode;o.mathFloatingPointFunctions=v.functions;id<MTLLibrary>lib=[device newLibraryWithSource:v.algebra?simplified:original options:o error:&error];demand(lib!=nil,error?error.description.UTF8String:"compile");v.pipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"place_markers"] error:&error];demand(v.pipeline!=nil,"pipeline");demand(v.threads<=v.pipeline.maxTotalThreadsPerThreadgroup,"threadgroup");v.trials=[NSMutableArray array];}
 id<MTLBuffer>bl=[device newBufferWithLength:4ull*d.n options:MTLResourceStorageModeShared],bt=[device newBufferWithLength:4ull*d.n options:MTLResourceStorageModeShared],bv=[device newBufferWithLength:4ull*d.n options:MTLResourceStorageModeShared],bo=[device newBufferWithLength:12ull*d.n options:MTLResourceStorageModeShared];demand(bl&&bt&&bv&&bo,"allocation");memcpy(bl.contents,d.layer.data(),4ull*d.n);memcpy(bt.contents,d.t.data(),4ull*d.n);memcpy(bv.contents,d.v.data(),4ull*d.n);std::vector<float>host(3ull*d.n);
 auto dispatch=[&](Variant&v){auto begin=Clock::now();id<MTLCommandBuffer>cb=[queue commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];[enc setComputePipelineState:v.pipeline];[enc setBuffer:bl offset:0 atIndex:0];[enc setBuffer:bt offset:0 atIndex:1];[enc setBuffer:bv offset:0 atIndex:2];[enc setBuffer:bo offset:0 atIndex:3];[enc setBytes:edges32 length:sizeof(edges32) atIndex:4];[enc setBytes:&d.n length:4 atIndex:5];[enc dispatchThreads:MTLSizeMake(d.n,1,1) threadsPerThreadgroup:MTLSizeMake(v.threads,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];double wall=ms(begin);demand(cb.status==MTLCommandBufferStatusCompleted,"GPU command failed");double gpu=(cb.GPUEndTime-cb.GPUStartTime)*1000.;demand(gpu>0&&std::isfinite(gpu),"GPU clock");return [@{@"resident_wall_ms":@(wall),@"gpu_device_ms":@(gpu)}mutableCopy];};
 for(auto&v:variants)for(int i=0;i<3;i++)dispatch(v);
 std::mt19937 random(20261113);std::vector<unsigned>order(variants.size());for(unsigned i=0;i<order.size();i++)order[i]=i;
 NSMutableArray*executionOrder=[NSMutableArray array];
 for(unsigned r=0;r<10;r++){
  std::shuffle(order.begin(),order.end(),random);
  for(auto k:order){@autoreleasepool{auto&v=variants[k];auto start=Clock::now();NSMutableDictionary*trial=dispatch(v);double visible=ms(start);auto t=Clock::now();if(!v.directParallel)memcpy(host.data(),bo.contents,12ull*d.n);double copy=ms(t),e2e=ms(start);t=Clock::now();Check check;
   if(v.directParallel){auto c=parallel.validate((const float*)bo.contents,3ull*d.n,d.ref,d.layer,edges64);check={c.expected,c.actual,c.nonfinite,c.error_count,c.outside,c.max_error,c.max_radius_error,c.pass};}
   else check=validate(host.data(),host.size(),d.ref,d.layer);
   double validation=ms(t),verified=ms(start);trial[@"compute_output_and_validation_ms"]=@(verified);trial[@"direct_shared_output"]=@(v.directParallel);trial[@"validator_workers"]=@(v.directParallel?18:1);
   // Independent serial oracle is deliberately outside the optimized timed path.
   const Check oracle=validate((const float*)bo.contents,3ull*d.n,d.ref,d.layer);
   demand(check.expected==oracle.expected&&check.actual==oracle.actual&&check.nonfinite==oracle.nonfinite&&check.error_count==oracle.error_count&&check.outside==oracle.outside&&check.max_error==oracle.max_error&&check.max_radius_error==oracle.max_radius_error&&check.pass==oracle.pass,"parallel checker disagrees with serial oracle");trial[@"all_checker_fields_equal_serial_oracle"]=@YES;
   trial[@"trial"]=@(r);trial[@"full_output_copy_ms"]=@(copy);trial[@"resident_plus_copy_ms"]=@(e2e);trial[@"shared_output_cpu_visible_ms"]=@(visible);trial[@"validation_ms"]=@(validation);trial[@"validation"]=checkJSON(check);[v.trials addObject:trial];[executionOrder addObject:@{@"round":@(r),@"variant":ns(v.name)}];demand(check.pass,"full accuracy gate failed");printf("round=%u variant=%s gpu_ms=%.6f max_error=%.9g pass=true\n",r,v.name,[trial[@"gpu_device_ms"]doubleValue],check.max_error);fflush(stdout);
  }}
 }
 NSMutableArray*results=[NSMutableArray array];for(auto&v:variants)[results addObject:@{@"variant":ns(v.name),@"math_mode":v.mode==MTLMathModeSafe?@"safe":@"fast",@"intrinsics":v.functions==MTLMathFloatingPointFunctionsPrecise?@"precise":@"fast",@"formula":v.algebra?@"algebraic sqrt rewrite":@"original acos/sin/cos",@"threads":@(v.threads),@"direct_shared_output_and_parallel_validation":@(v.directParallel),@"trials":v.trials}];
 jsonfile(argv[3],@{@"schema":@1,@"experiment":@"native-metal-integrated-optimization",@"n":@(d.n),@"seed":@(d.seed),@"device":device.name,@"warmups_per_variant":@3,@"trials_per_variant":@10,@"randomized_round_seed":@20261113,@"requested_gpu_buffer_bytes":@(24ull*d.n),@"metal_allocated_bytes":@(device.currentAllocatedSize),@"recommended_working_set_bytes":@(device.recommendedMaxWorkingSetSize),@"variants":results,@"execution_order":executionOrder,@"scope":@"Original formula only. Every output checked by identical full-coordinate FP64 accuracy gates. Direct variants validate shared Metal buffer after GPU completion with18persistent CPUworkers andnoextra outputcopy. Every trial separately crosschecked against serial oracle outside timed path. Randomized interleaved variants,10trials each. Compute_output_and_validation includes dispatch+completion+optionalcopy+fullvalidation; inputprep,allocation,compilation and independent serialoracle excluded."});return 0;
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}}
