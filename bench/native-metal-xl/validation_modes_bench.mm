// Strict CPU arithmetic is required: -fno-fast-math -ffp-contract=off.
#define main historicalBenchmarkMain
#include "native_bench.mm"
#undef main
#include "validation_gpu_cache.hpp"
#include "validation_sparse.hpp"
#include <random>

static unsigned boundedUnsigned(const char* text,unsigned maximum) {
    const std::string value(text);
    demand(!value.empty() && value.find_first_not_of("0123456789")==std::string::npos,"expected positive integer");
    const unsigned long long parsed=std::stoull(value);
    demand(parsed>0 && parsed<=maximum,"integer argument out of range");
    return static_cast<unsigned>(parsed);
}

static Check historical(const xl_validation::Check& c) {
    return {c.expected,c.actual,c.nonfinite,c.error_count,c.outside,
            c.max_error,c.max_radius_error,c.pass};
}
static bool identical(const Check& a, const Check& b) {
    return a.expected==b.expected && a.actual==b.actual &&
        a.nonfinite==b.nonfinite && a.error_count==b.error_count &&
        a.outside==b.outside && a.pass==b.pass &&
        memcmp(&a.max_error,&b.max_error,sizeof(double))==0 &&
        memcmp(&a.max_radius_error,&b.max_radius_error,sizeof(double))==0;
}
static NSMutableDictionary* cacheJSON(const xl_validation::GPUCachedValidator::Result& r) {
    const auto& c=r.certificate;
    return [@{@"mode":@"full_gpu_comparison_exact_cpu_revalidation",
        @"validation":checkJSON(historical(c.check)),
        @"validation_ms":@(r.total_ms), @"gpu_compare_device_ms":@(r.gpu_device_ms),
        @"gpu_compare_wall_ms":@(r.gpu_wall_ms),
        @"cpu_update_and_reduce_ms":@(r.cpu_update_and_reduce_ms),
        @"rechecked_markers":@(c.rechecked_markers),@"reused_markers":@(c.reused_markers),
        @"rechecked_blocks":@(c.rechecked_blocks),@"reused_blocks":@(c.reused_blocks),
        @"generation":@(c.generation),@"context_generation":@(c.context_generation),
        @"cache_valid":@(c.cache_valid),@"full_coverage":@YES} mutableCopy];
}

int main(int argc,char** argv) {@autoreleasepool {try {
    demand(argc==4 || argc==5 || argc==7,"usage: validation_modes_bench INPUT_PREFIX OUTPUT_JSON WORKERS [--controls | --run full-gpu|full-cpu|sparse|sparse-rotating ITERATIONS]");
    demand(!std::filesystem::exists(argv[2]),"preserve existing result");
    const bool controls=argc==5 && std::string(argv[4])=="--controls";
    demand(argc!=5 || controls,"unknown mode");
    const bool runtime=argc==7 && std::string(argv[4])=="--run";
    demand(argc!=7 || runtime,"unknown runtime option");
    const std::string runtimeMode=runtime?argv[5]:"";
    const unsigned iterations=runtime?boundedUnsigned(argv[6],10000):0;
    if(runtime)demand((runtimeMode=="full-gpu" || runtimeMode=="full-cpu" ||
        runtimeMode=="sparse" || runtimeMode=="sparse-rotating") && iterations>0 && iterations<=10000,"runtime mode/iterations");
    const unsigned workers=boundedUnsigned(argv[3],256);
    demand(workers>0 && workers<=256,"workers");
    edges(); Input d=input(argv[1]); const size_t count=3ull*d.n;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    demand(device!=nil,"Metal device unavailable");
    const size_t gpuBytes=36ull*d.n+4ull*((d.n+4095ull)/4096);
    demand(gpuBytes<device.recommendedMaxWorkingSetSize/4,"GPU budget exceeds quarter recommended working set");
    NSError* error=nil;
    auto source=[&](const char* path){
        NSString* s=[NSString stringWithContentsOfFile:ns(path) encoding:NSUTF8StringEncoding error:&error];
        demand(s!=nil,"shader read"); return s;
    };
    MTLCompileOptions* options=[MTLCompileOptions new];
    options.mathMode=MTLMathModeSafe;
    options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsFast;
    id<MTLLibrary> library=[device newLibraryWithSource:source("bench/native-metal-xl/place.metal") options:options error:&error];
    demand(library!=nil,error?error.description.UTF8String:"placement compile");
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"place_markers"] error:&error];
    demand(pipeline!=nil && pipeline.maxTotalThreadsPerThreadgroup>=256,"placement pipeline");
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLBuffer> layer=[device newBufferWithLength:4ull*d.n options:MTLResourceStorageModeShared];
    id<MTLBuffer> times=[device newBufferWithLength:4ull*d.n options:MTLResourceStorageModeShared];
    id<MTLBuffer> values=[device newBufferWithLength:4ull*d.n options:MTLResourceStorageModeShared];
    id<MTLBuffer> output=[device newBufferWithLength:12ull*d.n options:MTLResourceStorageModeShared];
    demand(layer&&times&&values&&output,"placement allocations");
    memcpy(layer.contents,d.layer.data(),4ull*d.n);
    memcpy(times.contents,d.t.data(),4ull*d.n);
    memcpy(values.contents,d.v.data(),4ull*d.n);
    auto place=[&](){
        auto start=Clock::now();
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        demand(command&&encoder,"placement command/encoder unavailable");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:layer offset:0 atIndex:0]; [encoder setBuffer:times offset:0 atIndex:1];
        [encoder setBuffer:values offset:0 atIndex:2]; [encoder setBuffer:output offset:0 atIndex:3];
        [encoder setBytes:edges32 length:sizeof(edges32) atIndex:4];
        [encoder setBytes:&d.n length:sizeof(d.n) atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(d.n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        demand(command.status==MTLCommandBufferStatusCompleted,"placement command failed");
        demand(command.GPUEndTime>command.GPUStartTime,"placement timestamps invalid");
        return @{@"placement_wall_ms":@(ms(start)),
            @"placement_gpu_ms":@((command.GPUEndTime-command.GPUStartTime)*1000.)};
    };
    auto start=Clock::now();
    xl_validation::GPUCachedValidator cached(device,source("bench/native-metal-xl/compare_blocks.metal"),workers);
    cached.resetContext(d.ref,d.layer,edges64);
    const double contextSetup=ms(start);
    xl_validation::ParallelValidator full(workers);
    place();
    auto first=cached.validate(output,count);
    const Check oracle=validate((const float*)output.contents,count,d.ref,d.layer);
    demand(identical(historical(first.certificate.check),oracle),"first certification disagrees");
    demand(oracle.pass,"initial placement failed original checks");
    NSMutableDictionary* initial=cacheJSON(first);
    initial[@"context_copy_allocation_compile_ms"]=@(contextSetup);
    initial[@"all_fields_equal_serial_oracle"]=@YES;

    if(controls) {
        demand(d.n>=8193 && d.n<100000,"controls require 8193..99999 markers");
        std::vector<float> baseline(count); memcpy(baseline.data(),output.contents,count*4);
        NSMutableArray* cases=[NSMutableArray array];
        auto check=[&](const char* name){
            auto r=cached.validate(output,count);
            Check expected=validate((const float*)output.contents,count,d.ref,d.layer);
            demand(identical(historical(r.certificate.check),expected),std::string("control mismatch: ")+name);
            auto entry=cacheJSON(r);entry[@"case"]=ns(name);entry[@"all_fields_equal_serial_oracle"]=@YES;
            [cases addObject:entry]; return r;
        };
        demand(check("identical_repeat").certificate.rechecked_markers==0,"identical output rechecked");
        auto bits=(uint32_t*)output.contents; auto floats=(float*)output.contents;
        bits[count-1]^=1u; demand(check("last_word_partial_block").certificate.rechecked_blocks==1,"tail not checked");
        bits[count-1]^=1u;check("restore_last_word");
        for(const auto& mutation:std::vector<std::pair<const char*,float>>{
            {"nan",std::numeric_limits<float>::quiet_NaN()},
            {"positive_infinity",std::numeric_limits<float>::infinity()},
            {"negative_infinity",-std::numeric_limits<float>::infinity()},
            {"over_tolerance",2.f},{"subnormal",std::numeric_limits<float>::denorm_min()},
            {"negative_zero",-0.f}}) {
            floats[3*4096]=mutation.second;check(mutation.first);check("repeat_changed_value");
            floats[3*4096]=baseline[3*4096];check("restore_changed_value");
        }
        floats[0]=floats[1]=floats[2]=0;check("zero_radius");
        memcpy(output.contents,baseline.data(),count*4);check("restore_zero_radius");
        floats[2]=1.f;check("outside_band");
        memcpy(output.contents,baseline.data(),count*4);check("restore_band");
        for(size_t i=0;i<d.n;i+=4096) bits[3*i]^=1u;
        auto all=check("every_block_changed");demand(all.certificate.rechecked_markers==d.n,"not all blocks rechecked");
        bool rejected=false;try{cached.validate(output,count-1);}catch(const std::invalid_argument&){rejected=true;}
        demand(rejected,"short count accepted");
        demand(check("full_after_count_rejection").certificate.rechecked_markers==d.n,"count rejection did not invalidate");
        cached.invalidate();demand(check("explicit_invalidation").certificate.rechecked_markers==d.n,"explicit invalidation");
        d.ref[0]+=0.01;cached.resetContext(d.ref,d.layer,edges64);
        demand(check("reference_replacement").certificate.rechecked_markers==d.n,"reference replacement not invalidated");
        jsonfile(argv[2],@{@"experiment":@"GPU full certificate controls",@"n":@(d.n),@"cases":cases,
            @"wrong_count_rejected":@YES,@"initial":initial,@"pass":@YES});
        printf("GPU certificate controls passed: %lu cases\n",(unsigned long)cases.count);return 0;
    }

    xl_validation::SparseOptions sparseOptions;
    sparseOptions.sample_markers=65536;
    sparseOptions.seed=20261114;
    start=Clock::now();auto sparsePlan=xl_validation::makeSparsePlan(d.n,sparseOptions);
    const double planMs=ms(start);
    if(runtime) {
        NSMutableArray* runs=[NSMutableArray array];
        for(unsigned iteration=0;iteration<iterations;iteration++) {@autoreleasepool {
            auto begin=Clock::now();auto placement=place();auto checkStart=Clock::now();
            NSMutableDictionary* entry;
            if(runtimeMode=="full-gpu") entry=cacheJSON(cached.validate(output,count));
            else if(runtimeMode=="full-cpu") {
                auto c=full.validate((const float*)output.contents,count,d.ref,d.layer,edges64);
                demand(c.pass,"runtime full CPU gate failed");
                entry=[@{@"mode":@"full_cpu",@"full_coverage":@YES,
                    @"validation":checkJSON(historical(c)),@"validation_ms":@(ms(checkStart))}mutableCopy];
            } else {
                double planning=0;
                if(runtimeMode=="sparse-rotating") {
                    auto planningStart=Clock::now();sparseOptions.iteration=iteration;
                    sparsePlan=xl_validation::makeSparsePlan(d.n,sparseOptions);planning=ms(planningStart);
                }
                auto c=xl_validation::validateSparse((const float*)output.contents,count,d.ref,d.layer,edges64,sparsePlan);
                demand(c.check.pass,"runtime sampled gate failed");
                entry=[@{@"mode":ns(runtimeMode),@"effective_validation_mode":ns(xl_validation::validationModeName(c.mode)),
                    @"full_coverage":@(c.covers_all_markers),@"checked_markers":@(c.checked_markers),
                    @"checked_coordinates":@(c.checked_coordinates),@"coverage":@(c.coverage),
                    @"sample_seed":@(c.seed),@"sample_iteration":@(c.iteration),
                    @"sample_planning_ms":@(planning),@"sample_plan_reused":@(runtimeMode=="sparse"),
                    @"validation":checkJSON(historical(c.check)),@"validation_including_planning_ms":@(ms(checkStart))}mutableCopy];
            }
            demand([entry[@"validation"][@"pass"]boolValue],"runtime validation gate failed");
            entry[@"compute_and_validation_ms"]=@(ms(begin));entry[@"placement"]=placement;
            entry[@"iteration"]=@(iteration);[runs addObject:entry];
        }}
        // One final audit, not an unreported serial audit after every iteration.
        start=Clock::now();auto last=validate((const float*)output.contents,count,d.ref,d.layer);
        double auditMs=ms(start);demand(last.pass,"final full runtime audit failed");
        jsonfile(argv[2],@{@"experiment":@"native runtime validation mode",@"n":@(d.n),@"device":device.name,
            @"workers":@(workers),@"runtime_mode":ns(runtimeMode),@"initial_certification":initial,
            @"runs":runs,@"initial_sample_planning_ms":@(planMs),@"final_full_audit":checkJSON(last),
            @"final_full_audit_ms":@(auditMs),@"independent_serial_audit_each_iteration":@NO,
            @"note":@"Initial and final full audits are explicit. Full GPU mode covers every output word each iteration and exactly rechecks changed blocks. Sparse modes cover only recorded sampled coordinates; passing sampled checks cannot establish correctness of unchecked intermediate outputs. Rotating sample planning is included in each repeated clock."});
        return 0;
    }
    NSMutableArray* trials=[NSMutableArray array];
    std::mt19937 random(20261114);std::vector<unsigned> order={0,1,2};
    for(unsigned warmup=0;warmup<3;warmup++) {place();cached.validate(output,count);full.validate((const float*)output.contents,count,d.ref,d.layer,edges64);}
    for(unsigned round=0;round<10;round++) {
        std::shuffle(order.begin(),order.end(),random);
        for(unsigned mode:order) {@autoreleasepool {
            auto begin=Clock::now();auto placement=place();start=Clock::now();
            NSMutableDictionary* trial;Check result;
            if(mode==0) {
                result=historical(full.validate((const float*)output.contents,count,d.ref,d.layer,edges64));
                trial=[@{@"mode":@"full_cpu",@"validation_ms":@(ms(start)),@"validation":checkJSON(result),@"full_coverage":@YES} mutableCopy];
            } else if(mode==1) {
                auto r=cached.validate(output,count);result=historical(r.certificate.check);trial=cacheJSON(r);
            } else {
                auto r=xl_validation::validateSparse((const float*)output.contents,count,d.ref.data(),d.ref.size(),d.layer.data(),d.n,edges64,sparsePlan);
                result=historical(r.check);
                trial=[@{@"mode":@"sparse_cpu",@"validation_ms":@(ms(start)),@"validation":checkJSON(result),
                    @"full_coverage":@(r.covers_all_markers),@"checked_markers":@(r.checked_markers),
                    @"checked_coordinates":@(r.checked_coordinates),@"total_markers":@(d.n),
                    @"coverage":@(r.coverage),@"covers_all_markers":@(r.covers_all_markers),
                    @"sample_plan_reused":@YES,@"sample_iteration":@(r.iteration),
                    @"effective_validation_mode":ns(xl_validation::validationModeName(r.mode)),
                    @"counts_valid":@(r.counts_valid),@"requested_markers":@(r.requested_markers),
                    @"boundary_markers":@(r.boundary_markers),@"sample_seed":@(r.seed)} mutableCopy];
            }
            const double combined=ms(begin);
            auto exact=validate((const float*)output.contents,count,d.ref,d.layer);
            demand(exact.pass&&result.pass,"accuracy gate failure");
            bool fullCoverage=[trial[@"full_coverage"]boolValue];
            if(fullCoverage)demand(identical(result,exact),"full checker mismatch");
            trial[@"round"]=@(round);trial[@"placement"]=placement;
            trial[@"compute_and_validation_ms"]=@(combined);
            trial[@"independent_full_serial_audit_pass"]=@(exact.pass);
            trial[@"all_fields_equal_serial_oracle"]=fullCoverage?@YES:(id)[NSNull null];
            [trials addObject:trial];printf("round=%u mode=%u check_ms=%.4f combined_ms=%.4f\n",round,mode,[trial[@"validation_ms"]doubleValue],combined);fflush(stdout);
        }}
    }
    NSMutableArray* changes=[NSMutableArray array];
    for(unsigned scenario=0;scenario<2;scenario++) {
        place();cached.validate(output,count);
        for(unsigned round=0;round<10;round++) {
            auto bits=(uint32_t*)output.contents;
            if(scenario==0){bits[0]^=1u;bits[3ull*(d.n/2)]^=1u;bits[count-1]^=1u;}
            else for(size_t i=0;i<d.n;i+=4096)bits[3*i]^=1u;
            auto r=cached.validate(output,count);
            auto exact=validate((const float*)output.contents,count,d.ref,d.layer);
            demand(identical(historical(r.certificate.check),exact),"changed output mismatch");
            auto trial=cacheJSON(r);trial[@"scenario"]=scenario==0?@"three_changed_words":@"every_block_changed";
            trial[@"round"]=@(round);trial[@"all_fields_equal_serial_oracle"]=@YES;
            [changes addObject:trial];
        }
    }
    jsonfile(argv[2],@{@"schema":@1,@"experiment":@"full GPU certificate and opt-in sparse validation",
        @"n":@(d.n),@"device":device.name,@"workers":@(workers),@"initial_certification":initial,
        @"trials":trials,@"changed_output_trials":changes,@"requested_gpu_buffer_bytes":@(gpuBytes),
        @"metal_allocated_bytes":@(device.currentAllocatedSize),@"recommended_working_set_bytes":@(device.recommendedMaxWorkingSetSize),
        @"sparse_plan_creation_ms":@(planMs),@"sparse_seed":@(sparseOptions.seed),@"sparse_iteration":@(sparseOptions.iteration),
        @"notes":@[@"Full GPU mode compares every output bit to a complete CPU-certified snapshot; exact original CPU checks recalculate changed blocks. It does not perform FP64 arithmetic on the GPU.",
          @"Context owns reference/layers/edges. Initial certification and all-changed paths are reported separately. Unchanged deterministic outputs are the favorable reuse case.",
          @"Sparse is explicit reduced coverage and cannot certify unchecked coordinates. No finite sample proves all possible inputs.",
          @"All measured full results are independently compared bit-for-bit to the serial oracle outside the reported timer. Sparse timed outputs also receive an independent full serial audit outside that timer.",
          @"Input loading, context setup/allocation, sample planning and independent serial audit are excluded from repeated trial clocks. Initial setup and sample planning have separate clocks.",
          @"Changed-output mutations occur before the checker-only timer; the GPU compare/copy, completion fence, exact CPU rechecks and reduction are included."]});
    return 0;
} catch(const std::exception& error) {fprintf(stderr,"%s\n",error.what());return 1;} }}
