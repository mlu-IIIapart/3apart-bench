#pragma once
#import <Metal/Metal.h>
#include "validation_cached_blocks.hpp"
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace xl_validation {
// The context is copied into the CPU certificate. Call resetContext when the
// reference, layers, edges or checker semantics change. The caller must keep
// output exclusively owned and unmodified for the entire synchronous check.
// Complete the producer command before calling validate: this object has its
// own command queue and does not infer dependencies on another producer queue.
// GPU completion precedes every CPU read. A partial/error operation invalidates
// all certificates; a failed accuracy result is retained as a failed result.
class GPUCachedValidator {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> pipeline_;
    id<MTLBuffer> snapshot_, flags_;
    CachedBlockValidator cpu_;
    uint32_t markers_ = 0;
    size_t blocks_ = 0;
    std::mutex caller_;
public:
    struct Result {
        decltype(std::declval<CachedBlockValidator>().validate(nullptr,0,nullptr,0)) certificate;
        double gpu_device_ms = 0;
        double gpu_wall_ms = 0;
        double cpu_update_and_reduce_ms = 0;
        double total_ms = 0;
    };
    GPUCachedValidator(id<MTLDevice> device, NSString* source, unsigned workers)
        : device_(device), cpu_(workers) {
        queue_ = [device newCommandQueue];
        NSError* error = nil;
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) throw std::runtime_error(error.description.UTF8String);
        pipeline_ = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"compare_blocks"] error:&error];
        if (!queue_ || !pipeline_ || pipeline_.maxTotalThreadsPerThreadgroup < 256)
            throw std::runtime_error("GPU certificate pipeline unavailable");
    }
    void resetContext(const std::vector<double>& reference,
                      const std::vector<int32_t>& layers, const double* edges) {
        std::lock_guard<std::mutex> lock(caller_);
        cpu_.invalidate();
        if (layers.empty() || layers.size() > UINT32_MAX)
            throw std::invalid_argument("GPU certificate marker count");
        cpu_.reset(reference, layers, edges);
        markers_ = uint32_t(layers.size());
        blocks_ = (size_t(markers_) + 4095) / 4096;
        snapshot_ = [device_ newBufferWithLength:size_t(markers_)*12 options:MTLResourceStorageModeShared];
        flags_ = [device_ newBufferWithLength:blocks_*sizeof(uint32_t) options:MTLResourceStorageModeShared];
        if (!snapshot_ || !flags_) throw std::runtime_error("GPU certificate allocation failed");
        memset(snapshot_.contents, 0, snapshot_.length);
    }
    void invalidate() { std::lock_guard<std::mutex> lock(caller_); cpu_.invalidate(); }
    size_t extraGPUBytes() const { return snapshot_.length + flags_.length; }
    Result validate(id<MTLBuffer> output, size_t actualCoordinates) {
        std::lock_guard<std::mutex> lock(caller_);
        using Clock = std::chrono::steady_clock;
        auto elapsed = [](auto begin) { return std::chrono::duration<double,std::milli>(Clock::now()-begin).count(); };
        auto begin = Clock::now();
        try {
            if (!output || !snapshot_ || !flags_ || actualCoordinates != size_t(markers_)*3 ||
                output.length < actualCoordinates*sizeof(float)) {
                cpu_.invalidate();
                throw std::invalid_argument("GPU certificate output count/length");
            }
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!command || !encoder)
                throw std::runtime_error("GPU certificate command/encoder unavailable");
            [encoder setComputePipelineState:pipeline_];
            [encoder setBuffer:output offset:0 atIndex:0];
            [encoder setBuffer:snapshot_ offset:0 atIndex:1];
            [encoder setBuffer:flags_ offset:0 atIndex:2];
            [encoder setBytes:&markers_ length:sizeof(markers_) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(blocks_,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error("GPU certificate command failed");
            Result result;
            result.gpu_wall_ms = elapsed(begin);
            result.gpu_device_ms = (command.GPUEndTime-command.GPUStartTime)*1000.;
            if (!(result.gpu_device_ms > 0) || !std::isfinite(result.gpu_device_ms))
                throw std::runtime_error("GPU certificate timestamp invalid");
            auto cpuBegin = Clock::now();
            // Validate the newly copied snapshot, not a potentially divergent
            // caller pointer. All summaries and comparison history now describe
            // the exact same immutable bytes until the next validate call.
            result.certificate = cpu_.validate((const float*)snapshot_.contents,
                actualCoordinates, (const uint32_t*)flags_.contents, blocks_);
            result.cpu_update_and_reduce_ms = elapsed(cpuBegin);
            result.total_ms = elapsed(begin);
            return result;
        } catch (...) { cpu_.invalidate(); throw; }
    }
};
} // namespace xl_validation
