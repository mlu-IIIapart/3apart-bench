#include <metal_stdlib>
using namespace metal;

// Equality is on raw bits, including signed zero and NaN payloads. No hashing,
// floating-point arithmetic or tolerance approximation occurs in this kernel.
// One group handles one 4096-marker block; the final block may be shorter.
kernel void compare_blocks(device const uint* output [[buffer(0)]],
                           device uint* snapshot [[buffer(1)]],
                           device uint* changed [[buffer(2)]],
                           constant uint& markers [[buffer(3)]],
                           uint block [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_threadgroup]]) {
    constexpr uint blockMarkers = 4096;
    constexpr uint groupThreads = 256;
    const ulong begin = ulong(block) * blockMarkers * 3;
    const ulong end = min(begin + ulong(blockMarkers) * 3, ulong(markers) * 3);
    uint differs = 0;
    for (ulong word = begin + lane; word < end; word += groupThreads) {
        const uint current = output[word];
        differs |= current != snapshot[word];
    }
    threadgroup uint flags[groupThreads];
    flags[lane] = differs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = groupThreads / 2; stride; stride /= 2) {
        if (lane < stride) flags[lane] |= flags[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) changed[block] = flags[0];
    // Identical blocks need no snapshot write. Changed blocks copy every word,
    // after the complete comparison, so CPU summaries describe one generation.
    if (flags[0]) {
        for (ulong word = begin + lane; word < end; word += groupThreads)
            snapshot[word] = output[word];
    }
}
