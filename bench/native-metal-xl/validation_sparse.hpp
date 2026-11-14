#pragma once
// Explicit, opt-in sampled CPU validation. A sparse pass describes ONLY the
// sampled markers; it is not a full-output certificate. Compile without
// -ffast-math, and synchronize GPU completion before reading its output.
// All original double-precision arithmetic, std::hypot, and gates are retained
// for every sampled point. Count checks always cover the complete input sizes.
#include "validation_parallel.hpp"
#include <limits>
#include <unordered_set>

namespace xl_validation {

enum class ValidationMode { Full, Sparse };
inline const char* validationModeName(ValidationMode mode) {
  return mode==ValidationMode::Full ? "full" : "sparse";
}

struct SparseOptions {
  size_t sample_markers=65536;
  size_t full_below_markers=65536;
  uint64_t seed=0x415155494c41584cULL;
  uint64_t iteration=0;
};

class SparsePlan {
  size_t markers_, requested_, boundary_markers_;
  uint64_t seed_, iteration_;
  ValidationMode mode_;
  std::vector<size_t> indices_;
  SparsePlan(size_t markers,const SparseOptions& options,ValidationMode mode,
             size_t boundaries,std::vector<size_t> indices)
    : markers_(markers),requested_(options.sample_markers),boundary_markers_(boundaries),
      seed_(options.seed),iteration_(options.iteration),mode_(mode),indices_(std::move(indices)) {}
  friend SparsePlan makeSparsePlan(size_t,const SparseOptions&);
public:
  // No mutable index access: plans cannot acquire duplicate/out-of-range points.
  size_t total_markers() const { return markers_; }
  size_t requested_markers() const { return requested_; }
  size_t boundary_markers() const { return boundary_markers_; }
  size_t checked_markers() const { return mode_==ValidationMode::Full ? markers_ : indices_.size(); }
  uint64_t seed() const { return seed_; }
  uint64_t iteration() const { return iteration_; }
  ValidationMode mode() const { return mode_; }
  // Full plans have implicit indices [0,N), so constructing one is O(1).
  const std::vector<size_t>& indices() const { return indices_; }
};

namespace sparse_detail {
inline uint64_t mix(uint64_t x) {
  x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
  x=(x^(x>>27))*0x94d049bb133111ebULL;
  return x^(x>>31);
}
class Random {
  uint64_t state_;
public:
  Random(uint64_t seed,uint64_t iteration)
    : state_(mix(seed)^mix(iteration+0x9e3779b97f4a7c15ULL)) {}
  uint64_t next() { state_+=0x9e3779b97f4a7c15ULL; return mix(state_); }
  // Rejection avoids modulo bias. Fixed integer operations avoid depending on
  // a standard library's uniform_int_distribution implementation.
  uint64_t below(uint64_t bound) {
    const uint64_t threshold=(uint64_t(0)-bound)%bound;
    for (;;) { const uint64_t x=next(); if(x>=threshold) return x%bound; }
  }
};
inline void neighbors(std::vector<size_t>& points,size_t at,size_t n) {
  if(at>0) points.push_back(at-1);
  points.push_back(at);
  if(at<n-1) points.push_back(at+1);
}
inline void strideBoundaries(std::vector<size_t>& points,size_t n,size_t stride) {
  const size_t count=(n-1)/stride;
  constexpr size_t cap=1024;
  const size_t selected=std::min(count,cap);
  for(size_t j=0;j<selected;j++) {
    // Cover every boundary up to the cap; beyond it choose evenly spaced
    // boundary ordinals, including first/last, without overflowing size_t.
    const size_t ordinal=count<=cap ? j+1 :
      1+j*((count-1)/(cap-1))+(j*((count-1)%(cap-1)))/(cap-1);
    neighbors(points,ordinal*stride,n);
  }
}
inline std::vector<size_t> boundaries(size_t n) {
  std::vector<size_t> points;
  if(n==0) return points;
  neighbors(points,0,n); neighbors(points,n-1,n);
  for(size_t q=1;q<4;q++) {
    // floor(q*N/4), with no overflowing multiplication.
    neighbors(points,(n/4)*q+((n%4)*q)/4,n);
  }
  strideBoundaries(points,n,262144);
  strideBoundaries(points,n,25000000);
  std::sort(points.begin(),points.end());
  points.erase(std::unique(points.begin(),points.end()),points.end());
  return points;
}
} // namespace sparse_detail

inline SparsePlan makeSparsePlan(size_t markers,const SparseOptions& options) {
  static_assert(sizeof(size_t)<=sizeof(uint64_t),"sampling supports at most 64-bit indices");
  if(options.sample_markers==0) throw std::invalid_argument("sample_markers must be positive");
  if(markers>std::numeric_limits<size_t>::max()/3) throw std::invalid_argument("marker coordinate count overflow");
  // Small datasets are always checked in full, even if the caller lowers the
  // configurable cutoff. Default cutoff is 65536; hard minimum is 4096.
  constexpr size_t minimum_full_markers=4096;
  if(markers<=std::max(options.full_below_markers,minimum_full_markers) || options.sample_markers>=markers)
    return SparsePlan(markers,options,ValidationMode::Full,0,{});

  auto anchors=sparse_detail::boundaries(markers);
  const size_t boundary_count=anchors.size();
  // Mandatory anchors can make actual coverage larger than requested.
  const size_t count=std::max(options.sample_markers,boundary_count);
  if(count>=markers) return SparsePlan(markers,options,ValidationMode::Full,boundary_count,{});
  const size_t random_count=count-boundary_count,available=markers-boundary_count;
  sparse_detail::Random random(options.seed,options.iteration);
  // Floyd's algorithm samples complement ranks uniformly without replacement;
  // unlike retry-until-unique sampling, it remains bounded near full coverage.
  std::unordered_set<size_t> chosen;
  chosen.reserve(random_count);
  for(size_t j=available-random_count;j<available;j++) {
    const size_t candidate=size_t(random.below(uint64_t(j)+1));
    if(!chosen.insert(candidate).second) chosen.insert(j);
  }
  std::vector<size_t> ranks(chosen.begin(),chosen.end());
  std::sort(ranks.begin(),ranks.end());
  std::vector<size_t> indices=anchors;
  indices.reserve(count);
  size_t skipped=0;
  for(size_t rank:ranks) {
    size_t at=rank+skipped;
    while(skipped<boundary_count && anchors[skipped]<=at) { ++skipped; ++at; }
    indices.push_back(at);
  }
  std::sort(indices.begin(),indices.end());
  return SparsePlan(markers,options,ValidationMode::Sparse,boundary_count,std::move(indices));
}

struct SparseCheck {
  Check check;
  ValidationMode mode=ValidationMode::Sparse;
  size_t total_markers=0,requested_markers=0,boundary_markers=0;
  size_t checked_markers=0,checked_coordinates=0;
  uint64_t seed=0,iteration=0;
  bool counts_valid=false,covers_all_markers=false;
  double coverage=0;
};

inline SparseCheck validateSparse(const float* output,size_t actualCoordinates,
                                 const double* reference,size_t referenceCoordinates,
                                 const int32_t* layers,size_t markers,const double* edges,
                                 const SparsePlan& plan) {
  if(markers!=plan.total_markers()) throw std::invalid_argument("sparse plan marker count mismatch");
  SparseCheck result;
  result.mode=plan.mode(); result.total_markers=markers;
  result.requested_markers=plan.requested_markers(); result.boundary_markers=plan.boundary_markers();
  result.seed=plan.seed(); result.iteration=plan.iteration();
  Check& c=result.check;
  c.expected=3*markers; c.actual=actualCoordinates;
  if(actualCoordinates!=c.expected || referenceCoordinates!=c.expected) return result;
  result.counts_valid=true;
  if(markers && (!output || !reference || !layers || !edges)) throw std::invalid_argument("null validation buffer");
  auto point=[&](size_t i) {
    for(size_t k=0;k<3;k++) {
      const size_t at=3*i+k;
      if(!std::isfinite(output[at]) || !std::isfinite(reference[at])) { c.nonfinite++; continue; }
      const double error=std::abs(double(output[at])-reference[at]);
      c.max_error=std::max(c.max_error,error);
      c.error_count+=error>5e-4;
    }
    const double x=output[3*i],y=output[3*i+1],z=output[3*i+2];
    const double r=std::hypot(x,y,z),h=z/(r?r:1);
    const int l=layers[i];
    if(!std::isfinite(r) || r==0 || l<0 || l>=24 || !(h<=edges[l] && h>=edges[l+1])) c.outside++;
    if(std::isfinite(r)) c.max_radius_error=std::max(c.max_radius_error,std::abs(r-1.));
  };
  if(plan.mode()==ValidationMode::Full) { for(size_t i=0;i<markers;i++) point(i); }
  else { for(size_t i:plan.indices()) point(i); }
  result.checked_markers=plan.checked_markers(); result.checked_coordinates=3*result.checked_markers;
  result.covers_all_markers=result.checked_markers==markers;
  result.coverage=markers ? double(result.checked_markers)/double(markers) : 1.;
  c.pass=c.nonfinite==0 && c.error_count==0 && c.outside==0;
  return result;
}

inline SparseCheck validateSparse(const float* output,size_t actualCoordinates,
                                 const std::vector<double>& reference,const std::vector<int32_t>& layers,
                                 const double* edges,const SparsePlan& plan) {
  return validateSparse(output,actualCoordinates,reference.data(),reference.size(),layers.data(),layers.size(),edges,plan);
}
} // namespace xl_validation
