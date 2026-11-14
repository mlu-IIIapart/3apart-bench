#pragma once
// Exact historical validation with per-block summary reuse. Compile without
// fast math or floating-point contraction. GPU completion and exact changed-
// block detection against the PRECEDING validated snapshot are caller duties.
// A zero flag is a promise that every output bit in that block is unchanged;
// this class never substitutes a checksum, sampling, or tolerance comparison.
#include "validation_parallel.hpp"
#include <array>
#include <limits>
#include <utility>

namespace xl_validation {

struct CachedBlockResult {
  Check check;
  size_t rechecked_markers=0, reused_markers=0;
  size_t rechecked_blocks=0, reused_blocks=0;
  uint64_t generation=0, context_generation=0;
  // Cache freshness is independent of the numerical acceptance gate.
  bool cache_valid=false;
};

class CachedBlockValidator {
public:
  static constexpr size_t block_markers=4096;
private:
  std::vector<std::thread> threads_;
  std::mutex mutex_, caller_mutex_;
  std::condition_variable wake_, done_;
  std::atomic<size_t> next_{0};
  size_t dispatch_generation_=0, remaining_=0;
  bool stopping_=false, has_context_=false, valid_=false;
  uint64_t context_generation_=0, completed_generation_=0, valid_generation_=0;
  std::vector<double> reference_;
  std::vector<int32_t> layers_;
  std::array<double,25> edges_{};
  std::vector<Check> blocks_;
  std::vector<size_t> work_;
  const float* output_=nullptr;

  void invalidate_locked() {
    valid_=false;
    valid_generation_=0;
  }

  void worker() {
    size_t seen=0;
    for(;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock,[&]{return stopping_ || seen!=dispatch_generation_;});
      if(stopping_) return;
      seen=dispatch_generation_;
      lock.unlock();
      for(;;) {
        const size_t task=next_.fetch_add(1,std::memory_order_relaxed);
        if(task>=work_.size()) break;
        const size_t b=work_[task], begin=b*block_markers;
        const size_t end=std::min(begin+block_markers,layers_.size());
        Check p;
        p.expected=p.actual=3*(end-begin);
        for(size_t i=begin;i<end;i++) {
          // Numerical operations and predicates match ParallelValidator and
          // the independent historical serial checker exactly.
          for(size_t k=0;k<3;k++) {
            const size_t c=3*i+k;
            if(!std::isfinite(output_[c]) || !std::isfinite(reference_[c])) {
              p.nonfinite++; continue;
            }
            const double e=std::abs(double(output_[c])-reference_[c]);
            p.max_error=std::max(p.max_error,e);
            p.error_count += e>5e-4;
          }
          const double x=output_[3*i], y=output_[3*i+1], z=output_[3*i+2];
          const double r=std::hypot(x,y,z), h=z/(r?r:1);
          const int l=layers_[i];
          if(!std::isfinite(r) || r==0 || l<0 || l>=24 ||
             !(h<=edges_[l] && h>=edges_[l+1])) p.outside++;
          if(std::isfinite(r)) p.max_radius_error=std::max(p.max_radius_error,std::abs(r-1.));
        }
        p.pass=p.nonfinite==0 && p.error_count==0 && p.outside==0;
        blocks_[b]=p;
      }
      lock.lock();
      if(--remaining_==0) done_.notify_one();
    }
  }

public:
  explicit CachedBlockValidator(unsigned workers=18) {
    if(workers==0 || workers>256) throw std::invalid_argument("cached validator workers");
    try {
      for(unsigned i=0;i<workers;i++) threads_.emplace_back([this]{worker();});
    } catch(...) {
      { std::lock_guard<std::mutex> lock(mutex_); stopping_=true; }
      wake_.notify_all();
      for(auto& t:threads_) t.join();
      throw;
    }
  }
  CachedBlockValidator(const CachedBlockValidator&)=delete;
  CachedBlockValidator& operator=(const CachedBlockValidator&)=delete;
  ~CachedBlockValidator() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_=true; }
    wake_.notify_all();
    for(auto& t:threads_) t.join();
  }

  // The context is owned and cannot be changed through caller aliases. Passing
  // rvalues transfers ownership; lvalues are copied once outside timed scans.
  // Reset is the only way to change reference, layers, edges, or marker count.
  void reset(std::vector<double> reference, std::vector<int32_t> layers,
             const double* edges) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    invalidate_locked();
    if(layers.size()>std::numeric_limits<size_t>::max()/3 ||
       reference.size()!=3*layers.size() || edges==nullptr)
      throw std::invalid_argument("cached validator context shape");
    std::array<double,25> owned_edges;
    std::copy_n(edges,25,owned_edges.begin());
    std::vector<Check> fresh(layers.size()/block_markers+(layers.size()%block_markers!=0));
    reference_=std::move(reference);
    layers_=std::move(layers);
    edges_=owned_edges;
    blocks_=std::move(fresh);
    work_.clear();
    ++context_generation_;
    has_context_=true;
  }

  void invalidate() {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    invalidate_locked();
  }

  CachedBlockResult validate(const float* output, size_t actualCoordinates,
                             const uint32_t* changedFlags, size_t flagCount) {
    // The output and supplied flags remain immutable for this call. Caller
    // calls serialize; the persistent workers never outlive the call's data.
    std::lock_guard<std::mutex> caller(caller_mutex_);
    if(!has_context_) throw std::logic_error("cached validator requires reset");
    CachedBlockResult r;
    r.context_generation=context_generation_;
    r.check.expected=3*layers_.size();
    r.check.actual=actualCoordinates;
    if(actualCoordinates!=r.check.expected) {
      invalidate_locked();
      return r;
    }
    if(actualCoordinates!=0 && output==nullptr) {
      invalidate_locked();
      throw std::invalid_argument("cached validator null output");
    }
    if(valid_ && (flagCount!=blocks_.size() || (flagCount && changedFlags==nullptr))) {
      invalidate_locked();
      throw std::invalid_argument("cached validator changed-flag count");
    }
    work_.clear();
    for(size_t b=0;b<blocks_.size();b++) {
      const size_t markers=std::min(block_markers,layers_.size()-b*block_markers);
      if(!valid_ || changedFlags[b]!=0) {
        work_.push_back(b); r.rechecked_blocks++; r.rechecked_markers+=markers;
      } else {
        r.reused_blocks++; r.reused_markers+=markers;
      }
    }
    if(!work_.empty()) {
      std::unique_lock<std::mutex> lock(mutex_);
      output_=output;
      next_=0; remaining_=threads_.size(); ++dispatch_generation_;
      wake_.notify_all();
      done_.wait(lock,[&]{return remaining_==0;});
    }
    for(const auto& p:blocks_) {
      r.check.nonfinite+=p.nonfinite;
      r.check.error_count+=p.error_count;
      r.check.outside+=p.outside;
      r.check.max_error=std::max(r.check.max_error,p.max_error);
      r.check.max_radius_error=std::max(r.check.max_radius_error,p.max_radius_error);
    }
    r.check.pass=r.check.nonfinite==0 && r.check.error_count==0 && r.check.outside==0;
    valid_=true;
    valid_generation_=++completed_generation_;
    r.generation=valid_generation_;
    r.cache_valid=true;
    return r;
  }

  CachedBlockResult validate(const float* output, size_t actualCoordinates,
                             const std::vector<uint32_t>& changedFlags) {
    return validate(output,actualCoordinates,changedFlags.data(),changedFlags.size());
  }
};
} // namespace xl_validation
