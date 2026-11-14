#pragma once
// Full-output CPU validation. Compile without -ffast-math. No tolerance,
// precision, finite-value, coordinate-count, radius, or band gate is relaxed.
// The caller must synchronize GPU completion before exposing a shared buffer.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace xl_validation {
struct Check {
  size_t expected=0, actual=0, nonfinite=0, error_count=0, outside=0;
  double max_error=0, max_radius_error=0;
  bool pass=false;
};

class ParallelValidator {
  struct alignas(128) Partial {
    size_t nonfinite=0, error_count=0, outside=0;
    double max_error=0, max_radius_error=0;
  };
  std::vector<std::thread> threads_;
  std::vector<Partial> partial_;
  std::mutex mutex_, caller_mutex_;
  std::condition_variable wake_, done_;
  std::atomic<size_t> next_{0};
  size_t generation_=0, remaining_=0, n_=0, chunk_;
  bool stopping_=false;
  const float* out_=nullptr;
  const double* ref_=nullptr;
  const int32_t* layer_=nullptr;
  const double* edges_=nullptr;

  void worker(size_t index) {
    size_t seen=0;
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock,[&]{return stopping_ || seen!=generation_;});
      if(stopping_) return;
      seen=generation_;
      lock.unlock();
      Partial p;
      for (;;) {
        const size_t begin=next_.fetch_add(chunk_,std::memory_order_relaxed);
        if(begin>=n_) break;
        const size_t end=std::min(begin+chunk_,n_);
        for(size_t i=begin;i<end;i++) {
          // Arithmetic and predicates match the original serial checker exactly.
          // Fusing its coordinate and radius scans saves a second output read.
          for(size_t k=0;k<3;k++) {
            const size_t c=3*i+k;
            if(!std::isfinite(out_[c]) || !std::isfinite(ref_[c])) {
              p.nonfinite++; continue;
            }
            const double e=std::abs(double(out_[c])-ref_[c]);
            p.max_error=std::max(p.max_error,e);
            p.error_count += e>5e-4;
          }
          const double x=out_[3*i], y=out_[3*i+1], z=out_[3*i+2];
          const double r=std::hypot(x,y,z), h=z/(r?r:1);
          const int l=layer_[i];
          if(!std::isfinite(r) || r==0 || l<0 || l>=24 ||
             !(h<=edges_[l] && h>=edges_[l+1])) p.outside++;
          if(std::isfinite(r)) p.max_radius_error=std::max(p.max_radius_error,std::abs(r-1.));
        }
      }
      partial_[index]=p;
      lock.lock();
      if(--remaining_==0) done_.notify_one();
    }
  }
public:
  // 262144 markers/chunk keeps scheduling overhead negligible while permitting
  // heterogeneous performance/efficiency cores to claim work dynamically.
  explicit ParallelValidator(unsigned workers, size_t chunkMarkers=262144)
    : chunk_(chunkMarkers) {
    if(workers==0 || workers>256 || chunk_==0) throw std::invalid_argument("validator workers/chunk");
    partial_.resize(workers);
    try {
      for(unsigned i=0;i<workers;i++) threads_.emplace_back([this,i]{worker(i);});
    } catch(...) {
      { std::lock_guard<std::mutex> lock(mutex_); stopping_=true; }
      wake_.notify_all();
      for(auto& t:threads_) t.join();
      throw;
    }
  }
  ParallelValidator(const ParallelValidator&)=delete;
  ParallelValidator& operator=(const ParallelValidator&)=delete;
  ~ParallelValidator() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_=true; }
    wake_.notify_all();
    for(auto& t:threads_) t.join();
  }
  Check validate(const float* output,size_t actualCoordinates,
                 const double* reference,size_t referenceCoordinates,
                 const int32_t* layers,size_t markers,const double* edges) {
    // Calls on one validator serialize; different instances are independent.
    std::lock_guard<std::mutex> caller(caller_mutex_);
    Check c; c.expected=3*markers; c.actual=actualCoordinates;
    if(actualCoordinates!=c.expected || referenceCoordinates!=c.expected) return c;
    if(markers==0) { c.pass=true; return c; }
    std::unique_lock<std::mutex> lock(mutex_);
    out_=output; ref_=reference; layer_=layers; n_=markers; edges_=edges;
    next_=0; remaining_=threads_.size(); ++generation_;
    wake_.notify_all(); done_.wait(lock,[&]{return remaining_==0;});
    for(const auto& p:partial_) {
      c.nonfinite+=p.nonfinite; c.error_count+=p.error_count; c.outside+=p.outside;
      c.max_error=std::max(c.max_error,p.max_error);
      c.max_radius_error=std::max(c.max_radius_error,p.max_radius_error);
    }
    c.pass=c.nonfinite==0 && c.error_count==0 && c.outside==0;
    return c;
  }
  Check validate(const float* output,size_t actualCoordinates,
                 const std::vector<double>& reference,const std::vector<int32_t>& layers,
                 const double* edges) {
    return validate(output,actualCoordinates,reference.data(),reference.size(),layers.data(),layers.size(),edges);
  }
};
} // namespace xl_validation
