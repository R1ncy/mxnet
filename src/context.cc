// Copyright (c) 2015 by Contributors
#include "mxnet/context.h"
#include "dmlc/logging.h"

#include <vector>

using namespace std;

namespace mxnet {

class ContextManagerImpl : public ContextManager {
 public:
  ContextManagerImpl() {
    // TODO(minjie): initialize
  }
  const RunContext* GetRunContext(const Context &ctx, int streamid) const {
    switch (ctx.dev_mask) {
      case mshadow::cpu::kDevMask:
        return &cpu_ctx_;
      case mshadow::gpu::kDevMask:
        return &gpu_ctx_.at(ctx.dev_id).stream_ctx_.at(streamid);
      default:
        LOG(FATAL) << "unknown dev mask:" << ctx.dev_mask;
    };
    return nullptr;
  }
  size_t NumGpus() const {
    return gpu_ctx_.size();
  }
  size_t NumStreams(int gpuid) const {
    return gpu_ctx_.at(gpuid).stream_ctx_.size();
  }

 private:
  struct GpuRunContext {
    vector<RunContext> stream_ctx_;
  };

 private:
  vector<GpuRunContext> gpu_ctx_;
  RunContext cpu_ctx_;
};

ContextManager* ContextManager::get() {
  static ContextManagerImpl cm;
  return &cm;
}

}  // namespace mxnet
