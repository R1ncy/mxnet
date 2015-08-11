// Copyright (c) 2015 by Contributors
#include "mxnet/context.h"
#include "dmlc/logging.h"
#if MXNET_USE_CUDA
#include "mxnet/cuda_utils.h"
#endif  // MXNET_USE_CUDA

#include <vector>

#if MXNET_USE_CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif  // MXNET_USE_CUDA

#if MXNET_USE_CUDNN
#include <cudnn.h>
#endif  // MXNET_USE_CUDNN

#define DEFAULT_NUM_STREAMS 4

using namespace std;

namespace mxnet {

class ContextManagerImpl : public ContextManager {
 public:
  ContextManagerImpl() {
    // initialize contexts
#if MXNET_USE_CUDA
    int num_devices;
    CUDA_CALL(cudaGetDeviceCount(&num_devices));
    for (int i = 0; i < num_devices; ++i) {
      GpuRunContext gctx;
      for (int j = 0; j < DEFAULT_NUM_STREAMS; ++j) {
        cudaStream_t *stream = new cudaStream_t;
        CUDA_CALL(cudaStreamCreate(stream));
        cublasHandle_t *cublas_handle = new cublasHandle_t;
        CUBLAS_CALL(cublasCreate(cublas_handle));
        CUBLAS_CALL(cublasSetStream(*cublas_handle, *stream));
#if MXNET_USE_CUDNN
        cudnnHandle_t *cudnn_handle = new cudnnHandle_t;
        CUDNN_CALL(cudnnCreate(cudnn_handle));
        CUDNN_CALL(cudnnSetStream(*cudnn_handle, *stream));
        gctx.stream_ctx.push_back(RunContext{stream, cublas_handle, cudnn_handle});
#else  // MXNET_USE_CUDNN
        gctx.stream_ctx.push_back(RunContext{stream, cublas_handle, nullptr});
#endif  // MXNET_USE_CUDNN
      }
      gpu_ctx_.push_back(gctx);
    }
#endif  // MXNET_USE_CUDA
    cpu_ctx_ = {nullptr, nullptr, nullptr};
  }
  const RunContext* GetRunContext(const Context &ctx, int streamid) const {
    switch (ctx.dev_mask) {
      case mshadow::cpu::kDevMask:
        return &cpu_ctx_;
      case mshadow::gpu::kDevMask:
        return &gpu_ctx_.at(ctx.dev_id).stream_ctx.at(streamid);
      default:
        LOG(FATAL) << "unknown dev mask:" << ctx.dev_mask;
    }
    return nullptr;
  }
  size_t NumGpus() const {
    return gpu_ctx_.size();
  }
  size_t NumStreams(int gpuid) const {
    return gpu_ctx_.at(gpuid).stream_ctx.size();
  }

 private:
  struct GpuRunContext {
    vector<RunContext> stream_ctx;
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
