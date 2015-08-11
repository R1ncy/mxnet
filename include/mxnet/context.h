// Copyright (c) 2015 by Contributors
#ifndef MXNET_CONTEXT_H_
#define MXNET_CONTEXT_H_

#include <mshadow/tensor.h>

namespace mxnet {

/*! \brief context information about the execution enviroment */
struct Context {
  /*! \brief the device type we run the op can be cpu::kDevMask or gpu::kDevMask */
  int dev_mask;
  /*! \brief device id we are going to run it on */
  int dev_id;
  /*! \brief constructor */
  Context() : dev_mask(mshadow::cpu::kDevMask), dev_id(0) {}
  /*!
   * \brief constructor of context
   * \param dev_mask the device mask
   * \param dev_id the device id
   */
  Context(int dev_mask, int dev_id)
      : dev_mask(dev_mask), dev_id(dev_id) {}
  /*!
   * \brief check if current context equals another one
   * \param b another context to compare
   * \return whether dev mask and id are same
   */
  inline bool operator==(const Context &b) const {
    return dev_mask == b.dev_mask && dev_id == b.dev_id;
  }
};

/*!
 * \brief execution context provides the information needed
 *  in runtime to actually execute the operation
 */
struct RunContext {
  /*!\brief the stream of the device,
   * can be NULL or Stream<gpu>* in GPU mode */
  void *stream;
  /*!\brief cublas handler bound to the same stream above*/
  void *cublas_handler;
  /*!\brief cudnn handler bound to the same stream above*/
  void *cudnn_handler;
};

/*! \brief A manager class that takes in charge of all computing resources
 * such as GPU contexts and handlers
 */
class ContextManager {
 public:
  /*!
   * \brief get the runtime context from the given context and streamid
   * \param ctx the context information (i.e, what device, which card)
   * \param streamid which stream 
   * \return the pointer to the runtime context
   */
  const RunContext* GetRunContext(const Context &ctx, int streamid) const;
  /*!
   * \brief get the number of gpus available for mxnet
   * \return number of gpus
   */
  size_t NumGpus() const;
  /*!
   * \brief get the number of streams available for the given card id
   * \param gpuid gpu card id
   * \return number of streams
   */
  size_t NumStreams(int gpuid) const;
  /*!
   * \brief get the context manager singleton instance
   * \return context manager instance
   */
  static ContextManager* get();
};

}  // namespace mxnet

#endif  // MXNET_CONTEXT_H_
