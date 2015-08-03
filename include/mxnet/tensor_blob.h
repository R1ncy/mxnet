/*!
 *  Copyright (c) 2015 by Contributors
 * \file tensor_blob.h
 * \brief tensor blob used to hold static memory used by
 */
#ifndef MXNET_TENSOR_BLOB_H_
#define MXNET_TENSOR_BLOB_H_
#include <mshadow/tensor.h>

namespace mxnet {

/*! \brief dynamic shape type */
typedef mshadow::TShape TShape;
/*! \brief storage container type */
typedef mshadow::TBlob TBlob;
}  // namespace mxnet
#endif  // MXNET_TENSOR_BLOB_H_
