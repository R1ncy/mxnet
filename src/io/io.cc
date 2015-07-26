#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <string>
#include <vector>
#include <dmlc/logging.h>
#include <mshadow/tensor.h>
#include "mxnet/io.h"
#include "iter_mnist-inl.h"
#include "../utils/random.h"

namespace mxnet	{
  IIterator<DataBatch> *CreateIterator(const std::vector< std::pair<std::string, std::string> > &cfg) {
  size_t i = 0;
  IIterator<DataBatch> *it = NULL;
  for (; i < cfg.size(); ++i) {
    const char *name = cfg[i].first.c_str();
    const char *val  = cfg[i].second.c_str();
    if (!strcmp(name, "iter")) {
      if (!strcmp(val, "mnist")) {
		CHECK(it == NULL) << "mnist cannot chain over other iterator";
		it = new MNISTIterator(); continue;
	  }
      utils::Error("unknown iterator type %s", val);
    }
    if (it != NULL) {
      it->SetParam(name, val);
    }
  }
  CHECK(it != NULL) << "must specify iterator by iter=itername";
  return it;
}
} // namespace mxnet
