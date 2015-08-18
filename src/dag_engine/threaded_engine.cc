// Copyright (c) 2015 by Contributors
#include <queue>
#include <memory>
#include <tuple>
#include <utility>
#include <atomic>
#include <thread>
#include <random>

#include "dmlc/logging.h"
#include "dmlc/memory.h"
#include "mxnet/dag_engine.h"
#include "../common/memory.h"
#include "../common/spin_lock.h"
#include "../common/concurrent_blocking_queue.h"

using namespace std;

namespace mxnet {

#define DEFAULT_NUM_WORKER_THREADS 4

class ThreadedEngine : public DAGEngine {
 public:
  explicit ThreadedEngine(int numthreads = DEFAULT_NUM_WORKER_THREADS): numthreads_(numthreads) {
    for (int i = 0; i < numthreads; ++i) {
      worker_queues_.push_back(new common::ConcurrentBlockingQueue<OpDescr_*>());
      workers_.emplace_back(&ThreadedEngine::WorkerRoutine, this, i);
    }
  }
  ~ThreadedEngine() {
    for (int i = 0; i < numthreads_; ++i) {
      worker_queues_[i]->SignalForKill();
      delete worker_queues_[i];
      workers_[i].join();
    }
  }
  Operator NewOperator(AsyncFn fn,
                       const std::vector<Variable> &use_vars,
                       const std::vector<Variable> &mutate_vars) override {
    // TODO(minjie): TBD
    return nullptr;
  }
  void DeleteOperator(Operator op) override {
    // TODO(minjie): TBD
  }
  void Push(Operator op, Context exec_ctx) override {
    // TODO(minjie): TBD
  }
  void PushAsync(AsyncFn exec_fun,
            Context exec_ctx,
            const vector<Variable> &use_vars,
            const vector<Variable> &mutate_vars) override {
    shared_ptr<OpDescr_> opd(new OpDescr_{exec_fun, exec_ctx, use_vars, mutate_vars},
        [this] (OpDescr_* o) { this->OnDepsResolved(o); });
    for ( Variable v : use_vars ) {  // read
      VarDescr_* vard = static_cast<VarDescr_*>(v);  // safe to cast here
      spin_lock(&vard->lock);
      if (vard->rw < 0) {
        vard->waitings.push(make_pair(opd, DepType::kRead));
      } else {
        ++vard->rw;
      }
      spin_unlock(&vard->lock);
    }
    for ( Variable v : mutate_vars ) {  // write
      VarDescr_* vard = static_cast<VarDescr_*>(v);  // safe to cast here
      spin_lock(&vard->lock);
      if (vard->rw != 0) {
        vard->waitings.push(make_pair(opd, DepType::kWrite));
      } else {
        vard->rw = -1;
      }
      spin_unlock(&vard->lock);
    }
  }
  void Push(Fn exec_fun,
            Context exec_ctx,
            const vector<Variable> &use_vars,
            const vector<Variable> &mutate_vars) override {
    this->PushAsync([exec_fun](RunContext ctx, Callback on_complete) {
        exec_fun(ctx); on_complete();
      }, exec_ctx, use_vars, mutate_vars);
  }
  void PushDelete(Fn delete_fun, Context exec_ctx, Variable var) override {
    this->Push([delete_fun, var] (RunContext ctx) {
          delete_fun(ctx);
          delete static_cast<VarDescr_*>(var);  // TODO(minjie): use variable pool instead
        }, exec_ctx, {}, {var});
  }
  Variable NewVar() override {
    // in practice return a ptr to a cell
    // that have the info about the variable
    // use ptr directly instead of ID because this avoids an indirect mapping
    // TODO(minjie): use variable pool instead
    VarDescr_* vd = new VarDescr_;
    vd->lock = SPINLOCK_INITIALIZER;
    vd->rw = 0;
    return vd;
  }
  void WaitForVar(Variable var) override {
    // TODO(minjie): tbd
  }
  void WaitForAll() override {
    // TODO(minjie): tbd
  }

 private:
  enum class DepType {
    kRead = 0,
    kWrite,
  };
  struct OpDescr_;
  struct VarDescr_;
  struct RuntimeOp_;

  struct OpDescr_ {
    // execution function
    AsyncFn fn;
    // read dependencies
    vector<Variable> read_vars;
    // write dependencies
    vector<Variable> write_vars;
    // whether this operator needs to be deleted upon finish execution
    bool transient;
  };
  struct VarDescr_ {
    spinlock lock;
    int rw;  // a semaphore-like count
             // if rw > 0, the variable has several readers and the number
             //   means how many operators are currently reading it;
             // if rw < 0, the varaible has one writer (should be -1)
    queue<pair<RuntimeOp_*, DepType>> waitings;
  };
  struct RuntimeOp_ {
    OpDescr_* op_descr;
    atomic<int> dep_count;
    Context ctx;
  };
  void TriggerWaiting(VarDescr_* vard) {
    // ATTENTION: this function should be called with vard->lock held.
    CHECK(vard->rw == 0) << "the variable should be free during triggering";
    if (!vard->waitings.empty()) {
      // pop all reads first
      while (vard->waitings.front().second == DepType::kRead) {
        vard->waitings.pop();
        ++vard->rw;
      }
      if (vard->rw == 0) {
        // pop the next write
        vard->waitings.pop();
        vard->rw = -1;
      }
    }
  }
  void OnOpFinished(OpDescr_* opd) {
    CHECK(opd) << "completing a nullptr op!";
    for (Variable v : opd->read_vars) {
      VarDescr_* vard = static_cast<VarDescr_*>(v);  // safe to cast here
      spin_lock(&vard->lock);
      CHECK(vard->rw > 0) << "incorrect rw count (reader):" << vard->rw;
      if (--vard->rw == 0) {
        TriggerWaiting(vard);
      }
      spin_unlock(&vard->lock);
    }
    for (Variable v : opd->write_vars) {
      VarDescr_* vard = static_cast<VarDescr_*>(v);  // safe to cast here
      spin_lock(&vard->lock);
      CHECK(vard->rw == -1) << "incorrect rw count (writer):" << vard->rw;
      vard->rw = 0;
      TriggerWaiting(vard);
      spin_unlock(&vard->lock);
    }
    delete opd;  // delete the operator
  }
  RunContext GetRunContext(const Context& ctx) {
    // TODO(minjie): get the correct runtime context
    return RunContext();
  }
  void OnDepsResolved(OpDescr_* opd) {
    static default_random_engine generator;
    static uniform_int_distribution<int> distribution(0, numthreads_ - 1);
    int thrid = distribution(generator);
    // LOG(INFO) << "schedule operator " << opd << " to thread #" << thrid;
    worker_queues_[thrid]->Push(opd);
  }
  void WorkerRoutine(int thrid) {
    OpDescr_* opd = nullptr;
    while (!worker_queues_[thrid]->Pop(&opd)) {
      // LOG(INFO) << "worker thread #" << thrid << " got operator " << opd;
      opd->op(GetRunContext(opd->exec_ctx), [this, opd] () { this->OnOpFinished(opd); });
      opd = nullptr;
    }
  }

 private:
  const int numthreads_;
  vector<common::ConcurrentBlockingQueue<RuntimeOp_*>*> worker_queues_;
  vector<thread> workers_;

  dmlc::PoolAllocator<VarDescr_> var_allocator_;
  dmlc::PoolAllocator<OpDescr_> op_allocator_;
  dmlc::PoolAllocator<RuntimeOp_> runtime_op_allocator_;
};

// implements the singleton factory
DAGEngine* DAGEngine::Get() {
  static ThreadedEngine engine;
  return &engine;
}
}  // namespace mxnet
