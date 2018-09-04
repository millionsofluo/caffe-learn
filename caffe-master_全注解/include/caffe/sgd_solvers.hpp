#ifndef CAFFE_SGD_SOLVERS_HPP_
#define CAFFE_SGD_SOLVERS_HPP_

#include <string>
#include <vector>

#include "caffe/solver.hpp"

namespace caffe {
//  用SGD方法的求解器类，派生于solver
/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
 //  显示构造函数，注意在构造时调用自身的presolver（）函数
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) { PreSolve(); }
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) { PreSolve(); }
  virtual inline const char* type() const { return "SGD"; }

//  获得权值更新历史
  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
 //  求解前的准备工作
  void PreSolve();
 /*提供了一系列的learning rate方案：
fixed：lr永远不变
step：lr=baselr*gamma^{iter / stepsize}
exp：lr=baselr*gamma^{iter}
inv：lr=baselr*(1+gamma*iter)^{-power}
multistep：直接写iter在某个范围内时lr应该是多少
poly：lr=baselr*(1-\frac{iter}{maxiter})^{power}
sigmoid：lr=baselr*\frac{1}{1+e^{-gamma*(iter-stepsize)}}
这些方案各有优劣，选择自己顺手的就好。*/
  Dtype GetLearningRate();  //  得到学习速率
  virtual void ApplyUpdate();  //  应用更新
  //考虑了一些单一Batch不足以完成训练的问题，通过限制每个Batch的更新量来控制更新总量
  //  对某一权值进行归一化
  virtual void Normalize(int param_id); 
  //计算正则项的梯度了。Caffe提供两种正则方法——L2和L1，其中L2采用了标准的梯度下降方法，L1采用了sub-gradient的计算方法。
  //  对某权值进行规整化
  virtual void Regularize(int param_id); 
  //  计算某权值在特定学习速率下的更新值
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  /*ClipGradients   梯度抑制
这一步主要是对梯度值做一个限制，如果梯度值过大，那么这里就会对梯度做一个修剪，
对所有的参数乘以一个缩放因子，使得所有参数的平方和不超过参数中设定的梯度总值。
这个功能感觉上像是对全局函数设置了一个Trust Region，可以防止更新的量过大二导致梯度发散。
我认为这一步的想法是很好的，但是实际操作中可能会有问题。
实际中可能只有部分参数的梯度比较大，而其他参数的梯度本身比较小，
那么对所有的参数乘以相同的因子会让一些本来比较小的参数变得更小，这样会带来一些不公平*/
  virtual void ClipGradients();
  //  以下与solver中对应函数功能一致
  virtual void SnapshotSolverState(const string& model_filename);
  virtual void SnapshotSolverStateToBinaryProto(const string& model_filename);
  virtual void SnapshotSolverStateToHDF5(const string& model_filename);
  virtual void RestoreSolverStateFromHDF5(const string& state_file);
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file);
  // history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots
  /*  
  	  以下是SGD求解时需要的临时存储，
  	  history_中保留历史增量数据
  	  updata_中保留更新相关数据，不需要打快照
  	  temp_ 中保留在计算梯度，更新值时可能需要的其他信息，不要快照
  	  */
  vector<shared_ptr<Blob<Dtype> > > history_, update_, temp_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  explicit NesterovSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) {}
  explicit NesterovSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) {}
  virtual inline const char* type() const { return "Nesterov"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};

template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaGradSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { constructor_sanity_check(); }
  explicit AdaGradSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }
  virtual inline const char* type() const { return "AdaGrad"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with AdaGrad.";
  }

  DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
};


template <typename Dtype>
class RMSPropSolver : public SGDSolver<Dtype> {
 public:
  explicit RMSPropSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { constructor_sanity_check(); }
  explicit RMSPropSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }
  virtual inline const char* type() const { return "RMSProp"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with RMSProp.";
    CHECK_GE(this->param_.rms_decay(), 0)
        << "rms_decay should lie between 0 and 1.";
    CHECK_LT(this->param_.rms_decay(), 1)
        << "rms_decay should lie between 0 and 1.";
  }

  DISABLE_COPY_AND_ASSIGN(RMSPropSolver);
};

template <typename Dtype>
class AdaDeltaSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaDeltaSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { AdaDeltaPreSolve(); }
  explicit AdaDeltaSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdaDeltaPreSolve(); }
  virtual inline const char* type() const { return "AdaDelta"; }

 protected:
  void AdaDeltaPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdaDeltaSolver);
};

/**
 * @brief AdamSolver, an algorithm for first-order gradient-based optimization
 *        of stochastic objective functions, based on adaptive estimates of
 *        lower-order moments. Described in [1].
 *
 * [1] D. P. Kingma and J. L. Ba, "ADAM: A Method for Stochastic Optimization."
 *     arXiv preprint arXiv:1412.6980v8 (2014).
 */
template <typename Dtype>
class AdamSolver : public SGDSolver<Dtype> {
 public:
  explicit AdamSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { AdamPreSolve();}
  explicit AdamSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdamPreSolve(); }
  virtual inline const char* type() const { return "Adam"; }

 protected:
  void AdamPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdamSolver);
};

}  // namespace caffe

#endif  // CAFFE_SGD_SOLVERS_HPP_
