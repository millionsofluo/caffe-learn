#ifndef CAFFE_SGD_SOLVERS_HPP_
#define CAFFE_SGD_SOLVERS_HPP_

#include <string>
#include <vector>

#include "caffe/solver.hpp"

namespace caffe {
//  ��SGD������������࣬������solver
/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
 //  ��ʾ���캯����ע���ڹ���ʱ���������presolver��������
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) { PreSolve(); }
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) { PreSolve(); }
  virtual inline const char* type() const { return "SGD"; }

//  ���Ȩֵ������ʷ
  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
 //  ���ǰ��׼������
  void PreSolve();
 /*�ṩ��һϵ�е�learning rate������
fixed��lr��Զ����
step��lr=baselr*gamma^{iter / stepsize}
exp��lr=baselr*gamma^{iter}
inv��lr=baselr*(1+gamma*iter)^{-power}
multistep��ֱ��дiter��ĳ����Χ��ʱlrӦ���Ƕ���
poly��lr=baselr*(1-\frac{iter}{maxiter})^{power}
sigmoid��lr=baselr*\frac{1}{1+e^{-gamma*(iter-stepsize)}}
��Щ�����������ӣ�ѡ���Լ�˳�ֵľͺá�*/
  Dtype GetLearningRate();  //  �õ�ѧϰ����
  virtual void ApplyUpdate();  //  Ӧ�ø���
  //������һЩ��һBatch���������ѵ�������⣬ͨ������ÿ��Batch�ĸ����������Ƹ�������
  //  ��ĳһȨֵ���й�һ��
  virtual void Normalize(int param_id); 
  //������������ݶ��ˡ�Caffe�ṩ�������򷽷�����L2��L1������L2�����˱�׼���ݶ��½�������L1������sub-gradient�ļ��㷽����
  //  ��ĳȨֵ���й�����
  virtual void Regularize(int param_id); 
  //  ����ĳȨֵ���ض�ѧϰ�����µĸ���ֵ
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  /*ClipGradients   �ݶ�����
��һ����Ҫ�Ƕ��ݶ�ֵ��һ�����ƣ�����ݶ�ֵ������ô����ͻ���ݶ���һ���޼���
�����еĲ�������һ���������ӣ�ʹ�����в�����ƽ���Ͳ������������趨���ݶ���ֵ��
������ܸо������Ƕ�ȫ�ֺ���������һ��Trust Region�����Է�ֹ���µ�������������ݶȷ�ɢ��
����Ϊ��һ�����뷨�Ǻܺõģ�����ʵ�ʲ����п��ܻ������⡣
ʵ���п���ֻ�в��ֲ������ݶȱȽϴ󣬶������������ݶȱ���Ƚ�С��
��ô�����еĲ���������ͬ�����ӻ���һЩ�����Ƚ�С�Ĳ�����ø�С�����������һЩ����ƽ*/
  virtual void ClipGradients();
  //  ������solver�ж�Ӧ��������һ��
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
  	  ������SGD���ʱ��Ҫ����ʱ�洢��
  	  history_�б�����ʷ��������
  	  updata_�б�������������ݣ�����Ҫ�����
  	  temp_ �б����ڼ����ݶȣ�����ֵʱ������Ҫ��������Ϣ����Ҫ����
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
