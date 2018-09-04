#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"
#include "caffe/util/benchmark.hpp"
/*
 * (1)solver_factory��register��create��ͬ����Solver�Ļ��ƣ�
 * (2)ͨ��signal_handler����ȡϵͳ�źţ��������û���Ĭ�ϵ����ý�����Ӧ�Ĵ���
 * (3)Solver::Solve�����ľ���ʵ�ֵķ�����
 * (4)SGDSolver::ApplyUpdate�����ľ���ʵ�֡�ǰ���������ֶ����ڻ���ģ�
 * ���һ����SGDSolver�������ģ�����û���Ҫʵ���Լ���Solver�࣬
 * ҲӦ�����Ƶ�ȥ�̳л��࣬��ʵ���Լ���ApplyUpdate�������ڴ����ĩβͨ��
 * register�����ע�ᣬ����Ա��ɹ��ĵ��á�
 */

namespace caffe {

/**
  * @brief Enumeration of actions that a client of the Solver may request by
  * implementing the Solver's action request function, which a
  * client may optionally provide in order to request early termination
  * or saving a snapshot without exiting. In the executable caffe, this
  * mechanism is used to allow the snapshot to be saved when stopping
  * execution with a SIGINT (Ctrl-C).
     ctrl+cֹͣʱ�Ķ��������գ�ֹͣ��û��
       * �����˼���ǰ�Ctrl-Cʱ���ᱣ�浱ǰѵ��ʱ��ģ�ͣ�
  * �������ѵ���ն˲�С�ı��ر�ʱ�����Խ����ϴμ���ѵ��
  */
  namespace SolverAction {
    enum Enum {
      NONE = 0,  // Take no special action.
      STOP = 1,  // Stop training. snapshot_after_train controls whether a snapshot is created.
                 //  ֹͣѵ�� snapshot_after_train �����Ƿ񴴽�һ������
      SNAPSHOT = 2  // Take a snapshot, and keep training. ��һ�����ռ���ѵ��
    };
  }

/**
 * @brief Type of a function that returns a Solver Action enumeration.
            ���صĺ���������ö���͵�SolverAction���ܿ��յ��Ǹ���
            ѧ��java�Ŀ������Ϊ�ع����������������˻�Ǯ��һ���û�ת����һ���˻�ʱ��
 ��;���������⣬һ���û�Ǯ�Ѿ����ˣ���һ��ȴû�����ӣ���ʱ��Ҫ�ع�������
 ������ʱѵ����ʱ���ж��ˣ�Ȼ��ع������ϴζϵ㣬����ѵ����
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *      ����ִ��net���Ż��Ľӿ�
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
 //  �������
template <typename Dtype>
class Solver {
 public:
 //  ������ʾ���캯�����ֱ��SolverParameter�����solver�����ļ�����
  explicit Solver(const SolverParameter& param);
  explicit Solver(const string& param_file);
  //  ��ʼ��
  void Init(const SolverParameter& param);
  void InitTrainNet();  // ��ʼ��ѵ��net
  void InitTestNets();  //  ��ʼ������net

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
 //  �������������ĵ���������ú�����ʹ�������۲�Ӧ�ò�ȡʲô���������ջ�����ǰ�˳�ѵ����
  void SetActionFunction(ActionCallback func);
  SolverAction::Enum GetRequestedAction();
  // The main entry of the solver function. In default, iter will be zero. 
  // Pass in a non-zero iter number to resume training for a pre-trained net.
   //  ��������������Ҫ��ڣ���һ��resume_file�лָ�ѵ��
   //  ���iter = 0 ʱ��ʼѵ�� 
    // ��������Ĭ��iterΪ0,��0��iter���뵽Ԥѵ����������
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  //  ���е�iter�ε���
  void Step(int iters);
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  //  ��resume_file�лָ�ѵ��  
  //�洢����ʵ����δ洢solver������ģ���С�Ӧ��ʵ��RestoreSolverState()����
  //��������Ǵ洢����SolverState�����״̬
  void Restore(const char* resume_file);
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
    // Solver::Snapshot��Ҫ�ǻ����Ŀ��չ��ܣ��洢ѧϰ������
  void Snapshot();  // ���պ���
  virtual ~Solver() {}  //  ����������
  //����solver����
  inline const SolverParameter& param() const { return param_; } 
    //��������
  inline shared_ptr<Net<Dtype> > net() { return net_; }
    //���ز�������
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() const { return iter_; }

  // Invoked at specific points during an iteration
  //  �ڵ����е����ض��ĵ� 
   // Ƕ���࣬�����Ķ������ڲ���Ķ������໥������
  // Ƕ��������������ж�����һ�����ͳ�Ա�������͵ķ���Ȩ������������
  class Callback {
   protected:
    virtual void on_start() = 0; //������������ÿһ��GPU��
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  const vector<Callback*>& callbacks() const { return callbacks_; }
  void add_callback(Callback* value) {
    callbacks_.push_back(value);
  }

  void CheckSnapshotWritePermissions();
  /**
   * @brief Returns the solver type.
   * ����slover����
   */
  virtual inline const char* type() const { return ""; }

 protected:
  // Make and apply the update value for the current iteration.
  // �ڻ��ܵ��߳��Ͻ��в�������
  //  �Ե�ǰ����������Ӧ�ø���ֵ�����麯������Ҫ����������ȥ���� 
   // ���麯������Ҫ������ʵ�֣����ɺ�Ӧ�õ�ǰ�����ĸ��µ�ֵ
  virtual void ApplyUpdate() = 0;
  string SnapshotFilename(const string extension);
  string SnapshotToBinaryProto();  // ������ļ����ͣ���ͬ
  string SnapshotToHDF5();
  // The test routine/ ���Գ���
  //  ��������в���
  void TestAll();
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  void DisplayOutputBlobs(const int net_id);
  void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);

  SolverParameter param_;  //  ���ڴ�prototxt�л�ȡ����  //Solver����
  int iter_;   //  ��ǰ��������
  int current_step_;  //  ��ǰstep��С������ѧϰ���ʲ���˥������
   // ѵ�����磬����ֻ��һ��
  shared_ptr<Net<Dtype> > net_;  //  ����net����ָ�룬����ѵ�� 
   // ������������ж�� 
   vector<shared_ptr<Net<Dtype> > > test_nets_; //  ����net����ָ�룬���ڲ���
  vector<Callback*> callbacks_;  //  �ص������б�
  vector<Dtype> losses_;
  Dtype smoothed_loss_;

  // A function that can be set by a client of the Solver 
  // to provide indication that it wants a snapshot saved and/or to exit early.
  //  ���������ձ������ǰ�˳�����ʾ 
   // ͨ��������ѡ��ȷ�ϰ�ť��ѡ�񱣴滹���˳����ա�
  ActionCallback action_request_function_;

  // True iff a request to stop early was received.
  //  ��ǰ�˳�����
  bool requested_early_exit_;

  // Timing information, handy to tune e.g. nbr of GPUs
  //  ��ʱ��Ϣ�����ڵ���
  Timer iteration_timer_;
  float iterations_last_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
