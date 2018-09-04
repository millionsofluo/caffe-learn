#include <cstdio>

#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    //  ����ⲿ���������úã�ֱ�ӵ���
    return action_request_function_();
  }
  return SolverAction::NONE;
}

//�����Init()�������г�ʼ������Solver scaffolding
//  �����������������캯�����ֱ��SolverParameter�����solver�����ļ���������ʼ��
template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param)
    : net_(), callbacks_(), requested_early_exit_(false) {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file)
    : net_(), callbacks_(), requested_early_exit_(false) {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(param_file, &param);
  Init(param);
}

//  ����Init��������Solverparameter��Ϊ��������
/*
���ܣ���ʼ������
���裺
1. �������������
2. ����һ��Net�ռ�������Ĺ��캯�����г�ʼ��
param_file=train_net_��net_ָ�����ռ�
3. �����test_net��������һ��Net�ռ䣬test_net_ָ�����ռ�
���룺SolverParameter���͵�param
�������
*/
template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters:��parameter��ʼ��solver "
    << std::endl << param.DebugString();
    //Ϊsolver������ݳ�Աparam_��ֵ
  param_ = param;   //  ���ⲿSolverParameter���󿽱����ڲ�
    // Ĭ��Ϊ1
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
    //�����յĵ�д��Ȩ��
  CheckSnapshotWritePermissions();
    //random_seedĬ��Ϊ-1��
  if (param_.random_seed() >= 0) {
  	   //����Caffe�����ռ����set_random_seed������������caffe���set_random_seed������
  //param_.random_seed()ʵ���ϵ��õ���::google::protobuf::int64 random_seed()
    Caffe::set_random_seed(param_.random_seed() + Caffe::solver_rank());  //  �������������
  }
  // Scaffolding code
    // �����ṹ
  InitTrainNet();    // ��ʼ��ѵ������
  InitTestNets();    // c��ʼ����������
  if (Caffe::root_solver()) {
    LOG(INFO) << "Solver scaffolding done.������";
  }
  //  ������������
    // iter_��ʼ��Ϊ0
  iter_ = 0;
  //  ѧϰ���ʲ�������
  current_step_ = 0;
}

// ��ʼ��ѵ������
template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {  // ��ʼ��train���磬�ȼ��һ��
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
   //����ֻ����һ��train net
	  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  //  ����NetParameter ����
  // ��ȡѵ������ṹ����
  NetParameter net_param;  // �������
  //  ��Solve�л�ȡNetParameter
  if (param_.has_train_net_param()) { //���ָ�������������ļ�prototxt��ֵ��Ϊtrue
    LOG_IF(INFO, Caffe::root_solver())  //  ��ӡlog
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());  //  �����train_net_param()�᷵��prototxt�ļ�·��
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
    //  ��ȡ��������������ݣ�����������net_param�ṹ��
    //  �ú�������protobuffer��������ı����ṹ�����ת����
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  //  ���õ�ǰ����ѵ��״̬
   //������ȷ������״̬��ѵ����Ĭ�Ͽ�ʼ��Ȼ������ͨ�������涨���κ�״̬��
  //�������ѵ��״̬�����Ž⣩
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  //�ӵ͵��߻�ȡstate,���մ�������ȼ�SolverParameter�����е�train_state,
  //��Ȼ��Ḳ�ǵ�֮ǰ��ȡ��state��
  net_state.MergeFrom(param_.train_state());
  //�����ȡ��state����ΪNetparameter�е�state��ֵ��Ȼ����Ը���LayerParameter�е�
  //include��exclude��ȷ���ò��Ƿ�Ӧ�ð����������С�
  net_param.mutable_state()->CopyFrom(net_state);
    //����ģ����Ĺ��캯��������net�ĳ�ʼ��
  net_.reset(new Net<Dtype>(net_param));//  ����ѵ������
}

//  ��ʼ�����ԣ�Ԥ�⣩����
//��Ҫע�����TestNet�����ж������TrainNetֻ����һ��
template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
	//  �����ȼ��һ��
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
    //�����ж��test net
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  //  ȷ����ǰsolver�����ж��ٸ�Ԥ������
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
    //�õ������������
  vector<NetParameter> net_params(num_test_net_instances);
  //  Ϊÿ��Ԥ���������ò�����������Դ������solver.prototxt�е�test_net_param��Ŀ
  //  ����solver.prototxt��ָ���� test_net �ļ�
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  //  ʣ�µ�Ԥ���������Ҳ����ʼ��
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  //  ��ʼ�����ñ��� test_nets_
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    //  ����ÿ��Ԥ�������״̬
    	// ������ȷ������״̬��ѵ����Ĭ�Ͽ�ʼ��Ȼ������ͨ�������涨���κ�״̬��
	// ����������״̬�����Ž⣩
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    test_nets_[i].reset(new Net<Dtype>(net_params[i]));  //  ����ÿ����������
    test_nets_[i]->set_debug_info(param_.debug_info());  //  ���õ�����Ϣ
  }
}

//  �����������ĵ������̣����ĺ���
template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
	  // ���ÿ�ʼ�ĵ�������(����Ǵ�֮ǰ��snapshot�ָ��ģ���iter_
  // ����snapshotʱ�ĵ�������)�ͽ����ĵ�������
	//  ��ڲ��� iters ��ʾ��Ҫѭ�����ٴ�
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  // �����lossΪǰaverage_loss��loss��ƽ��ֵ����solver.prototxt�����ã�Ĭ��Ϊ1��
  // losses�洢֮ǰ��average_loss��loss��smoothed_lossΪ���Ҫ����ľ�ֵ
  int average_loss = this->param_.average_loss();
  losses_.clear();
  smoothed_loss_ = 0;
  iteration_timer_.Start();

//  ѭ����ʼ  //����
  while (iter_ < stop_iter) {
    // zero-init the params
    //  ����ѵ�����������Ȩֵdiff
    	// �����һ�����в������ݶ�
    net_->ClearParamDiffs();
        // test_initializationĬ��Ϊtrue
    // �ж��Ƿ���Ҫ����
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())) {
      if (Caffe::root_solver()) {
      	  //  ������Ԥ�⣬������������
        TestAll();
      }
            // �ж��Ƿ���Ҫ��ǰ���ܵ���
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        //  ֹͣ
        break;
      }
    }

    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }
        // �жϵ�ǰ���������Ƿ���Ҫ��ʾloss����Ϣ
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    // accumulate the loss and gradient
    //  ��ʧ���������ֵ�ۼ�
    Dtype loss = 0;
    // iter_sizeҲ����solver.prototxt�����ã�ʵ���ϵ�batch_size=iter_size*���綨�����batch_size��
    // ���ÿһ�ε�����loss��iter_size�ε����ĺͣ��ٳ���iter_size�����loss��ͨ������`Net::ForwardBackward`�����õ���
    // ��������ҵ��������GPU���Դ治����ʱ��ʹ�ã������ұ������batch_size����Ϊ128�����ǻ�out_of_memory��
    // ���������������������batch_size=32��iter_size=4����ʵ����ÿ�ε������Ǵ�����128������
    // accumulate gradients over `iter_size` x `batch_size` instances
    for (int i = 0; i < param_.iter_size(); ++i) {
    	    /*
     * ������Net�еĴ��룬��Ҫ�����ǰ�����ļ��㣬
     * ǰ�����ڼ���ģ�͵����������Loss����������
     * ����ÿһ������Ͳ������ݶȡ�
     */
      loss += net_->ForwardBackward();
    }
    //  ȡƽ��
     //accumulate���ۻ��� gradients over `iter_size` x `batch_size` instances��
    //Ĭ������£�iter_size=1,��Ĭ������£�һ��iteratioһ��batch
    loss /= param_.iter_size();
    // ����Ҫ�����smoothed_loss�����losses�ﻹû�д湻average_loss��loss
    //�򽫵�ǰ��loss���룬����Ѿ��湻�ˣ���֮ǰ���滻��
    // average the loss across iterations for smoothed reporting
    /*
     * ���������Ҫ��Loss��ƽ��������Caffe��ѵ����ʽ��SGD�������޷������е�����ͬʱ
     * ����ģ�ͽ���ѵ������ô�������ݲ�����Loss�Ϳ��ܻ��ȫ������ƽ��Loss��ͬ���ڱ�Ҫ
     * ʱ��Loss����ʷ�����и��µ�Loss��ƽ���Ϳ��Լ���Loss�������⡣
     */
    // average the loss across iterations for smoothed reporting
    //  ��loss��ƽ���˲�
    UpdateSmoothedLoss(loss, start_iter, average_loss);
        //�����ǰ������Ϣ
    if (display) {
      float lapse = iteration_timer_.Seconds();
      float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);
      	//  ��ӡ��ǰƽ���˲������ʧ����ֵ
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << " (" << per_s << " iter/s, " << lapse << "s/"
          << param_.display() << " iters), loss = " << smoothed_loss_;
      iteration_timer_.Start();
      iterations_last_ = iter_;
      //  ��ȡѵ���������blob����ʽ��֮���ӡ���
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }
    //  Ӧ�ø���
     // ִ���ݶȵĸ��£���������ڻ���`Solver`��û��ʵ�֣������ÿ�������Լ���ʵ��
    //������������`SGDSolver`��ʵ��
    ApplyUpdate();
    //  ��solver�������Ǹ����麯����ʵ��Ҫ��������SGDSolver�в鿴
    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    //  ������������
        // ����������1
    ++iter_;
     // ����GetRequestedAction��ʵ����ͨ��action_request_function_����ָ�����֮ǰ���ú�(ͨ��`SetRequestedAction`)��
    // signal_handler��`CheckForSignals`���������������������
    // �����֮ǰ�Ƿ�����ϵͳ�ź��Լ��źŵ����ͺ���������(����Ĭ��)�ķ�ʽ���ش���ķ�ʽ

    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
    //  �����Ҫ���������
     // �жϵ�ǰ�����Ƿ���Ҫsnapshot�����request����`SNAPSHOT`��Ҳ��Ҫ
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      Snapshot();
    }
        // ���requestΪ`STOP`���޸�`requested_early_exit_`Ϊtrue��֮��ͻ���ǰ��������
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }
  }
}
/*
�������������ѵ����Ҳ����������Caffeѵ��ĳ��ģ�ͣ���ʱ��ʵ������������caffe.cpp�е�
train()���������������ʵ������ʵ����һ��Solver���󣬳�ʼ���������Solver�е�Solve()����
���ô˷���ѵ�����磬���л����Step()���������������� param_.max_iter() - iter_ ��
*/
template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
	// ��鵱ǰ�Ƿ���root_solver(��GPUģʽ�£�ֻ��root_solver��������һ���ֵĴ���)

  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  // Initialize to false every time we start solving.
  //  ÿ�ζ�Ҫ��ʼ��Ϊ false
    // requested_early_exit_`һ��ʼ����ֵΪfalse��Ҳ��������û��Ҫ�����Ż�����ǰ�˳�

  requested_early_exit_ = false;
    // �ж�`resume_file`���ָ���Ƿ�NULL��
  //�����������Ҫ��resume_file�洢��·�����ȡ֮ǰѵ����״̬

  if (resume_file) {
  	  //  ���ָ���˿����ļ����ʹӿ��ջָ�ѵ��
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  int start_iter = iter_;
  //����һ������ѵ�������磬û��bottom��top�������������ҽ����ṩdummy vecs

  // Ȼ�������'Step'�������������ִ����ʵ�ʵ��𲽵ĵ�������
  // ����������
  Step(param_.max_iter() - iter_);  //  �ؼ�������������
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
    // ����������������ϵͳ�ź���ǰ�������ж��Ƿ���Ҫ��ѵ������֮��snapshot
  // ���������solver.prototxt������
  //  �����һ�ο��գ�������solver.prototxt��˵��snapshot_after_train := false������ֹ�������
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  //  ��ȫ�������ɺ��ٽ���һ�ζ����ѵ����Ԥ�⣬��ʾ��ʧ����
    // �Ż��������һ�������ѵ���Ͳ��Թ���չʾѵ�����Ե�loss���������
  // �ж��Ƿ���Ҫ�������loss
  if (param_.display() && iter_ % param_.display() == 0) {
    int average_loss = this->param_.average_loss();
    Dtype loss;
    net_->Forward(&loss);

    UpdateSmoothedLoss(loss, start_iter, average_loss);

    LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
  }
    // �ж��Ƿ���Ҫ���Test
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
	//  ����test_nets_�е�ÿ������
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    Test(test_net_id);
  }
}

//  �Ե���net��������
template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
    //����Ƿ���layer�����ڶ������
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  //  ��õ���test_net����
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  // ����������solver.prototxt����test_iter �趨
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
        //�����ѵ��������ж����󷢳�����ʱִ�б������
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    //  Ԥ������ִ��ǰ�򴫲�����
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
    	//  ��¼lossֵ
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO)     << "Test interrupted.";
    return;
  }
  //   ��ӡ��ʧ����ֵ
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  //  ��ӡ׼ȷ�ʣ���ʧ����
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
        //���ε���Loss��ƽ��ֵ��Ҳ��������batch��ƽ��ֵ��
    //һ�ε����õ���һ��test batch-size ��ͼƬ
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
  }
}

//  solver �Ŀ��չ���
//�����ǰ����״̬��һ���ļ��С�
template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  	  //  Ȩֵ����֧�����ָ�ʽ��HDF5��protobuffer��������ļ����趨
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }
//  ����������
  SnapshotSolverState(model_filename);
}

//  ����д��Ȩ�޼��
//check���յ�д��Ȩ��
template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {
  if (Caffe::root_solver() && param_.snapshot()) {
    CHECK(param_.has_snapshot_prefix())
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
          << param_.snapshot_prefix() << "'.  Make sure "
          << "that the directory exists and is writeable.";
    }
  }
}

//  ��ȡ����ȫ��
//Snapshot������
template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string extension) {
	//  ǰ׺��solver�ļ�ָ��
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

//  ����գ�protobuffer��ʽ
//Snapshot����Ϊ������proto��ģ��
template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");  // �õ�ģ���ļ���
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());  //  ��net_ת��ΪNetParamer�����ǵ�����protobuffer����
  WriteProtoToBinaryFile(net_param, model_filename);  //  �ٴ�ProtoBuffer����д������ļ������� *.caffemodel
  return model_filename;
}

//  ����գ�HDF5��ʽ
//Snapshot����ΪHDF5ģ��
template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  //  ֱ�ӵ���net�Դ���toHDF5����
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

//  �ӿ��յĻظ�����
//��һ���ļ��ж�������״̬�������Դ��Ǹ�״̬�ָ���
template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  string state_filename(state_file);
  //  ��������.h5��׺���жϿ��ո�ʽ����ѡ��ָ�����
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

//  lossƽ������
//����ƽ�����Loss
template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
    int average_loss) {
  if (losses_.size() < average_loss) {
    losses_.push_back(loss);
    int size = losses_.size();
    smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
  } else {
    int idx = (iter_ - start_iter) % average_loss;
    smoothed_loss_ += (loss - losses_[idx]) / average_loss;
    losses_[idx] = loss;
  }
}

///ģ����ʾʵ����
INSTANTIATE_CLASS(Solver);

}  // namespace caffe
