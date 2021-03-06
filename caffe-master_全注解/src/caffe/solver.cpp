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
    //  如果外部请求函数设置好，直接调用
    return action_request_function_();
  }
  return SolverAction::NONE;
}

//会调用Init()方法进行初始化，即Solver scaffolding
//  下面是他的两个构造函数，分别从SolverParameter对象和solver描述文件创建，初始化
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

//  调用Init（），将Solverparameter作为参数传递
/*
功能：初始化网络
步骤：
1. 设置随机数种子
2. 申请一块Net空间以下面的构造函数进行初始化
param_file=train_net_，net_指向这块空间
3. 如果有test_net，则申请一块Net空间，test_net_指向这块空间
输入：SolverParameter类型的param
输出：无
*/
template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters:用parameter初始化solver "
    << std::endl << param.DebugString();
    //为solver类的数据成员param_赋值
  param_ = param;   //  将外部SolverParameter对象拷贝到内部
    // 默认为1
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
    //检测快照的的写入权限
  CheckSnapshotWritePermissions();
    //random_seed默认为-1，
  if (param_.random_seed() >= 0) {
  	   //调用Caffe命名空间里的set_random_seed函数，而不是caffe类的set_random_seed函数；
  //param_.random_seed()实际上调用的是::google::protobuf::int64 random_seed()
    Caffe::set_random_seed(param_.random_seed() + Caffe::solver_rank());  //  设置随机数种子
  }
  // Scaffolding code
    // 搭建网络结构
  InitTrainNet();    // 初始化训练网络
  InitTestNets();    // c初始化测试网络
  if (Caffe::root_solver()) {
    LOG(INFO) << "Solver scaffolding done.建好了";
  }
  //  迭代次数清零
    // iter_初始化为0
  iter_ = 0;
  //  学习速率步长清零
  current_step_ = 0;
}

// 初始化训练网络
template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {  // 初始化train网络，先检查一下
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
   //有且只能有一个train net
	  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  //  创建NetParameter 对象
  // 读取训练网络结构参数
  NetParameter net_param;  // 载入参数
  //  从Solve中获取NetParameter
  if (param_.has_train_net_param()) { //如果指定了网络描述文件prototxt，值了为true
    LOG_IF(INFO, Caffe::root_solver())  //  打印log
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());  //  这里的train_net_param()会返回prototxt文件路径
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
    //  读取并解析里面的内容，将参数载入net_param结构体
    //  该函数是由protobuffer工具完成文本到结构体变量转化的
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
  //  设置当前网络训练状态
   //设置正确的网络状态，训练从默认开始，然后融入通过网络层规定在任何状态，
  //最后融入训练状态（最优解）
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  //从低到高获取state,最终从最高优先级SolverParameter类型中的train_state,
  //显然这会覆盖掉之前获取的state。
  net_state.MergeFrom(param_.train_state());
  //这里获取的state可以为Netparameter中的state赋值，然后可以根据LayerParameter中的
  //include和exclude来确定该层是否应该包含在网络中。
  net_param.mutable_state()->CopyFrom(net_state);
    //调用模板类的构造函数，进行net的初始化
  net_.reset(new Net<Dtype>(net_param));//  创建训练网络
}

//  初始化测试（预测）网络
//需要注意的是TestNet可以有多个，而TrainNet只能有一个
template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
	//  还是先检查一下
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
    //可以有多个test net
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  //  确定当前solver里面有多少个预测网络
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
    //得到测试网络参数
  vector<NetParameter> net_params(num_test_net_instances);
  //  为每个预测网络设置参数，参数来源可能是solver.prototxt中的test_net_param条目
  //  或者solver.prototxt中指定的 test_net 文件
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  //  剩下的预测网络参数也做初始化
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
  //  初始化内置变量 test_nets_
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    //  设置每个预测网络的状态
    	// 设置正确的网络状态，训练从默认开始，然后融入通过网络层规定在任何状态，
	// 最后融入测试状态（最优解）
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    test_nets_[i].reset(new Net<Dtype>(net_params[i]));  //  创建每个测试网络
    test_nets_[i]->set_debug_info(param_.debug_info());  //  设置调试信息
  }
}

//  这个是求解器的迭代过程，核心函数
template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
	  // 设置开始的迭代次数(如果是从之前的snapshot恢复的，那iter_
  // 等于snapshot时的迭代次数)和结束的迭代次数
	//  入口参数 iters 表示需要循环多少次
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  // 输出的loss为前average_loss次loss的平均值，在solver.prototxt里设置，默认为1，
  // losses存储之前的average_loss个loss，smoothed_loss为最后要输出的均值
  int average_loss = this->param_.average_loss();
  losses_.clear();
  smoothed_loss_ = 0;
  iteration_timer_.Start();

//  循环开始  //迭代
  while (iter_ < stop_iter) {
    // zero-init the params
    //  清零训练网络的所有权值diff
    	// 清空上一次所有参数的梯度
    net_->ClearParamDiffs();
        // test_initialization默认为true
    // 判断是否需要测试
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())) {
      if (Caffe::root_solver()) {
      	  //  周期性预测，评估网络质量
        TestAll();
      }
            // 判断是否需要提前介绍迭代
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        //  停止
        break;
      }
    }

    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }
        // 判断当前迭代次数是否需要显示loss等信息
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    // accumulate the loss and gradient
    //  损失函数，误差值累加
    Dtype loss = 0;
    // iter_size也是在solver.prototxt里设置，实际上的batch_size=iter_size*网络定义里的batch_size，
    // 因此每一次迭代的loss是iter_size次迭代的和，再除以iter_size，这个loss是通过调用`Net::ForwardBackward`函数得到的
    // 这个设置我的理解是在GPU的显存不够的时候使用，比如我本来想把batch_size设置为128，但是会out_of_memory，
    // 借助这个方法，可以设置batch_size=32，iter_size=4，那实际上每次迭代还是处理了128个数据
    // accumulate gradients over `iter_size` x `batch_size` instances
    for (int i = 0; i < param_.iter_size(); ++i) {
    	    /*
     * 调用了Net中的代码，主要完成了前向后向的计算，
     * 前向用于计算模型的最终输出和Loss，后向用于
     * 计算每一层网络和参数的梯度。
     */
      loss += net_->ForwardBackward();
    }
    //  取平均
     //accumulate（累积） gradients over `iter_size` x `batch_size` instances。
    //默认情况下，iter_size=1,即默认情况下，一个iteratio一个batch
    loss /= param_.iter_size();
    // 计算要输出的smoothed_loss，如果losses里还没有存够average_loss个loss
    //则将当前的loss插入，如果已经存够了，则将之前的替换掉
    // average the loss across iterations for smoothed reporting
    /*
     * 这个函数主要做Loss的平滑。由于Caffe的训练方式是SGD，我们无法把所有的数据同时
     * 放入模型进行训练，那么部分数据产生的Loss就可能会和全样本的平均Loss不同，在必要
     * 时候将Loss和历史过程中更新的Loss求平均就可以减少Loss的震荡问题。
     */
    // average the loss across iterations for smoothed reporting
    //  对loss做平滑滤波
    UpdateSmoothedLoss(loss, start_iter, average_loss);
        //输出当前迭代信息
    if (display) {
      float lapse = iteration_timer_.Seconds();
      float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);
      	//  打印当前平滑滤波后的损失函数值
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << " (" << per_s << " iter/s, " << lapse << "s/"
          << param_.display() << " iters), loss = " << smoothed_loss_;
      iteration_timer_.Start();
      iterations_last_ = iter_;
      //  获取训练网络输出blob，格式化之后打印输出
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
    //  应用更新
     // 执行梯度的更新，这个函数在基类`Solver`中没有实现，会调用每个子类自己的实现
    //，后面具体分析`SGDSolver`的实现
    ApplyUpdate();
    //  在solver类中这是个纯虚函数，实现要到派生类SGDSolver中查看
    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    //  迭代次数递增
        // 迭代次数加1
    ++iter_;
     // 调用GetRequestedAction，实际是通过action_request_function_函数指针调用之前设置好(通过`SetRequestedAction`)的
    // signal_handler的`CheckForSignals`函数，这个函数的作用是
    // 会根据之前是否遇到系统信号以及信号的类型和我们设置(或者默认)的方式返回处理的方式

    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
    //  如果需要，打个快照
     // 判断当前迭代是否需要snapshot，如果request等于`SNAPSHOT`则也需要
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      Snapshot();
    }
        // 如果request为`STOP`则修改`requested_early_exit_`为true，之后就会提前结束迭代
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }
  }
}
/*
对整个网络进行训练（也就是你运行Caffe训练某个模型）的时候，实际上是在运行caffe.cpp中的
train()函数，而这个函数实际上是实例化一个Solver对象，初始化后调用了Solver中的Solve()方法
调用此方法训练网络，其中会调用Step()方法来迭代，迭代 param_.max_iter() - iter_ 次
*/
template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
	// 检查当前是否是root_solver(多GPU模式下，只有root_solver才运行这一部分的代码)

  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  // Initialize to false every time we start solving.
  //  每次都要初始化为 false
    // requested_early_exit_`一开始被赋值为false，也就是现在没有要求在优化结束前退出

  requested_early_exit_ = false;
    // 判断`resume_file`这个指针是否NULL，
  //如果不是则需要从resume_file存储的路径里读取之前训练的状态

  if (resume_file) {
  	  //  如果指定了快照文件，就从快照恢复训练
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  int start_iter = iter_;
  //对于一个正在训练的网络，没有bottom或top向量被给，而且仅仅提供dummy vecs

  // 然后调用了'Step'函数，这个函数执行了实际的逐步的迭代过程
  // 最大迭代次数
  Step(param_.max_iter() - iter_);  //  关键函数，在上面
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
    // 迭代结束或者遇到系统信号提前结束后，判断是否需要在训练结束之后snapshot
  // 这个可以在solver.prototxt里设置
  //  求解后打一次快照，可以在solver.prototxt中说明snapshot_after_train := false，来禁止这个功能
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
  //  在全部求解完成后再进行一次额外的训练，预测，显示损失函数
    // 优化完后，运行一个额外的训练和测试过程展示训练测试的loss或者输出。
  // 判断是否需要输出最后的loss
  if (param_.display() && iter_ % param_.display() == 0) {
    int average_loss = this->param_.average_loss();
    Dtype loss;
    net_->Forward(&loss);

    UpdateSmoothedLoss(loss, start_iter, average_loss);

    LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
  }
    // 判断是否需要最后Test
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
	//  遍历test_nets_中的每个对象
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    Test(test_net_id);
  }
}

//  对单个net进行评估
template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
    //检查是否有layer共享于多个网络
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  //  获得单个test_net对象
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  // 迭代次数在solver.prototxt中由test_iter 设定
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
        //如果在训练或测试中断请求发出后，随时执行保存快照
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
    //  预测网络执行前向传播计算
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
    	//  记录loss值
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
  //   打印损失函数值
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  //  打印准确率，损失函数
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
        //求多次迭代Loss的平均值，也就是求多个batch的平局值，
    //一次迭代用的是一个test batch-size 的图片
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
  }
}

//  solver 的快照功能
//输出当前网络状态到一个文件中。
template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  	  //  权值快照支持两种格式，HDF5和protobuffer，在求解文件中设定
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }
//  求解器打快照
  SnapshotSolverState(model_filename);
}

//  快照写入权限检查
//check快照的写入权限
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

//  获取快照全名
//Snapshot的名字
template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string extension) {
	//  前缀由solver文件指定
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

//  打快照，protobuffer格式
//Snapshot保存为二进制proto的模型
template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");  // 得到模型文件名
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());  //  将net_转化为NetParamer，就是导出到protobuffer对象
  WriteProtoToBinaryFile(net_param, model_filename);  //  再从ProtoBuffer对象写入磁盘文件，就是 *.caffemodel
  return model_filename;
}

//  打快照，HDF5格式
//Snapshot保存为HDF5模型
template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  //  直接调用net自带的toHDF5函数
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

//  从快照的回复功能
//从一个文件中读入网络状态，并可以从那个状态恢复。
template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  string state_filename(state_file);
  //  根据有无.h5后缀来判断快照格式，以选择恢复方法
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

//  loss平滑函数
//更新平滑后的Loss
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

///模板显示实例化
INSTANTIATE_CLASS(Solver);

}  // namespace caffe
