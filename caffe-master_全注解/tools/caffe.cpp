/*
	caffe框架编译后会生成 动态链接库 libcaffe.so ，本身不能独立运行
	要运行则需要一个main()函数作为入口，调用caffe的API
	tool/目录下就是一些调用 libcaffe.so的使用工具源码
	*/

/*
 * 这个文件调用caffe进行网络训练的主要代码，
 * 内含train，test，time等函数对网络进行训练，测试，微调，时间计算等*/

#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>//  简化命令行参数解析库
#include <glog/logging.h>  //  包含这个文件以使用GLOG

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"  //  字符串处理库
#include "caffe/caffe.hpp"    //  包含这个头文件就可以使用caffe中的所有组件（Blob，Lyaer，Net，Solver等等），具体可以在这个头文件中查看
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

//  定义函数参数：参数名，默认值， -help是的说明文字
//  使用方法： -iterations = 60
DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
    "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0,
    "Optional; network level.");
DEFINE_string(stage, "",
    "Optional; network stages (not to be confused with phase), "
    "separated by ','.");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",  //  中断信号
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",  //  终止信号
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");

// A simple registry for caffe commands.
//  简单的caffe命令注册表
/*
	g_brew_map实现过程:
	这个是用typedef定义函数指针方法。这个程序定义一个BrewFunction函数指针类型，
	在caffe.cpp 中 BrewFunction 作为GetBrewFunction()函数的返回类型，
	可以是 train()，test()，device_query()，time() 这四个函数指针的其中一个。
	在train()，test()，中可以调用solver类的函数，从而进入到net，进入到每一层，运行整个caffe程序。
*/
 //  定义函数指针类型 , BrewFunction 就是为对应的brew函数的指针
typedef int (*BrewFunction)(); 
 //  c++标准map容器，容器类型名为BrewMap ，声明map容器变量
 //  因为输入参数可能为train，test，device_query，time，所以定义一个容器类
 // string       为argv[1], 表示train或test或device_query或time。
typedef std::map<caffe::string, BrewFunction> BrewMap; BrewMap g_brew_map;  



//在C/C++的宏中，”#”的功能是将其后面的宏参数进行字符串化操作(Stringfication)，
//简单说就是在对它所引用的宏变量通过替换后在其左右各加上一个双引号。
//”##”被称为连接符(concatenator)，用来将两个子串Token连接为一个Token。
//注意这里连接的对象是Token就行，而不一定是宏的变量。
//---------------------------------------------------------------------------------
//  宏定义，比如RegisterBrewFunction(train)时，
//  相当于在容器中g_brew_map注册了train函数的函数指针和其对应的名字“train
// RegisterBrewFunction()
// 在RegisterBrewFunction(train)     或
// 在RegisterBrewFunction(test)          或
// 在RegisterBrewFunction(device_query) 或
// 在RegisterBrewFunction(time)          或
// 处调用
// g_brew_map[#func] = &func;即把string和func放入g_brew_map中进行注册
// RegisterBrewFunction这个宏在每一个实现主要功能的函数之后将这个函数的名字和其对应的函数指针添加到了g_brew_map中，
// 然后在main函数中，通过GetBrewFunction得到了我们需要调用的那个函数的函数指针，并完成了调用。
//---------------------------------------------------------------------------------
//  g_brew_map 初始化
#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

/*	这个作用和#define RegisterBrewFunction(func) g_brew_map[#func]=&func; 
	 这个宏定义功能类似，其中，func可以为：train，test，device_query，time。
 
  	综上， caffe中定义了train()，test()，device_query()，time()四种方式。  
  	如果需要，咱们可以增加其他的方式，然后通过RegisterBrewFunction() 函数注册一下即可。 */
//---------------------------------------------------------------------------------
// 输入string     的name参数
// 根据g_brew_map   的map
// 返回BrewFunction 的函数指针
//---------------------------------------------------------------------------------
//在g_brew_map容器中查找对应函数指针并返回
//LOG来源于google的glog库，控制程序的日志输出消息和测试消息
static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) { //  判断输入的是不是g_brew_map中train，test，device_query，time中一个
    return g_brew_map[name]; // 如果是的话，就调用相应的train(),test()，device_query()，time()
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    //  该语句不可达，为了减少旧编译器警告
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
//  解析GPU id，或使用所有可用的gpu设备
//  //解析可用GPU，使用所有硬件  
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

//---------------------------------------------------------------------------------
// Parse phase from flags       
// 解析cmd参数里的Train或Test
//---------------------------------------------------------------------------------
caffe::Phase get_phase_from_flags(caffe::Phase default_value) {
  if (FLAGS_phase == "")
    return default_value;
  if (FLAGS_phase == "TRAIN")
    return caffe::TRAIN;
  if (FLAGS_phase == "TEST")
    return caffe::TEST;
  LOG(FATAL) << "phase must be \"TRAIN\" or \"TEST\"";
  return caffe::TRAIN;  // Avoid warning
}

// Parse stages from flags
//  得到阶段标志
vector<string> get_stages_from_flags() {
  vector<string> stages;
  boost::split(stages, FLAGS_stage, boost::is_any_of(","));   //  split 分割字符串
  return stages;
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);
/**
	caffe命令格式为：
	        caffe<comand><args>
	        
	 如果需要增加一个命令，就定义一个函数 int command() , 然后注册：
	 RegisterBrewFunction(action);
	*/
//---------------------------------------------------------------------------------
// Device Query: show diagnostic information （GPU诊断信息） for a GPU device.
//  设备查询命令，显示GPU设备诊断信息，列出每个GPU的详细参数
//---------------------------------------------------------------------------------
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
   // 通过cudaGetDeviceCount(&count)得到GPU个数，然后返回GPU id 0,1,...,count-1
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);   // 设置当前GPU为gpus[i]
    caffe::Caffe::DeviceQuery();  // 获得当前GPU信息
  }
  return 0;
}
RegisterBrewFunction(device_query);  //  这个就是注册上面的那个命令


//---------------------------------------------------------------------------------
// Load the weights from the specified caffemodel(s) into the train and test nets.
// 从指定的caffemodel 中想训练，预测网络载入训练过的权值
//  加载特定模型的权重到网络中
// 被CopyLayers(solver.get(), FLAGS_weights);调用
// weights即为：微调的模型，如bvlc_reference_caffenet.caffemodel
//---------------------------------------------------------------------------------
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  // 微调，把model_list 拆分成 model_names[0]、model_names[1]、model_names[2]
  boost::split(model_names, model_list, boost::is_any_of(",") );  
   // 共有model_names.size()个需要微调的模型，对于一个微调的model而言
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);// net()
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]); // test_nets()[j]
    }
  }
}

//---------------------------------------------------------------------------------
// Translate the signal effect the user specified on the command-line to the corresponding enumeration.
// 将输入的命令行信息转化为相应的枚举类型
// ctrl+c中断控制台 或 死机 时执行什么操作
//---------------------------------------------------------------------------------
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
  return caffe::SolverAction::NONE;
}

//---------------------------------------------------------------------------------
// Train / Finetune a model.
//  训练 精调一个模型
//---------------------------------------------------------------------------------
int train() {
 // google的glog库，检查--solver、--snapshot和--weight并输出消息；
 //  必须有指定solver，snapshot和weight两者指定其一；
   // 检查参数里面--weights和--snapshot有没有同时出现
  // 因为--snapshot表示的是继续训练模型，这种情况对应于用户之前暂停了模型训练，现在继续训练。
  //   而--weights 是在从头启动训练的时候需要的参数，表示对模型的finetune 
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";  //  有没有solver
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())  //  有没有snapshot或weight
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";
  vector<string> stages = get_stages_from_flags();//  是否有stage参数

 // 类SolverParameter来自caffe.pb.h,是由caffe.proto转化而来的
//（由google protobuffer编译过来的类，具体声明可以见代码文件build/src/caffe/proto/caffe.pb.h）；
  caffe::SolverParameter solver_param;  // 实例化SolverParameter类，该类保存solver参数和相应的方法


  // 该函数声明在include/caffe/util/upgrade_proto.hpp中，实现在src/caffe/util/upgrade_proto.cpp中；
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param); // 将 --solver指定solver.prototxt文件内容解析到solver_param中，
  
  solver_param.mutable_train_state()->set_level(FLAGS_level);  //   solver_param的训练阶段有几个就设置上
  for (int i = 0; i < stages.size(); i++) {
    solver_param.mutable_train_state()->add_stage(stages[i]);  //  有几个有添加到stage中
  }

  // If the gpus flag is not provided, allow the mode and device to be set in the solver prototxt.
    // 根据命令参数-gpu或者solver.prototxt提供的信息，设置GPU
  if (FLAGS_gpu.size() == 0     //  这个if说如果命令行里没有gpu，且solver文件里面有模型，且模型里规定了gpu
      && solver_param.has_solver_mode()
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      if (solver_param.has_device_id()) {   //  如果solver文件里有设备id
          FLAGS_gpu = "" +
              boost::lexical_cast<string>(solver_param.device_id());  //  boost::lexical_cast，一个数据转换函数，把id号转化为string类型
      } else {  // Set default GPU if unspecified  设置默认的设备号
          FLAGS_gpu = "" + boost::lexical_cast<string>(0);   // boost::lexical_cast(0)，将数值0转换为字符串'“0”；
      }
  }

  //  得到gpu
  // 多GPU下，将GPU编号存入vector容器中（get_gpus()函数通过FLAGS_gpu获取）；
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);   // 如果没有GPU，使用CPU
  } else {
    ostringstream s; //  先定义一个s
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];   //  gpu信息写到s里
    }
    LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i) {
      cudaGetDeviceProperties(&device_prop, gpus[i]);  // 设置当前GPU为gpus[i]，给cuda他们用
      LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    }
#endif
    solver_param.set_device_id(gpus[0]);  //  在 solver_param 中设置前面得到的那个gpu设备是哪个
    Caffe::SetDevice(gpus[0]);   //  同cuda启动这个gpu，在common中定义的
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpus.size());  /// 看看一共有几个gpu
  }

// 处理snapshot, stop or none信号，其声明在include/caffe/util/signal_Handler.h中；GetRequestedAction在caffe.cpp中，
  //  这个是将    ‘stop’，‘snapshot’，‘none’     转换为标准信号；
  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  // 声明boost库中智能指针solver，指向caffe::Solver对象，该对象由CreateSolver创建
  /*下面就开始构造网络训练器solver，调用SolverRegistry的CreateSolver函数得到一个solver，
  在初始化solver的过程中，使用了之前解析好的用户定义的solver.prototxt文件，
  solver负担了整个网络的训练责任，详细结构后面再解析*/
  shared_ptr<caffe::Solver<float> >
  	   // 建立solver，利用之前的solver.prototxt解析成的solver_param
  //  这里涉及到工厂模式和map的特性，因为求解器有很多种，比如sgd之类的
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  // solver设置操作函数,
  //  根据上面那个得到的标准信号，采取相应的动作
  solver->SetActionFunction(signal_handler.GetActionFunction());

  /*这里查询了一下用户有没有定义snapshot参数和weights参数，
  因为如果定义了这两个参数，代表用户可能会希望从之前的中断训练处继续训练或者借用其他模型初始化网络，
  caffe在对两个参数相关的内容进行处理时都要用到solver指针*/
  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());      // 要么从snapshot中断处 继续训练
  } else if (FLAGS_weights.size()) {
    CopyLayers(solver.get(), FLAGS_weights);    // 要么从weights的已有现成模型 来微调
  }


  LOG(INFO) << "Starting Optimization“开始优化”";
 /*如果有不止一块gpu参与训练，那么将开启多gpu训练模式*/
  if (gpus.size() > 1) {
  // 这里是对于多GPU下的处理，NCCL是多GPU加速的
#ifdef USE_NCCL
    caffe::NCCL<float> nccl(solver);
    nccl.Run(gpus, FLAGS_snapshot.size() > 0 ? FLAGS_snapshot.c_str() : NULL);
#else
    LOG(FATAL) << "Multi-GPU execution not available - rebuild with USE_NCCL";
#endif
  } else {
  	  /*最核心的：使用Solve()接口正式开始优化网络*/
    solver->Solve(); // 使用solver求解，应该是包括 正向+反向传播 的过程
        // 初始化完成，开始优化网络（核心，重要）
  }
  LOG(INFO) << "Optimization Done优化完成.";
  return 0;
}
RegisterBrewFunction(train);  //  注册train命令


//---------------------------------------------------------------------------------
// Test: score a model.
//  测试：用模型打分
//---------------------------------------------------------------------------------
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";  // 检查网络结构：train_val.prototxt
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";   // 检查模型参数：如bvlc_reference_caffenet.caffemodel
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  //  设置设备id和模式
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);    // 读GPU信息
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  //  实例化caffe net 对象
   // 实例化caffe网络
  Net<float> caffe_net(FLAGS_model, caffe::TEST, FLAGS_level, &stages);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);    // 从caffemodel中读取blob数据
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
   // 对每次迭代计算loss和score
  for (int i = 0; i < FLAGS_iterations; ++i) {   // 每次迭代 
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&iter_loss);   // result:前向的结果，将结果写入iter_loss
    loss += iter_loss;// 计算loss
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();    // result_vec:result的值
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];   // score:result_vec的值
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;     // 将score的值相加
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  // 输出test结果
  loss /= FLAGS_iterations;  // 平均loss
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];  // output_name
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];  // loss_weight
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;  // 平均score
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);  //  注册test命令


// Time: benchmark the execution time of a model.
//  计时：评测模型执行时间
//测试网络模型的执行时间
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";
  caffe::Phase phase = get_phase_from_flags(caffe::TRAIN);
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  //  设置设备id 和模式
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  //  实例化一个caffe net
  Net<float> caffe_net(FLAGS_model, phase, FLAGS_level, &stages);

  // Do a clean forward and backward pass, so that memory allocation are done and future iterations will be more stable.
  //  做一次干净的前向，反向流程，保证完成存储区分配
  // 
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does not take any input blobs.
  // 速度测试，网络不需要任何输入blob
  float initial_loss;
  caffe_net.Forward(&initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);  //  注册time命令

int main(int argc, char** argv) {
    // Print output to stderr (while still logging). 
    //  stderr为默认输出到终端窗口，文件描述器代码为2
  FLAGS_alsologtostderr = 1;  
  // Set version  //  gflags库中设置版本版本信息
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.  //  gflags库中为main函数设置usage信息：
  gflags::SetUsageMessage("command line brew\n"  //四个命令
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"  // 训练、微调
      "  test            score a model\n"             // 测试（评分）
      "  device_query    show GPU diagnostic information\n"      // GPU信息
      "  time            benchmark model execution time");     // 模型执行时间
      
  // Run tool or show usage. //  运行工具或显示使用信息
  /*
  	  下面进行的是对gflags和glog的一些初始化
  	 还有一个需要注意的地方是，caffe架构中大量使用了gflags和glog，
	前者用于进行命令行参数的解析，而后者则是一个有效的日志记录工具，
	
  	  include/caffe/commom.hpp中声明的函数：
        GlobalInit函数定义在了caffe目录./src/caffe/common.cpp中，在下面是该函数的代码
        void GlobalInit(int* pargc, char*** pargv) {
        ::gflags::ParseCommandLineFlags(pargc, pargv, true);            
        ::google::InitGoogleLogging(*(pargv)[0]);                       
        // Provide a backtrace on segfault.

        #if !defined(_MSC_VER)                                          
          ::google::InstallFailureSignalHandler();                      
        #endif
    }
    在该函数中，ParseCommandLineFlags函数对gflags的参数进行了初始化，
    InitGoogleLogging函数初始化谷歌日志系统， 
    而InstallFailureSignalHandler注册信号处理句柄*/ 
  // 有一个作用，解析命令行，比如传入的是--solver，把他变成FLAGS_solver，就可以直接调用了
  caffe::GlobalInit(&argc, &argv);  
  //  main()函数中，输入的train，test，device_query，time。 通过下面两行进入程序。
  // if (argc == 2) {判断参数，参数为2，继续执行，否则输出usage信息。
  	  //  两个参数就是  train 和  --solver  ？？？？大概吧
 //  return GetBrewFunction(caffe::string(argv[1]))();
  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER     // 因为在commonsetting.props中<PythonSupport>为false，故WITH_PYTHON_LAYER未定义
    try {
#endif
//GetBrewFunction（）函数，返回BrewFunction函数指针。
 /*上面完成了一些初始化工作，而真正的程序入口就是下面这个GetBrewFunction函数，定义在上边
    这个函数的主要功能为：去查找g_brew_map容器，并在其中找到与caffe::string(argv[1]) 相匹配的函数指针 
    就是判断输入的是那个命令：train，test，device_query，time，然后调用相应的函数
    详见上面的GetBrewFunction()的定义*/
      return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");  // 参数输入不对
  }
}
/*
	总体而言，代码的精干部分抽出来是以下结构： mian函数->GetBrewFunction函数->train函数 
	从main函数出发，main函数里面在进行简短的对gflags与glog进行初始化以后，就开始进入了GetBrewFunction环节，
	solver.prototxt中配置的各种文件通过solve()接口进行网络的优化。
	还有一个需要注意的地方是，caffe架构中大量使用了gflags和glog，
	前者用于进行命令行参数的解析，而后者则是一个有效的日志记录工具，
	*/
