#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
 /* Forward 声明了 boost::thread 代替导入的 boost/thread.hpp 
 	 避免了 boost/NVCC 问题 (#1009, #1010) on OSX */
namespace boost { class mutex; }

namespace caffe {

/**
 * @brief An interface for the units of computation which can be composed into a Net.
 *     计算单元的接口，他将被组成一个网络   
 *
 * Layer%s must implement a Forward function, in which they take their input
 * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
    layer 必须实施前传函数，在他们输入的blob和计算输出 blob
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input Blob%s, given the error gradients with their output %s.
 * 他可能要实施一个Backward 函数，他会给输入的blob计算梯度的误差，给出输出的梯度误差
 */
 //layer是基本类。
//深度网络：一层一层的layer，相互之间通过blob传输数据连接。
//layer必须要实现一个forward function，前传函数功能可以自己定义。
//在forward中，会从Input也就是layer的bottom（前一层）中获取blob，计算输出的blob
//实现一个反向传播，根据他们的Input的blob以及outPut的error gradient梯度误差计算得到该层的梯度误差
template <typename Dtype>
class Layer {
 public:
  /**
   * You should not implement your own constructor. Any set up code should go to SetUp(), 
   * where the dimensions of the bottom blobs are provided to the layer.
   * 你不应该实现自己的构造函数，任何的set up 都应该到SetUp()中执行，bottom blob的大小被提交给这个layer
   */
    //Layer中三个重要参数：
   //1.layer_param_:是protobuf文件中存储的layer参数
   //2.blobs_:存储layer的参数，在程序中用的，layer学习到的参数
   //3.param_propagate_down_:这个bool表示是否计算各个blob参数的diff，即传播误差。
   
   	// 显示的构造函数不需要重写，任何初始工作在SetUp()中完成

	// 构造方法只复制层参数说明的值，如果层说明参数中提供了权值和偏置参数，也复制
	// 继承自Layer类的子类都会显示的调用Layer的构造函数
  explicit Layer(const LayerParameter& param)
    : layer_param_(param) {
      // Set phase and copy blobs (if there are any).
      // 设置阶段和copy blob（如果有任何的）
      phase_ = param.phase();  // 设置当前的阶段（训练/测试）//训练还是测试
      
      // 在layer类中被初始化，如果blobs_size() > 0
      // 在prototxt文件中一般没有提供blobs参数，所以这段代码一般不执行
      if (layer_param_.blobs_size() > 0) {
      	  /*按 layer_param_ 设置本身blob对象个数，
      	  	  并依次将每个blob对象尺寸调整位于layer_param_中
      	  	  blob相同的尺寸。*/
      	  	    //在初始化列表初始化LayerParameter，
      	  	    //之后blobs_这里存放的是一个指向blob类的shared_ptr指针的一个vector，
      	  	    //这里是申请空间，然后将出传入的layer_param中的blob拷贝过来
        blobs_.resize(layer_param_.blobs_size()); 
        for (int i = 0; i < layer_param_.blobs_size(); ++i) {
          blobs_[i].reset(new Blob<Dtype>());  //给一个新的没存地址
          blobs_[i]->FromProto(layer_param_.blobs(i));  //分配进去
        }
      }
    }
      // 虚析构
  virtual ~Layer() {}  

  /**
   * @brief Implements common layer setup functionality.
   *  实现公共层的设置功能
   * @param bottom the preshaped（预） input blobs
   * @param top
   *     the allocated but unshaped output blobs, to be shaped by Reshape
   *     输入未成形的blob，送去Reshape
   * Checks that the number of bottom and top blobs is correct.
      检查bottom 和 top blob的数量是否正确
   * Calls LayerSetUp to do special layer setup for individual layer types
      调用 LayerSetUp 给单独的层类型设置特殊的层
   * followed by Reshape to set up sizes of top blobs and internal buffers.
     然后进行形状调整，以便分配最大的内存缓冲区
   * Sets up the loss weight multiplier blobs for any non-zero loss weights.
      为所有非0的loss weight 设置减重乘法器
   * This method may not be overridden.  
      这个方法不能被覆盖
   */
   /**
   * @brief Implements common layer setup functionality.
   * @brief 实现每个对象的setup函数
   * @param bottom the preshaped input blobs
   * @param bottom 层的输入数据，blob中的存储空间已申请
   * @param top
   *     the allocated but unshaped output blobs, to be shaped by Reshape
   * @param top 层的输出数据，blob对象已构造但是其中的存储空间未申请，
   *     具体空间大小需根据bottom blob大小和layer_param_共同决定，具体在Reshape函数现实
   * Checks that the number of bottom and top blobs is correct.
   * Calls LayerSetUp to do special layer setup for individual layer types,
   * followed by Reshape to set up sizes of top blobs and internal buffers.
   * Sets up the loss weight multiplier blobs for any non-zero loss weights.
   * This method may not be overridden.
   * 1. 检查输入输出blob个数是否满足要求，每个层能处理的输入输出数据不一样
   * 2. 调用LayerSetUp函数初始化特殊的层，每个Layer子类需重写这个函数完成定制的初始化
   * 3. 调用Reshape函数为top blob分配合适大小的存储空间
   * 4. 为每个top blob设置损失权重乘子，非LossLayer为的top blob其值为零
   *
   * 此方法非虚函数，不用重写，模式固定
   */
  // 配置函数，实现常用层配置接口，不可被覆盖
    // layer 初始化设置
  void SetUp(const vector<Blob<Dtype>*>& bottom, //在模型初始化时重置 layers 及其相互之间的连接 ;
      const vector<Blob<Dtype>*>& top) {
    CheckBlobCounts(bottom, top);  // 检查blob
    LayerSetUp(bottom, top);  // 与层类型相关的配置过程
    Reshape(bottom, top);  // 对 top blob变形
    SetLossWeights(top);  // 设置loss weight 权值
  }

  /**
   * @brief Does layer-specific setup: your layer should implement this function as well as Reshape.
     @brief 定制初始化，每个子类layer必须实现此虚函数
   *
   * @param bottom
   *     the preshaped input blobs, whose data fields store the input data for this layer
   * @param bottom
   *     输入blob, 数据成员data_和diff_存储了相关数据
   * @param top
   *     the allocated but unshaped output blobs
   * @param top
   *     输出blob, blob对象已构造但数据成员的空间尚未申请
   *
   * This method should do one-time layer specific setup. 
      这个方法应该做一次分层设置
   * This includes reading and processing relevent parameters from the <code>layer_param_</code>.
      包括读取和处理相关参数，这些参数来自于<code>layer_param_</code>.
   * Setting up the shapes of top blobs and internal buffers should be done in <code>Reshape</code>,
      设置top blob和内存缓冲区的形状应该在其中进行
   *  which will be called before the forward pass to adjust the top blob sizes.
   * 在向前传递前将调用该值来调整 top blob大小。
    * 此方法执行一次定制化的层初始化，包括从layer_param_读入并处理相关的层权值和偏置参数，
   * 调用Reshape函数申请top blob的存储空间,由派生类重写
   */
   //LayerSetup就是对具体某一个layer的setup，被上面的那个函数所调用，
   //ShareInParallel和IsShared和SetShared分别是用来返回并行状态和获取这一Layer是否被多个nets所共享。
//默认：除了data layer都是关闭的。在多个GPU下的Train阶段以及share是true的情况下，is_shared将会被置成true
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  /**
   * @brief Adjust the shapes of top blobs and internal buffers to accommodate the shapes of the bottom blobs.
   *        调整top blob和内部缓冲区的形状，以适应bottom blob的形状
      * @brief 根据bottom blob的形状和layer_param_计算top blob的形状并为其分配存储空间
   *
   *
   * @param bottom the input blobs, with the requested input shapes
       bottom blob 要求输入的形状
   * @param top the top blobs, which should be reshaped as needed
   *    根据需要重塑 top blob 的形状
   * This method should reshape top blobs as needed according to the shapes of the bottom (input) blobs, 
       根据bottom 的形状来重塑 top 的形状
   * as well as reshaping any internal buffers and making any other necessary adjustments so that the layer can accommodate the bottom blobs.
   *     也可以调整任何内部缓冲，并进行其他必要的调整，使层能够适应bottom blob
   * 每个子类Layer必须重写的Reshape函数，完成top blob形状的设置并为其分配存储空间
   */
  //纯虚函数，形参为<Blob<Dtype>*>容器的const引用
   // 变形函数，修改top blob 和内部缓冲区的形状
     //这个reshape主要是layer用来根据输入的blob调节Internal buffer 以及输出的Blob
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
//////////////////////////////////////重要的函数/////////////////////////////////////////////////////////
  /**
   * @brief Given the bottom blobs, compute the top blobs and the loss.
   *     给定bottom blob  计算 top blob 和 loss
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
          输入的blob ，其data字段存储了该层的输入数据
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'outputs
   *      top 输出
   * \return The total loss from the layer.
   *    返回该层的总loss
   * The Forward wrapper calls the relevant device wrapper function(Forward_cpu or Forward_gpu) 
       前传 Forward 调用的函数就是封装好的(Forward_cpu or Forward_gpu) 
   * to compute the top blob values given the bottom blobs. 
       来计算给定bottom blob的top blob
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper then computes and returns the loss.
   *     如果该层有任何非零的loss_weights，那么就计算并返回loss
   *
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
       你自己写的层应该执行 Forward_cpu and (optionally) Forward_gpu
          * 这两个函数非虚函数，它们内部会调用如下虚函数(Forward_cpu and (optionally) Forward_gpu)完成数据前向传递和误差反向传播，
          * 根据执行环境的不同每个子类Layer必须重写CPU和GPU版本
   */
   //Forward：是一个装饰器，继承之后再调用的调用其相应的forward_cpu或者forward_gpu，
//根据输入的Input data blob计算相应的output data blob,同时，会反应这一层Layer的total loss
  inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Given the top blob error gradients, compute the bottom blob error gradients.
   *        给定 top blob 的误差梯度，计算 bottom blob的误差梯度
   *
   * @param top
   *     the output blobs, whose diff fields store the gradient of the error with respect to themselves
   *    output blob ， 他的diff字段将误差梯度存储在自己内部
   * @param propagate_down  多路开关
   *     a vector with equal length to bottom, 
          一个长度为bottom的向量，
   *    with each index indicating  whether to propagate the error gradients down to the bottom blob at the corresponding index
   *    每个索引指示是否将误差梯度传到对应的 bottom blob中
   * @param bottom
   *     the input blobs, whose diff fields will store the gradient of the error with respect to themselves after Backward is run
   *     输入的 blob ，他的diff字段在向后运行后将误差梯度存储在自己内部
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
   * top blob diffs.
   * 该函数调用相关的封装好的函数Backward_cpu or Backward_gpu，来实现真正的计算，由派生类负责
   * Your layer should implement Backward_cpu and (optionally) Backward_gpu.  （你自己加层就写上这两个函数）
   */
   //Backward：实现的是反向传播，也就是给定top blob的error gradient计算得到bottom的error gradient.
//输入时output blobs,在output blobs里面的diff存储的就是其相应的error gradients.
//其中propagate_down这个参数跟Bottom的长度是一样的，每一个index用来指定是否需要反向传播error gradients 到对应的bottom blob。
//bottom里面的diff区域存放的是Backward计算出来相应的gradient error
//给定相对于 top 层输出的梯度，计算其相对于输入的梯度，并传递到 bottom
 //层。一个有参数的 layer 需要计算相对于各个参数的梯度值并存储在内部。
  inline void Backward(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

  /**
   * @brief Returns the vector of learnable parameter blobs.
         返回学习参数的blob向量
            * 返回可学习的参数blobs
   */
   // 返回layer中可训练的权值，偏置项的blob向量
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }

  /**
   * @brief Returns the layer parameter.
       返回layer的参数
   */
  const LayerParameter& layer_param() const { return layer_param_; }

  /**
   * @brief Writes the layer parameter to a protocol buffer
       写入layer初始化参数到protobuffer缓冲区里
          * 把参数写进protocol buffer
   */
  virtual void ToProto(LayerParameter* param, bool write_diff = false);

  /**
   * @brief Returns the scalar loss associated with a top blob at a given index.
   //给定index返回相应的scalar loss
   */
     // 给定index返回相应的scalar loss
  inline Dtype loss(const int top_index) const {
    return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
  }

  /**
   * @brief Sets the loss associated with a top blob at a given index.
        设置与某个top blob相关的标量loss
   */
  inline void set_loss(const int top_index, const Dtype value) {
    if (loss_.size() <= top_index) {
      loss_.resize(top_index + 1, Dtype(0));
    }
    loss_[top_index] = value;
  }

  /**
   * @brief Returns the layer type.
      返回层类型，字符串形式，便于识别，由派生类负责
         * 返回层类型
   */
  virtual inline const char* type() const { return ""; }
//下面几个函数主要设置bottom或者top blob的数量状态，比较简单，通常需要layer类的派生类
  //重写，因为不同层指定的输入输出数量不同
  /**
   * @brief Returns the exact number of bottom blobs required by the layer,
   *        or -1 if no exact number is required.
   *     返回layer需要的输入blob数量，-1表示不关心，由派生类负责实现
   * This method should be overridden to return a non-negative value 
      这个方法应该被覆盖，以返回一个非负值
   * if your layer expects some exact number of bottom blobs.
      如果你想要一些确切的 bottom blob
   */
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of bottom blobs required by the layer,
   *        or -1 if no minimum number is required.
   *    返回一个layer必须的最小 bottom blob
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of bottom blobs.
      这个方法应该被覆盖，以返回一个非负的值，如果你想有一些最小的bottom blob
   */
  virtual inline int MinBottomBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of bottom blobs required by the layer,
   *        or -1 if no maximum number is required.
   *     同上  最大bottom blob
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of bottom blobs.
   */
  virtual inline int MaxBottomBlobs() const { return -1; }
  /**
   * @brief Returns the exact number of top blobs required by the layer,
   *        or -1 if no exact number is required.
   *     返回一个需要的top blob
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of top blobs.
   */
  virtual inline int ExactNumTopBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of top blobs required by the layer,
   *        or -1 if no minimum number is required.
   *   最小top blob
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of top blobs.
   */
  virtual inline int MinTopBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of top blobs required by the layer,
   *        or -1 if no maximum number is required.
   *   最大 top blob
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of top blobs.
   */
  virtual inline int MaxTopBlobs() const { return -1; }
  /**
   * @brief Returns true if the layer requires an equal number of bottom and top blobs.
   *        返回该层是否有相同的输入输出blob
   *
   * This method should be overridden to return true if your layer expects an
   * equal number of bottom and top blobs.
   */
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }

  /**
   * @brief Return whether "anonymous" top blobs are created automatically by the layer.
   *        返回是否允许匿名 top blob，由该层自动创建，
          若为真，在Net::Init（）函数中会创建足够多的top blob
          来满足该Layer 的 ExactNumTopBlobs() or MinTopBlobs().
   *
   * If this method returns true, Net::Init will create enough "anonymous" top
   * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
   * MinTopBlobs().
   */
  virtual inline bool AutoTopBlobs() const { return false; }

  /**
   * @brief Return whether to allow force_backward for a given bottom blob index.
   *        返回某些bottom blob是否允许强制反向传播，
           如果AllowForceBackward(i) == false 将忽略这个设定
   *
   * If AllowForceBackward(i) == false, we will ignore the force_backward
   * setting and backpropagate to blob i only if it needs gradient information
   * (as is done when force_backward == false).
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  /**
   * @brief Specifies whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   *     指定  该Layer是否计算相对权值或偏置项的梯度，具体相对谁，有param_id指定
   * You can safely ignore false values and always compute gradients for all parameters, 
   * but possibly with wasteful computation.
       你可以放心的忽略这个false，总是计算所有参数的梯度，但是可能会浪费计算资源
   */
  inline bool param_propagate_down(const int param_id) {
    return (param_propagate_down_.size() > param_id) ?
        param_propagate_down_[param_id] : false;
  }
  /**
   * @brief Sets whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
      *        设置是否对某个学习参数blob计算梯度
        设置  该Layer是否计算相对权值或偏置项的梯度，具体相对谁，有param_id指定
   */
  inline void set_param_propagate_down(const int param_id, const bool value) {
    if (param_propagate_down_.size() <= param_id) {
      param_propagate_down_.resize(param_id + 1, true);
    }
    param_propagate_down_[param_id] = value;
  }


 protected:
  /** The protobuf that stores the layer parameters 
       //protobuf文件中存储的layer参数,从protocal buffers格式的网络结构说明文件中读取
  //protected类成员，构造函数中初始化
     保存layer参数到protobuffer对象*/
  LayerParameter layer_param_;
  /** The phase: TRAIN or TEST 
  	    //层状态，参与网络的训练还是测试
  	  阶段：是train还是test*/
  Phase phase_;
  /** The vector that stores the learnable parameters as a set of blobs. 
  	    // 可学习参数层权值和偏置参数，使用向量是因为权值参数和偏置是分开保存在两个blob中的
  // 在基类layer中初始化(只是在描述文件定义了的情况下)
  	  layer内部的权值或偏置项，以blob方式组织*/
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  /** Vector indicating whether to compute the diff of each param blob.
  	    //// 标志每个可学习参数blob是否需要计算反向传递的梯度值
  	  标志位，是否计算对应参数的误差梯度 */
  vector<bool> param_propagate_down_;

  /** The vector that indicates whether each top blob has a non-zero weight in the objective function. 
   *  标志位，在目标函数中，是否每个top blob都有非零权重*/
     // 非LossLayer为零，LossLayer中表示每个top blob计算的loss的权重
  vector<Dtype> loss_;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////下面四个函数会经常看到，重要//////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /** @brief Using the CPU device, compute the layer output. 
  	  使用CPU模式计算输出
  	     * 纯虚函数，子类必须实现，使用cpu经行前向计算*/
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  /**
   * @brief Using the GPU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
         使用GPU计算层的输出，如果不行，返回使用CPU（）
   */
   /* void函数返回void函数
   * 为什么这么设置，是为了模板的统一性
   * template<class T>
   * T default_value()
   * {
    	return T();
   * }
   * 其中T可以为void
   */
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

  /**
   * @brief Using the CPU device, compute the gradients for any parameters and for the bottom blobs if propagate_down is true.
   *       使用CPU对任意梯度进行计算，如果传播是正确的，则对bottom blob进行计算
      * 纯虚函数，派生类必须实现
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) = 0;
  /**
   * @brief Using the GPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   *        Fall back to Backward_cpu() if unavailable.
        使用GPU计算参数梯度，同上
        如果不可用，则返回使用CPU
   */
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    // LOG(WARNING) << "Using CPU code as backup.";
    Backward_cpu(top, propagate_down, bottom);
  }

  /**
   * Called by the parent Layer's SetUp to check that the number of bottom
   * and top Blobs provided as input match the expected numbers specified by
   * the {ExactNum,Min,Max}{Bottom,Top}Blobs() functions.
       由父layer调用setup，检查bottom和top blob的数量作为输入，匹配指定的期望数字
       {ExactNum,Min,Max}{Bottom,Top}Blobs() functions.
   */
     // 检验输入/输出 Blob 数目是否满足 Layer 要求
       // 检查输出输出的blobs的个数是否在给定范围内
  virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
    if (ExactNumBottomBlobs() >= 0) {
      CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
          << type() << " Layer takes " << ExactNumBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MinBottomBlobs() >= 0) {
      CHECK_LE(MinBottomBlobs(), bottom.size())
          << type() << " Layer takes at least " << MinBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MaxBottomBlobs() >= 0) {
      CHECK_GE(MaxBottomBlobs(), bottom.size())
          << type() << " Layer takes at most " << MaxBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (ExactNumTopBlobs() >= 0) {
      CHECK_EQ(ExactNumTopBlobs(), top.size())
          << type() << " Layer produces " << ExactNumTopBlobs()
          << " top blob(s) as output.";
    }
    if (MinTopBlobs() >= 0) {
      CHECK_LE(MinTopBlobs(), top.size())
          << type() << " Layer produces at least " << MinTopBlobs()
          << " top blob(s) as output.";
    }
    if (MaxTopBlobs() >= 0) {
      CHECK_GE(MaxTopBlobs(), top.size())
          << type() << " Layer produces at most " << MaxTopBlobs()
          << " top blob(s) as output.";
    }
    if (EqualNumBottomTopBlobs()) {
      CHECK_EQ(bottom.size(), top.size())
          << type() << " Layer produces one top blob as output for each "
          << "bottom blob input.";
    }
  }

  /**
   * Called by SetUp to initialize the weights associated with any top blobs in the loss function.
   * Store non-zero loss weights in the diff blob.
       调用 SetUp来设置来初始化与任何top blob有关联的权重，在loss function中
       在diff blob中存储非零的loss weights
   */
     // 该函数在 Layer 的 SetUp 函数中被调用，主要目的是初始化与 Top Blob 相关的 loss 权重
  // 放到 Top Blob 的 diff 域，实际由 Forward() 计算 loss 函数
  // loss_weight == 0，表示当前层不参与 loss 函数计算，大部分 Layer 属于这一类
  // loss_weight == 1，表示当前层参与 loss 函数计算，损失层（LossLayer）属于这一类
  inline void SetLossWeights(const vector<Blob<Dtype>*>& top) {
  	   // 从protobuffer对象中获得Layer参数，这里需要用loss_weight参数
    const int num_loss_weights = layer_param_.loss_weight_size();
    if (num_loss_weights) {// 如果protobuffer中存在至少一个loss_weight参数
      // loss_weight参数个数应当与 Top Blob 数目相同，或者不要loss_weight参数
      CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
          "unspecified or specified once per top blob.";
       // 遍历每个 Top Blob
      for (int top_id = 0; top_id < top.size(); ++top_id) {
      	   // 从ProtoBuffer对象中获取loss_weight实际值（0 或者 1）
        const Dtype loss_weight = layer_param_.loss_weight(top_id);
        // 若为0，跳过
        if (loss_weight == Dtype(0)) { continue; }
        // 若不为0，则对网络做相关设置
        this->set_loss(top_id, loss_weight);// 本地记录loss_weight值
        const int count = top[top_id]->count();
        Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();
        // 将 loss_weight 值写入 Top Blob 的 diff 域，传递到其他需要使用的地方，实现远程同步
        caffe_set(count, loss_weight, loss_multiplier);
      }
    }
  }

 private:
  DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer

// Forward and backward wrappers. You should implement the cpu and gpu specific implementations instead,
// Forward and backward两个过程，你应该实现cpu和gpu的具体实现。
//  and should not change these functions.
// 不应该改变这些函数
// 前向传播函数、后向传播函数包装。不需要修改这两个函数
// 使用时只需要在派生类中改写 Forward_cpu、Forward_gpu、Backward_cpu、Backward_gpu 函数
// 前向传播和反向传播接口。 每个Layer的派生类都应该实现Forward_cpu()
template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype loss = 0;
  Reshape(bottom, top);
  switch (Caffe::mode()) { // 判断计算设备
  case Caffe::CPU: // 在CPU上执行前向运算
    Forward_cpu(bottom, top); // 调用CPU版的Forward函数
    // 计算 loss 值（如果有的话）
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      // 若为 LossLayer，则已经通过 Forward 函数计算出全局损失函数值，放在 Top Blob data 域
      const Dtype* data = top[top_id]->cpu_data(); 
      // 若 loss_weight 不为0，则已经在 SetLossWeights 函数中将 loss 权重放在 Top Blob diff 域
      const Dtype* loss_weights = top[top_id]->cpu_diff();
      loss += caffe_cpu_dot(count, data, loss_weights);
    }
    break;
  case Caffe::GPU:  //GPU模式，其他同上
    Forward_gpu(bottom, top);
#ifndef CPU_ONLY
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->gpu_data();
      const Dtype* loss_weights = top[top_id]->gpu_diff();
      Dtype blob_loss = 0;
      caffe_gpu_dot(count, data, loss_weights, &blob_loss);
      loss += blob_loss;
    }
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
  return loss;
}

// 反向传播函数，直接调用对应的设备函数
template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  switch (Caffe::mode()) {  //选一个
  case Caffe::CPU:
    Backward_cpu(top, propagate_down, bottom);
    break;
  case Caffe::GPU:
    Backward_gpu(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

// Serialize LayerParameter to protocol buffer
// 将LayerParameter（层的配置参数）序列化到协议缓冲区
//Layer的序列化函数,将layer的层说明参数layer_param_，
//层权值和偏置参数blobs_复制到LayerParameter对象，便于写到磁盘
template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
   // 复制层权值和偏置参数blobs_
  for (int i = 0; i < blobs_.size(); ++i) {// 权值和偏置项也会保存
    blobs_[i]->ToProto(param->add_blobs(), write_diff); // param->add_blobs() 会返回一个 BlobProto 指针
  }
}

}  // namespace caffe

#endif  // CAFFE_LAYER_H_
