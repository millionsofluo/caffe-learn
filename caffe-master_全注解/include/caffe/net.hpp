f#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG) specified by a NetParameter.
 *        有一个参数指定，将layer连接到一个指定的无循环图DAG
 *
 * TODO(dox): more thorough description.
 /*
 * 分析net.hpp及net.cpp之前，先分析insert_splits.hpp定义出的InsertSplits函数，
 * 若某层的top(即输出)被两个或两个以上的层作为输入或输入的一部分，则对该层增加空间
 * 位置与其成并列关系的一个或若干个SplitLayer
 */
 */
template <typename Dtype>
class Net {
 public:
   //显示构造函数,构造函数声明成explicit就可以防止隐式转换
  explicit Net(const NetParameter& param);
  explicit Net(const string& param_file, Phase phase,
      const int level = 0, const vector<string>* stages = NULL);
  virtual ~Net() {}  //析构函数  //虚析构函数是为了解决这样的一个问题：
  // 基类的指针指向派生类对象，并用基类的指针删除派生类对象。

  /// @brief Initialize a network with a NetParameter.
  // 用一个 NetParameter 来初始化网络
  void Init(const NetParameter& param);

  /**
   * @brief Run Forward and return the result.
   *      开始前向传播，并返回结果loss
   *     
   */
  const vector<Blob<Dtype>*>& Forward(Dtype* loss = NULL);
  /// @brief DEPRECATED; use Forward() instead.
  //  运行前向传播，输入的blob已经填充好了
  const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL) {
    LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: ForwardPrefilled() "
        << "will be removed in a future version. Use Forward().";
    return Forward(loss);
  }

  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that
    (1) computing from one layer to another might entail(继承)
   * extra computation on unrelated branches, and 
   (2) computation starting in the middle may be incorrect if all of the
   *  layers of a fan-in are not included.
   * 
   */
   /*执行从start层到end层的前向传递，采用简单的for循环调用。*/
   	   //net的几种传播形式
  Dtype ForwardFromTo(int start, int end);
  Dtype ForwardFrom(int start);
  Dtype ForwardTo(int end);
  /// @brief DEPRECATED; set input blobs then use Forward() instead.
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom,
      Dtype* loss = NULL); // 对输入的blob执行前向传播，返回一个blob

  /**
   * @brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
      *        对网络的diffs清零。
   *        在backward之前调用
   */
   // 清零所有的diff权值域，在反向传播之前运行
     // 清空上一次所有参数的梯度
  void ClearParamDiffs();

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   计算网络的backward时应该没有输入和输出，只计算梯度，数据已经在forward时提供
   
      * 反向传播不需要输入和输出，因为数据在前向传播的时候已经提供*/
   /*和前面的ForwardFromTo函数类似，调用从start层到end层的反向传递。*/
   	   //
  void Backward();
  void BackwardFromTo(int start, int end);
  void BackwardFrom(int start);
  void BackwardTo(int end);

  /**
   * @brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   * 计算输出特征的尺寸
   */
  void Reshape();
  //ForwardBackward：按顺序调用了Forward和Backward。
    // 进行一次正向传播，一次反向传播
  Dtype ForwardBackward() {
    Dtype loss;
    Forward(&loss);
    Backward();
    return loss;
  }

  /// @brief Updates the network weights based on the diff values computed.
  // 对net中所有layer自底向上进行变形，无序运行一次前向传播就可以计算各层所需的blob
  // 根据已经准备好的diff（由solve），更新网络权值，更新所有可学习参数
  void Update();
  /**
   * @brief Shares weight data of owner blobs with shared blobs.
   *   共享blob权重数据
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
       y由 Net::Init调用，非手动
   */
  void ShareWeights();

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
       对于已经初始化的网络，隐式的从其他网络复制pre-trained layers
   */
    // 从一个训练好的net里训练权值
  void ShareTrainedLayersWith(const Net* other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  // 对于一个已经初始化的网络，CopyTrainedLayersFrom()方法从另一个网络参数实例复制已经
  // 训练好的层
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string trained_filename);
  void CopyTrainedLayersFromBinaryProto(const string trained_filename);
  void CopyTrainedLayersFromHDF5(const string trained_filename);
  /// @brief Writes the net to a proto  ToProto函数完成网络的序列化到文件，循环调用了每个层的ToProto函数
  void ToProto(NetParameter* param, bool write_diff = false) const;
  /// @brief Writes the net to an HDF5 file.  序列化一个net到HDF5
  void ToHDF5(const string& filename, bool write_diff = false) const;

  /// @brief returns the network name.  返回网络名称
  inline const string& name() const { return name_; }
  /// @brief returns the layer names  返回层名称
  inline const vector<string>& layer_names() const { return layer_names_; }
  /// @brief returns the blob names 返回blob名称
  inline const vector<string>& blob_names() const { return blob_names_; }
  /// @brief returns the blobs 返回blob_
  inline const vector<shared_ptr<Blob<Dtype> > >& blobs() const {
    return blobs_;
  }
  /// @brief returns the layers 返回layer_
  inline const vector<shared_ptr<Layer<Dtype> > >& layers() const {
    return layers_;
  }
  /// @brief returns the phase: TRAIN or TEST  返回训练阶段
  inline Phase phase() const { return phase_; }
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
       返回每个layers 的bottom blob，通常不会
   */
  inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {
    return bottom_vecs_;
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
        返回每个layers的top blob，通常不会这么做
   */
  inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {
    return top_vecs_;
  }
  /// @brief returns the ids of the top blobs of layer i   
  //  返回第i层的top blob
    /// 返回指定层的top blobs
  inline const vector<int> & top_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
    return top_id_vecs_[i];
  }
  /// @brief returns the ids of the bottom blobs of layer i 
  //  返回第i层的bottom blob
    /// 返回指定层的底层blobs
  inline const vector<int> & bottom_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
    return bottom_id_vecs_[i];
  }
  //  返回每个layer的bottom blob是否需要反传
  inline const vector<vector<bool> >& bottom_need_backward() const {
    return bottom_need_backward_;
  }
  //  返回每层的bottom的损失权重
  inline const vector<Dtype>& blob_loss_weights() const {
    return blob_loss_weights_;
  }
  //  返回每个blob是否需要loss计算
  inline const vector<bool>& layer_need_backward() const {
    return layer_need_backward_;
  }
  /// @brief returns the parameters  返回所有权值
  inline const vector<shared_ptr<Blob<Dtype> > >& params() const {
    return params_;
  }
  //  返回所有可训练的权值
  inline const vector<Blob<Dtype>*>& learnable_params() const {
    return learnable_params_;
  }
  /// @brief returns the learnable parameter learning rate multipliers
  //  返回可训练的所有权值的速率倍乘因子
  inline const vector<float>& params_lr() const { return params_lr_; }
  inline const vector<bool>& has_params_lr() const { return has_params_lr_; }
  /// @brief returns the learnable parameter decay multipliers
  //  返回可训练权值的衰减因子
  inline const vector<float>& params_weight_decay() const {
    return params_weight_decay_;
  }
  inline const vector<bool>& has_params_decay() const {
    return has_params_decay_;
  }
  //  返回layer 名称和向下标量的映射对
  const map<string, int>& param_names_index() const {
    return param_names_index_;
  }
  //  返回权值所有者
  inline const vector<int>& param_owners() const { return param_owners_; }
  inline const vector<string>& param_display_names() const {
    return param_display_names_;
  }
  /// @brief Input and output blob numbers
  //  返回输入输出的blobs个数
  inline int num_inputs() const { return net_input_blobs_.size(); }
  inline int num_outputs() const { return net_output_blobs_.size(); }
  //  返回输入blob
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
  //  返回谁出blob
  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }
  //  返回输入blob下标
  inline const vector<int>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  //  返回输出blopb下标
  inline const vector<int>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  //  查找当前网络是否包含某一名称的blob
    /// 判断是否存在某个blob
  bool has_blob(const string& blob_name) const;
  //  如果包含把他找出来
    /// 根据blob名称返回blob值
  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
  //  查找当前网络是否包含某一名称的layer
    /// 判断是否存在某层
  bool has_layer(const string& layer_name) const;
  //  如果包含就把他找出来
  const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;

  void set_debug_info(const bool value) { debug_info_ = value; }

  // Helpers for Init.  帮助lnit的，帮助初始化
  /**
   * @brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
        过滤掉用户指定的某个阶段，级别，状态下不应该包含的layer
         根据当前状态，去掉某些不需要的层，比如测试时的dropout
   */
  static void FilterNet(const NetParameter& param,
      NetParameter* param_filtered);
  /// @brief return whether NetState state meets NetStateRule rule
  //  判断网络状态是否满足网络规则
  /** NetState描述网络的State，在caffe.proto里的定义如下:
message NetState {
  optional Phase phase = 1 [default = TEST];
  optional int32 level = 2 [default = 0];
  repeated string stage = 3;
}
Phase是个枚举类型的变量，取值为{TRAIN, TEST}，这个好理解，表示的是网络的两个阶段（训练和测试）；Level是个整型变量，stage是个字符串变量，这两个参数不太好翻译，caffe没有提供使用的例子，只能在理解了使用方法后根据需要去使用了。

NetStateRule描述的是一种规则，在层的定义里设置，用来决定Layer是否被加进网络，在caffe.proto里的定义如下:
message NetStateRule {
  optional Phase phase = 1;
  optional int32 min_level = 2;
  optional int32 max_level = 3;
  repeated string stage = 4;
  repeated string not_stage = 5;
}
net.cpp文件里的StateMeetsRule函数用来判断NetState是否符合NetStateRule的规则，符合的条件如下：
NetState的phase与NetStateRule的phase一致
NetState的level在NetStateRule的[min_level, max_level]区间里
NetState的stage包含NetStateRule所列出的所有stage并且不包含任何一个not_stage

网络在初始化的时候会调用函数net.cpp里的FilterNet函数，根据网络的NetState以及层的NetStateRule搭建符合规则的网络。NetState可以在网络的定义文件（NetParameter）或者在Solver文件（SolverParameter）中定义，具体的可以参考caffe.proto，SolverParameter里的优先级高于NetParameter。NetStateRule则需要在层的定义（LayerParameter）中设置，LayerParameter提供include和exclude两种规则，include的优先级高于exclude，有include的时候只看include，符合inlude才被加入；没有include的时候看exclude，符合exclude的层会被踢出网络，未设置规则的层则默认加进网络。
使用NetStateRule的好处就是可以灵活的搭建网络，可以只写一个网络定义文件，用不同的NetState产生所需要的网络，比如常用的那个train和test的网络就可以写在一起。 加上level和stage，用法就更灵活，这里可以发挥想象力了，举个例子，如下定义的网络经过初始化以后'innerprod'层就被踢出去了
state: { level: 2 }
name: 'example'
layer {
  name: 'data'
  type: 'Data'
  top: 'data'
  top: 'label'
}
layer {
  name: 'innerprod'
  type: 'InnerProduct'
  bottom: 'data'
  top: 'innerprod'
  include: { min_level: 3 }
}
layer {
  name: 'loss'
  type: 'SoftmaxWithLoss'
  bottom: 'innerprod'
  bottom: 'label'
}
   */
  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
      const string& layer_name);

  // Invoked at specific points during an iteration
  //   在迭代中调用特定的点
  class Callback {
   protected:
    virtual void run(int layer) = 0;

    template <typename T>
    friend class Net;
  };
  
  const vector<Callback*>& before_forward() const { return before_forward_; }
  void add_before_forward(Callback* value) {
    before_forward_.push_back(value);
  }
  const vector<Callback*>& after_forward() const { return after_forward_; }
  void add_after_forward(Callback* value) {
    after_forward_.push_back(value);
  }
  const vector<Callback*>& before_backward() const { return before_backward_; }
  void add_before_backward(Callback* value) {
    before_backward_.push_back(value);
  }
  const vector<Callback*>& after_backward() const { return after_backward_; }
  void add_after_backward(Callback* value) {
    after_backward_.push_back(value);
  }

 protected:
  // Helpers for Init.
  /// @brief Append a new top blob to the net.
  //  为网络追加一个top blob
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);
  /// @brief Append a new bottom blob to the net.
  //  为网络追加一个bottom blob
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);
  /// @brief Append a new parameter blob to the net.
  //   为网络追加一个权值blob 
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  /// @brief Helper for displaying debug info in Forward.
  //  显示前传调试信息
  void ForwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Backward.
    //  显示后传调试信息
  void BackwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Update.
    //  显示权值更新调试信息
  void UpdateDebugInfo(const int param_id);

  /// @brief The network name  网络名称
  string name_;
  /// @brief The phase: TRAIN or TEST  当前阶段是什么，测试还是训练
  Phase phase_;
  /// @brief Individual layers in the net  网络中的各个层// Layer容器
  vector<shared_ptr<Layer<Dtype> > > layers_;  //  独立层  layers_变量存储的是每层layer结构体的指针
  vector<string> layer_names_;  //  层名称
  map<string, int> layer_names_index_;  //  曾名称与索引映射表 // 关联容器，layer名称所对应的索引
  vector<bool> layer_need_backward_;//  标记不咯层是否需要后传  //每层layer是否需要计算反向传导
  /// @brief the blobs storing intermediate results between the layer.
  //  在层之间存储结果的中间blob  
  ////blobs_存储的是中间结果，是针对整个网络中所有非参数blob而设计的一个变量。
  vector<shared_ptr<Blob<Dtype> > > blobs_;  // 层与层中间传递数据的通道
  vector<string> blob_names_;  //  blob名称  //整个网络中，所有非参数blob的name
  map<string, int> blob_names_index_;  //  blob名称与索引映射表,关联容器  /// blob 名称索引键值对
   // 整个网络中，所有非参数blob，是否需要backward。
  // 注意，这里所说的所有非参数blob其实指的是AppendTop函数中遍历的所有top blob,
  // 并不是每一层的top+bottom,因为这一层的top就是下一层的bottom,网络是一层一层堆起来的。
  vector<bool> blob_need_backward_;  //  标记某个blob是否需要bp
  /// bottom_vecs stores the vectors containing the input for each layer.
  //  bottom_vecs 存储包含每个层的输入blob
  /// They don't actually host the blobs (blobs_ does), so we simply store pointers.
  /// 他们实际上并不是blob的所有者，所以我们只存储指针  
  //存储整个网络所有网络层的bottom blob指针,实际上存储的是前一层的top，
  //因为网络是一层一层堆起来的
  vector<vector<Blob<Dtype>*> > bottom_vecs_;  //  
  //存储整个网络所有网络层的bottom blob的ID
  vector<vector<int> > bottom_id_vecs_;  
  //整个网络所有网络层的bottom blob是否需要backward
  vector<vector<bool> > bottom_need_backward_;
  /// top_vecs stores the vectors containing the output for each layer
  //  top_vecs 存放每个层的输入blob，他实际上并不是这些blob的所有者，所以只存放指针
   // 存储整个网络所有网络层的top blob指针.
    vector<vector<Blob<Dtype>*> > top_vecs_;
  vector<vector<int> > top_id_vecs_;
  // 存储整个网络所有网络层的top blob的ID.top_id_vecs_中存储的最基本元素是
  // blob_id：每一个新的blob都会赋予其一个blob_id,top_vecs_则与之对应，
  // 但是这个blob_id可能是会有重复的（因为in-place）
  /// Vector of weight in the loss (or objective) function of each net blob,
  //  每个blob对全局损失函数的权重贡献
  /// indexed by blob_id.
  
  // 每次遍历一个layer的时候，都会resize blob_loss_weights_,
  // 然后调用模板类layer的loss函数返回loss_weight
  vector<Dtype> blob_loss_weights_;
  
  // 存储每层的可学习参数id
  // 存储的基本元素是net_param_id，每遍历一个参数blob
  // net_param_id和param_id_vecs_都会更新
  vector<vector<int> > param_id_vecs_;
    // 表示参数所属的layer在layers_中的位置
  // param_owners_ 是一个存储parameter "onwer"的一个向量  ——> -1
  // 表示当前Layer就是该parameter的"owner"
  vector<int> param_owners_;
  vector<string> param_display_names_;
//其元素为当layer_id 与当前param_id 组成的pair.vector<pair<int, int> > param_layer_indices_  
  vector<pair<int, int> > param_layer_indices_;
 // 是整个网络的参数non-empty name与index的映射。注意，这个name是ParamSpec 类型中的name。
  map<string, int> param_names_index_;
  /// blob indices for the input and the output of the net
  //  网络输入输出的blob索引
  // 整个网络的输入输出blob以及ID
  vector<int> net_input_blob_indices_;
  vector<int> net_output_blob_indices_; 
   // 网络输入输出的所有blob
  vector<Blob<Dtype>*> net_input_blobs_;
  vector<Blob<Dtype>*> net_output_blobs_;
  /// The parameters in the network.
  //  网络权值
   // 网络中的所有参数
  // 整个网络的参数blob。 !!!不管这个参数有没有non-emty name，是否参与share!!!
  
  vector<shared_ptr<Blob<Dtype> > > params_;
 //  可训练的网络权值
  vector<Blob<Dtype>*> learnable_params_;
  /**
   * The mapping from params_ -> learnable_params_: 
       从params_ -> learnable_params_的映射
   * we have learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   */
  vector<int> learnable_param_ids_;
  /// the learning rate multipliers for learnable_params_
  //  学习率倍乘因子
  vector<float> params_lr_;
  vector<bool> has_params_lr_;
  /// the weight decay multipliers for learnable_params_
  //  学习率衰减因子
  vector<float> params_weight_decay_;
  vector<bool> has_params_decay_;
  /// The bytes of memory used by this net
  //  记录网络占用 的内存大小  
  // 存储网络所用的字节数
  size_t memory_used_;
  /// Whether to compute and display debug info for the net.
  //  是否显示调试信息
  bool debug_info_;
  // Callbacks  回调函数
  vector<Callback*> before_forward_;
  vector<Callback*> after_forward_;
  vector<Callback*> before_backward_;
  vector<Callback*> after_backward_;

DISABLE_COPY_AND_ASSIGN(Net);  //禁止拷贝构造函数，赋值运算函数
};


}  // namespace caffe

#endif  // CAFFE_NET_HPP_
