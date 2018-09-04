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
 *        ��һ������ָ������layer���ӵ�һ��ָ������ѭ��ͼDAG
 *
 * TODO(dox): more thorough description.
 /*
 * ����net.hpp��net.cpp֮ǰ���ȷ���insert_splits.hpp�������InsertSplits������
 * ��ĳ���top(�����)���������������ϵĲ���Ϊ����������һ���֣���Ըò����ӿռ�
 * λ������ɲ��й�ϵ��һ�������ɸ�SplitLayer
 */
 */
template <typename Dtype>
class Net {
 public:
   //��ʾ���캯��,���캯��������explicit�Ϳ��Է�ֹ��ʽת��
  explicit Net(const NetParameter& param);
  explicit Net(const string& param_file, Phase phase,
      const int level = 0, const vector<string>* stages = NULL);
  virtual ~Net() {}  //��������  //������������Ϊ�˽��������һ�����⣺
  // �����ָ��ָ����������󣬲��û����ָ��ɾ�����������

  /// @brief Initialize a network with a NetParameter.
  // ��һ�� NetParameter ����ʼ������
  void Init(const NetParameter& param);

  /**
   * @brief Run Forward and return the result.
   *      ��ʼǰ�򴫲��������ؽ��loss
   *     
   */
  const vector<Blob<Dtype>*>& Forward(Dtype* loss = NULL);
  /// @brief DEPRECATED; use Forward() instead.
  //  ����ǰ�򴫲��������blob�Ѿ�������
  const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL) {
    LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: ForwardPrefilled() "
        << "will be removed in a future version. Use Forward().";
    return Forward(loss);
  }

  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that
    (1) computing from one layer to another might entail(�̳�)
   * extra computation on unrelated branches, and 
   (2) computation starting in the middle may be incorrect if all of the
   *  layers of a fan-in are not included.
   * 
   */
   /*ִ�д�start�㵽end���ǰ�򴫵ݣ����ü򵥵�forѭ�����á�*/
   	   //net�ļ��ִ�����ʽ
  Dtype ForwardFromTo(int start, int end);
  Dtype ForwardFrom(int start);
  Dtype ForwardTo(int end);
  /// @brief DEPRECATED; set input blobs then use Forward() instead.
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom,
      Dtype* loss = NULL); // �������blobִ��ǰ�򴫲�������һ��blob

  /**
   * @brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
      *        �������diffs���㡣
   *        ��backward֮ǰ����
   */
   // �������е�diffȨֵ���ڷ��򴫲�֮ǰ����
     // �����һ�����в������ݶ�
  void ClearParamDiffs();

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   ���������backwardʱӦ��û������������ֻ�����ݶȣ������Ѿ���forwardʱ�ṩ
   
      * ���򴫲�����Ҫ������������Ϊ������ǰ�򴫲���ʱ���Ѿ��ṩ*/
   /*��ǰ���ForwardFromTo�������ƣ����ô�start�㵽end��ķ��򴫵ݡ�*/
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
   * ������������ĳߴ�
   */
  void Reshape();
  //ForwardBackward����˳�������Forward��Backward��
    // ����һ�����򴫲���һ�η��򴫲�
  Dtype ForwardBackward() {
    Dtype loss;
    Forward(&loss);
    Backward();
    return loss;
  }

  /// @brief Updates the network weights based on the diff values computed.
  // ��net������layer�Ե����Ͻ��б��Σ���������һ��ǰ�򴫲��Ϳ��Լ�����������blob
  // �����Ѿ�׼���õ�diff����solve������������Ȩֵ���������п�ѧϰ����
  void Update();
  /**
   * @brief Shares weight data of owner blobs with shared blobs.
   *   ����blobȨ������
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
       y�� Net::Init���ã����ֶ�
   */
  void ShareWeights();

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
       �����Ѿ���ʼ�������磬��ʽ�Ĵ��������縴��pre-trained layers
   */
    // ��һ��ѵ���õ�net��ѵ��Ȩֵ
  void ShareTrainedLayersWith(const Net* other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  // ����һ���Ѿ���ʼ�������磬CopyTrainedLayersFrom()��������һ���������ʵ�������Ѿ�
  // ѵ���õĲ�
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string trained_filename);
  void CopyTrainedLayersFromBinaryProto(const string trained_filename);
  void CopyTrainedLayersFromHDF5(const string trained_filename);
  /// @brief Writes the net to a proto  ToProto���������������л����ļ���ѭ��������ÿ�����ToProto����
  void ToProto(NetParameter* param, bool write_diff = false) const;
  /// @brief Writes the net to an HDF5 file.  ���л�һ��net��HDF5
  void ToHDF5(const string& filename, bool write_diff = false) const;

  /// @brief returns the network name.  ������������
  inline const string& name() const { return name_; }
  /// @brief returns the layer names  ���ز�����
  inline const vector<string>& layer_names() const { return layer_names_; }
  /// @brief returns the blob names ����blob����
  inline const vector<string>& blob_names() const { return blob_names_; }
  /// @brief returns the blobs ����blob_
  inline const vector<shared_ptr<Blob<Dtype> > >& blobs() const {
    return blobs_;
  }
  /// @brief returns the layers ����layer_
  inline const vector<shared_ptr<Layer<Dtype> > >& layers() const {
    return layers_;
  }
  /// @brief returns the phase: TRAIN or TEST  ����ѵ���׶�
  inline Phase phase() const { return phase_; }
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
       ����ÿ��layers ��bottom blob��ͨ������
   */
  inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {
    return bottom_vecs_;
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
        ����ÿ��layers��top blob��ͨ��������ô��
   */
  inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {
    return top_vecs_;
  }
  /// @brief returns the ids of the top blobs of layer i   
  //  ���ص�i���top blob
    /// ����ָ�����top blobs
  inline const vector<int> & top_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
    return top_id_vecs_[i];
  }
  /// @brief returns the ids of the bottom blobs of layer i 
  //  ���ص�i���bottom blob
    /// ����ָ����ĵײ�blobs
  inline const vector<int> & bottom_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
    return bottom_id_vecs_[i];
  }
  //  ����ÿ��layer��bottom blob�Ƿ���Ҫ����
  inline const vector<vector<bool> >& bottom_need_backward() const {
    return bottom_need_backward_;
  }
  //  ����ÿ���bottom����ʧȨ��
  inline const vector<Dtype>& blob_loss_weights() const {
    return blob_loss_weights_;
  }
  //  ����ÿ��blob�Ƿ���Ҫloss����
  inline const vector<bool>& layer_need_backward() const {
    return layer_need_backward_;
  }
  /// @brief returns the parameters  ��������Ȩֵ
  inline const vector<shared_ptr<Blob<Dtype> > >& params() const {
    return params_;
  }
  //  �������п�ѵ����Ȩֵ
  inline const vector<Blob<Dtype>*>& learnable_params() const {
    return learnable_params_;
  }
  /// @brief returns the learnable parameter learning rate multipliers
  //  ���ؿ�ѵ��������Ȩֵ�����ʱ�������
  inline const vector<float>& params_lr() const { return params_lr_; }
  inline const vector<bool>& has_params_lr() const { return has_params_lr_; }
  /// @brief returns the learnable parameter decay multipliers
  //  ���ؿ�ѵ��Ȩֵ��˥������
  inline const vector<float>& params_weight_decay() const {
    return params_weight_decay_;
  }
  inline const vector<bool>& has_params_decay() const {
    return has_params_decay_;
  }
  //  ����layer ���ƺ����±�����ӳ���
  const map<string, int>& param_names_index() const {
    return param_names_index_;
  }
  //  ����Ȩֵ������
  inline const vector<int>& param_owners() const { return param_owners_; }
  inline const vector<string>& param_display_names() const {
    return param_display_names_;
  }
  /// @brief Input and output blob numbers
  //  �������������blobs����
  inline int num_inputs() const { return net_input_blobs_.size(); }
  inline int num_outputs() const { return net_output_blobs_.size(); }
  //  ��������blob
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
  //  ����˭��blob
  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }
  //  ��������blob�±�
  inline const vector<int>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  //  �������blopb�±�
  inline const vector<int>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  //  ���ҵ�ǰ�����Ƿ����ĳһ���Ƶ�blob
    /// �ж��Ƿ����ĳ��blob
  bool has_blob(const string& blob_name) const;
  //  ������������ҳ���
    /// ����blob���Ʒ���blobֵ
  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
  //  ���ҵ�ǰ�����Ƿ����ĳһ���Ƶ�layer
    /// �ж��Ƿ����ĳ��
  bool has_layer(const string& layer_name) const;
  //  ��������Ͱ����ҳ���
  const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;

  void set_debug_info(const bool value) { debug_info_ = value; }

  // Helpers for Init.  ����lnit�ģ�������ʼ��
  /**
   * @brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
        ���˵��û�ָ����ĳ���׶Σ�����״̬�²�Ӧ�ð�����layer
         ���ݵ�ǰ״̬��ȥ��ĳЩ����Ҫ�Ĳ㣬�������ʱ��dropout
   */
  static void FilterNet(const NetParameter& param,
      NetParameter* param_filtered);
  /// @brief return whether NetState state meets NetStateRule rule
  //  �ж�����״̬�Ƿ������������
  /** NetState���������State����caffe.proto��Ķ�������:
message NetState {
  optional Phase phase = 1 [default = TEST];
  optional int32 level = 2 [default = 0];
  repeated string stage = 3;
}
Phase�Ǹ�ö�����͵ı�����ȡֵΪ{TRAIN, TEST}���������⣬��ʾ��������������׶Σ�ѵ���Ͳ��ԣ���Level�Ǹ����ͱ�����stage�Ǹ��ַ���������������������̫�÷��룬caffeû���ṩʹ�õ����ӣ�ֻ���������ʹ�÷����������Ҫȥʹ���ˡ�

NetStateRule��������һ�ֹ����ڲ�Ķ��������ã���������Layer�Ƿ񱻼ӽ����磬��caffe.proto��Ķ�������:
message NetStateRule {
  optional Phase phase = 1;
  optional int32 min_level = 2;
  optional int32 max_level = 3;
  repeated string stage = 4;
  repeated string not_stage = 5;
}
net.cpp�ļ����StateMeetsRule���������ж�NetState�Ƿ����NetStateRule�Ĺ��򣬷��ϵ��������£�
NetState��phase��NetStateRule��phaseһ��
NetState��level��NetStateRule��[min_level, max_level]������
NetState��stage����NetStateRule���г�������stage���Ҳ������κ�һ��not_stage

�����ڳ�ʼ����ʱ�����ú���net.cpp���FilterNet���������������NetState�Լ����NetStateRule����Ϲ�������硣NetState����������Ķ����ļ���NetParameter��������Solver�ļ���SolverParameter���ж��壬����Ŀ��Բο�caffe.proto��SolverParameter������ȼ�����NetParameter��NetStateRule����Ҫ�ڲ�Ķ��壨LayerParameter�������ã�LayerParameter�ṩinclude��exclude���ֹ���include�����ȼ�����exclude����include��ʱ��ֻ��include������inlude�ű����룻û��include��ʱ��exclude������exclude�Ĳ�ᱻ�߳����磬δ���ù���Ĳ���Ĭ�ϼӽ����硣
ʹ��NetStateRule�ĺô����ǿ������Ĵ���磬����ֻдһ�����綨���ļ����ò�ͬ��NetState��������Ҫ�����磬���糣�õ��Ǹ�train��test������Ϳ���д��һ�� ����level��stage���÷��͸���������Է����������ˣ��ٸ����ӣ����¶�������羭����ʼ���Ժ�'innerprod'��ͱ��߳�ȥ��
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
  //   �ڵ����е����ض��ĵ�
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
  //  Ϊ����׷��һ��top blob
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);
  /// @brief Append a new bottom blob to the net.
  //  Ϊ����׷��һ��bottom blob
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);
  /// @brief Append a new parameter blob to the net.
  //   Ϊ����׷��һ��Ȩֵblob 
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  /// @brief Helper for displaying debug info in Forward.
  //  ��ʾǰ��������Ϣ
  void ForwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Backward.
    //  ��ʾ�󴫵�����Ϣ
  void BackwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Update.
    //  ��ʾȨֵ���µ�����Ϣ
  void UpdateDebugInfo(const int param_id);

  /// @brief The network name  ��������
  string name_;
  /// @brief The phase: TRAIN or TEST  ��ǰ�׶���ʲô�����Ի���ѵ��
  Phase phase_;
  /// @brief Individual layers in the net  �����еĸ�����// Layer����
  vector<shared_ptr<Layer<Dtype> > > layers_;  //  ������  layers_�����洢����ÿ��layer�ṹ���ָ��
  vector<string> layer_names_;  //  ������
  map<string, int> layer_names_index_;  //  ������������ӳ��� // ����������layer��������Ӧ������
  vector<bool> layer_need_backward_;//  ��ǲ������Ƿ���Ҫ��  //ÿ��layer�Ƿ���Ҫ���㷴�򴫵�
  /// @brief the blobs storing intermediate results between the layer.
  //  �ڲ�֮��洢������м�blob  
  ////blobs_�洢�����м�����������������������зǲ���blob����Ƶ�һ��������
  vector<shared_ptr<Blob<Dtype> > > blobs_;  // ������м䴫�����ݵ�ͨ��
  vector<string> blob_names_;  //  blob����  //���������У����зǲ���blob��name
  map<string, int> blob_names_index_;  //  blob����������ӳ���,��������  /// blob ����������ֵ��
   // ���������У����зǲ���blob���Ƿ���Ҫbackward��
  // ע�⣬������˵�����зǲ���blob��ʵָ����AppendTop�����б���������top blob,
  // ������ÿһ���top+bottom,��Ϊ��һ���top������һ���bottom,������һ��һ��������ġ�
  vector<bool> blob_need_backward_;  //  ���ĳ��blob�Ƿ���Ҫbp
  /// bottom_vecs stores the vectors containing the input for each layer.
  //  bottom_vecs �洢����ÿ���������blob
  /// They don't actually host the blobs (blobs_ does), so we simply store pointers.
  /// ����ʵ���ϲ�����blob�������ߣ���������ֻ�洢ָ��  
  //�洢������������������bottom blobָ��,ʵ���ϴ洢����ǰһ���top��
  //��Ϊ������һ��һ���������
  vector<vector<Blob<Dtype>*> > bottom_vecs_;  //  
  //�洢������������������bottom blob��ID
  vector<vector<int> > bottom_id_vecs_;  
  //������������������bottom blob�Ƿ���Ҫbackward
  vector<vector<bool> > bottom_need_backward_;
  /// top_vecs stores the vectors containing the output for each layer
  //  top_vecs ���ÿ���������blob����ʵ���ϲ�������Щblob�������ߣ�����ֻ���ָ��
   // �洢������������������top blobָ��.
    vector<vector<Blob<Dtype>*> > top_vecs_;
  vector<vector<int> > top_id_vecs_;
  // �洢������������������top blob��ID.top_id_vecs_�д洢�������Ԫ����
  // blob_id��ÿһ���µ�blob���ḳ����һ��blob_id,top_vecs_����֮��Ӧ��
  // �������blob_id�����ǻ����ظ��ģ���Ϊin-place��
  /// Vector of weight in the loss (or objective) function of each net blob,
  //  ÿ��blob��ȫ����ʧ������Ȩ�ع���
  /// indexed by blob_id.
  
  // ÿ�α���һ��layer��ʱ�򣬶���resize blob_loss_weights_,
  // Ȼ�����ģ����layer��loss��������loss_weight
  vector<Dtype> blob_loss_weights_;
  
  // �洢ÿ��Ŀ�ѧϰ����id
  // �洢�Ļ���Ԫ����net_param_id��ÿ����һ������blob
  // net_param_id��param_id_vecs_�������
  vector<vector<int> > param_id_vecs_;
    // ��ʾ����������layer��layers_�е�λ��
  // param_owners_ ��һ���洢parameter "onwer"��һ������  ����> -1
  // ��ʾ��ǰLayer���Ǹ�parameter��"owner"
  vector<int> param_owners_;
  vector<string> param_display_names_;
//��Ԫ��Ϊ��layer_id �뵱ǰparam_id ��ɵ�pair.vector<pair<int, int> > param_layer_indices_  
  vector<pair<int, int> > param_layer_indices_;
 // ����������Ĳ���non-empty name��index��ӳ�䡣ע�⣬���name��ParamSpec �����е�name��
  map<string, int> param_names_index_;
  /// blob indices for the input and the output of the net
  //  �������������blob����
  // ����������������blob�Լ�ID
  vector<int> net_input_blob_indices_;
  vector<int> net_output_blob_indices_; 
   // �����������������blob
  vector<Blob<Dtype>*> net_input_blobs_;
  vector<Blob<Dtype>*> net_output_blobs_;
  /// The parameters in the network.
  //  ����Ȩֵ
   // �����е����в���
  // ��������Ĳ���blob�� !!!�������������û��non-emty name���Ƿ����share!!!
  
  vector<shared_ptr<Blob<Dtype> > > params_;
 //  ��ѵ��������Ȩֵ
  vector<Blob<Dtype>*> learnable_params_;
  /**
   * The mapping from params_ -> learnable_params_: 
       ��params_ -> learnable_params_��ӳ��
   * we have learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   */
  vector<int> learnable_param_ids_;
  /// the learning rate multipliers for learnable_params_
  //  ѧϰ�ʱ�������
  vector<float> params_lr_;
  vector<bool> has_params_lr_;
  /// the weight decay multipliers for learnable_params_
  //  ѧϰ��˥������
  vector<float> params_weight_decay_;
  vector<bool> has_params_decay_;
  /// The bytes of memory used by this net
  //  ��¼����ռ�� ���ڴ��С  
  // �洢�������õ��ֽ���
  size_t memory_used_;
  /// Whether to compute and display debug info for the net.
  //  �Ƿ���ʾ������Ϣ
  bool debug_info_;
  // Callbacks  �ص�����
  vector<Callback*> before_forward_;
  vector<Callback*> after_forward_;
  vector<Callback*> before_backward_;
  vector<Callback*> after_backward_;

DISABLE_COPY_AND_ASSIGN(Net);  //��ֹ�������캯������ֵ���㺯��
};


}  // namespace caffe

#endif  // CAFFE_NET_HPP_
