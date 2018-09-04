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
 /* Forward ������ boost::thread ���浼��� boost/thread.hpp 
 	 ������ boost/NVCC ���� (#1009, #1010) on OSX */
namespace boost { class mutex; }

namespace caffe {

/**
 * @brief An interface for the units of computation which can be composed into a Net.
 *     ���㵥Ԫ�Ľӿڣ����������һ������   
 *
 * Layer%s must implement a Forward function, in which they take their input
 * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
    layer ����ʵʩǰ�������������������blob�ͼ������ blob
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input Blob%s, given the error gradients with their output %s.
 * ������Ҫʵʩһ��Backward ����������������blob�����ݶȵ�������������ݶ����
 */
 //layer�ǻ����ࡣ
//������磺һ��һ���layer���໥֮��ͨ��blob�����������ӡ�
//layer����Ҫʵ��һ��forward function��ǰ���������ܿ����Լ����塣
//��forward�У����InputҲ����layer��bottom��ǰһ�㣩�л�ȡblob�����������blob
//ʵ��һ�����򴫲����������ǵ�Input��blob�Լ�outPut��error gradient�ݶ�������õ��ò���ݶ����
template <typename Dtype>
class Layer {
 public:
  /**
   * You should not implement your own constructor. Any set up code should go to SetUp(), 
   * where the dimensions of the bottom blobs are provided to the layer.
   * �㲻Ӧ��ʵ���Լ��Ĺ��캯�����κε�set up ��Ӧ�õ�SetUp()��ִ�У�bottom blob�Ĵ�С���ύ�����layer
   */
    //Layer��������Ҫ������
   //1.layer_param_:��protobuf�ļ��д洢��layer����
   //2.blobs_:�洢layer�Ĳ������ڳ������õģ�layerѧϰ���Ĳ���
   //3.param_propagate_down_:���bool��ʾ�Ƿ�������blob������diff����������
   
   	// ��ʾ�Ĺ��캯������Ҫ��д���κγ�ʼ������SetUp()�����

	// ���췽��ֻ���Ʋ����˵����ֵ�������˵���������ṩ��Ȩֵ��ƫ�ò�����Ҳ����
	// �̳���Layer������඼����ʾ�ĵ���Layer�Ĺ��캯��
  explicit Layer(const LayerParameter& param)
    : layer_param_(param) {
      // Set phase and copy blobs (if there are any).
      // ���ý׶κ�copy blob��������κεģ�
      phase_ = param.phase();  // ���õ�ǰ�Ľ׶Σ�ѵ��/���ԣ�//ѵ�����ǲ���
      
      // ��layer���б���ʼ�������blobs_size() > 0
      // ��prototxt�ļ���һ��û���ṩblobs������������δ���һ�㲻ִ��
      if (layer_param_.blobs_size() > 0) {
      	  /*�� layer_param_ ���ñ���blob���������
      	  	  �����ν�ÿ��blob����ߴ����λ��layer_param_��
      	  	  blob��ͬ�ĳߴ硣*/
      	  	    //�ڳ�ʼ���б��ʼ��LayerParameter��
      	  	    //֮��blobs_�����ŵ���һ��ָ��blob���shared_ptrָ���һ��vector��
      	  	    //����������ռ䣬Ȼ�󽫳������layer_param�е�blob��������
        blobs_.resize(layer_param_.blobs_size()); 
        for (int i = 0; i < layer_param_.blobs_size(); ++i) {
          blobs_[i].reset(new Blob<Dtype>());  //��һ���µ�û���ַ
          blobs_[i]->FromProto(layer_param_.blobs(i));  //�����ȥ
        }
      }
    }
      // ������
  virtual ~Layer() {}  

  /**
   * @brief Implements common layer setup functionality.
   *  ʵ�ֹ���������ù���
   * @param bottom the preshaped��Ԥ�� input blobs
   * @param top
   *     the allocated but unshaped output blobs, to be shaped by Reshape
   *     ����δ���ε�blob����ȥReshape
   * Checks that the number of bottom and top blobs is correct.
      ���bottom �� top blob�������Ƿ���ȷ
   * Calls LayerSetUp to do special layer setup for individual layer types
      ���� LayerSetUp �������Ĳ�������������Ĳ�
   * followed by Reshape to set up sizes of top blobs and internal buffers.
     Ȼ�������״�������Ա���������ڴ滺����
   * Sets up the loss weight multiplier blobs for any non-zero loss weights.
      Ϊ���з�0��loss weight ���ü��س˷���
   * This method may not be overridden.  
      ����������ܱ�����
   */
   /**
   * @brief Implements common layer setup functionality.
   * @brief ʵ��ÿ�������setup����
   * @param bottom the preshaped input blobs
   * @param bottom ����������ݣ�blob�еĴ洢�ռ�������
   * @param top
   *     the allocated but unshaped output blobs, to be shaped by Reshape
   * @param top ���������ݣ�blob�����ѹ��쵫�����еĴ洢�ռ�δ���룬
   *     ����ռ��С�����bottom blob��С��layer_param_��ͬ������������Reshape������ʵ
   * Checks that the number of bottom and top blobs is correct.
   * Calls LayerSetUp to do special layer setup for individual layer types,
   * followed by Reshape to set up sizes of top blobs and internal buffers.
   * Sets up the loss weight multiplier blobs for any non-zero loss weights.
   * This method may not be overridden.
   * 1. ����������blob�����Ƿ�����Ҫ��ÿ�����ܴ��������������ݲ�һ��
   * 2. ����LayerSetUp������ʼ������Ĳ㣬ÿ��Layer��������д���������ɶ��Ƶĳ�ʼ��
   * 3. ����Reshape����Ϊtop blob������ʴ�С�Ĵ洢�ռ�
   * 4. Ϊÿ��top blob������ʧȨ�س��ӣ���LossLayerΪ��top blob��ֵΪ��
   *
   * �˷������麯����������д��ģʽ�̶�
   */
  // ���ú�����ʵ�ֳ��ò����ýӿڣ����ɱ�����
    // layer ��ʼ������
  void SetUp(const vector<Blob<Dtype>*>& bottom, //��ģ�ͳ�ʼ��ʱ���� layers �����໥֮������� ;
      const vector<Blob<Dtype>*>& top) {
    CheckBlobCounts(bottom, top);  // ���blob
    LayerSetUp(bottom, top);  // ���������ص����ù���
    Reshape(bottom, top);  // �� top blob����
    SetLossWeights(top);  // ����loss weight Ȩֵ
  }

  /**
   * @brief Does layer-specific setup: your layer should implement this function as well as Reshape.
     @brief ���Ƴ�ʼ����ÿ������layer����ʵ�ִ��麯��
   *
   * @param bottom
   *     the preshaped input blobs, whose data fields store the input data for this layer
   * @param bottom
   *     ����blob, ���ݳ�Աdata_��diff_�洢���������
   * @param top
   *     the allocated but unshaped output blobs
   * @param top
   *     ���blob, blob�����ѹ��쵫���ݳ�Ա�Ŀռ���δ����
   *
   * This method should do one-time layer specific setup. 
      �������Ӧ����һ�ηֲ�����
   * This includes reading and processing relevent parameters from the <code>layer_param_</code>.
      ������ȡ�ʹ�����ز�������Щ����������<code>layer_param_</code>.
   * Setting up the shapes of top blobs and internal buffers should be done in <code>Reshape</code>,
      ����top blob���ڴ滺��������״Ӧ�������н���
   *  which will be called before the forward pass to adjust the top blob sizes.
   * ����ǰ����ǰ�����ø�ֵ������ top blob��С��
    * �˷���ִ��һ�ζ��ƻ��Ĳ��ʼ����������layer_param_���벢������صĲ�Ȩֵ��ƫ�ò�����
   * ����Reshape��������top blob�Ĵ洢�ռ�,����������д
   */
   //LayerSetup���ǶԾ���ĳһ��layer��setup����������Ǹ����������ã�
   //ShareInParallel��IsShared��SetShared�ֱ����������ز���״̬�ͻ�ȡ��һLayer�Ƿ񱻶��nets������
//Ĭ�ϣ�����data layer���ǹرյġ��ڶ��GPU�µ�Train�׶��Լ�share��true������£�is_shared���ᱻ�ó�true
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  /**
   * @brief Adjust the shapes of top blobs and internal buffers to accommodate the shapes of the bottom blobs.
   *        ����top blob���ڲ�����������״������Ӧbottom blob����״
      * @brief ����bottom blob����״��layer_param_����top blob����״��Ϊ�����洢�ռ�
   *
   *
   * @param bottom the input blobs, with the requested input shapes
       bottom blob Ҫ���������״
   * @param top the top blobs, which should be reshaped as needed
   *    ������Ҫ���� top blob ����״
   * This method should reshape top blobs as needed according to the shapes of the bottom (input) blobs, 
       ����bottom ����״������ top ����״
   * as well as reshaping any internal buffers and making any other necessary adjustments so that the layer can accommodate the bottom blobs.
   *     Ҳ���Ե����κ��ڲ����壬������������Ҫ�ĵ�����ʹ���ܹ���Ӧbottom blob
   * ÿ������Layer������д��Reshape���������top blob��״�����ò�Ϊ�����洢�ռ�
   */
  //���麯�����β�Ϊ<Blob<Dtype>*>������const����
   // ���κ������޸�top blob ���ڲ�����������״
     //���reshape��Ҫ��layer�������������blob����Internal buffer �Լ������Blob
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
//////////////////////////////////////��Ҫ�ĺ���/////////////////////////////////////////////////////////
  /**
   * @brief Given the bottom blobs, compute the top blobs and the loss.
   *     ����bottom blob  ���� top blob �� loss
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
          �����blob ����data�ֶδ洢�˸ò����������
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'outputs
   *      top ���
   * \return The total loss from the layer.
   *    ���ظò����loss
   * The Forward wrapper calls the relevant device wrapper function(Forward_cpu or Forward_gpu) 
       ǰ�� Forward ���õĺ������Ƿ�װ�õ�(Forward_cpu or Forward_gpu) 
   * to compute the top blob values given the bottom blobs. 
       ���������bottom blob��top blob
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper then computes and returns the loss.
   *     ����ò����κη����loss_weights����ô�ͼ��㲢����loss
   *
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
       ���Լ�д�Ĳ�Ӧ��ִ�� Forward_cpu and (optionally) Forward_gpu
          * �������������麯���������ڲ�����������麯��(Forward_cpu and (optionally) Forward_gpu)�������ǰ�򴫵ݺ����򴫲���
          * ����ִ�л����Ĳ�ͬÿ������Layer������дCPU��GPU�汾
   */
   //Forward����һ��װ�������̳�֮���ٵ��õĵ�������Ӧ��forward_cpu����forward_gpu��
//���������Input data blob������Ӧ��output data blob,ͬʱ���ᷴӦ��һ��Layer��total loss
  inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Given the top blob error gradients, compute the bottom blob error gradients.
   *        ���� top blob ������ݶȣ����� bottom blob������ݶ�
   *
   * @param top
   *     the output blobs, whose diff fields store the gradient of the error with respect to themselves
   *    output blob �� ����diff�ֶν�����ݶȴ洢���Լ��ڲ�
   * @param propagate_down  ��·����
   *     a vector with equal length to bottom, 
          һ������Ϊbottom��������
   *    with each index indicating  whether to propagate the error gradients down to the bottom blob at the corresponding index
   *    ÿ������ָʾ�Ƿ�����ݶȴ�����Ӧ�� bottom blob��
   * @param bottom
   *     the input blobs, whose diff fields will store the gradient of the error with respect to themselves after Backward is run
   *     ����� blob ������diff�ֶ���������к�����ݶȴ洢���Լ��ڲ�
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
   * top blob diffs.
   * �ú���������صķ�װ�õĺ���Backward_cpu or Backward_gpu����ʵ�������ļ��㣬�������ฺ��
   * Your layer should implement Backward_cpu and (optionally) Backward_gpu.  �����Լ��Ӳ��д��������������
   */
   //Backward��ʵ�ֵ��Ƿ��򴫲���Ҳ���Ǹ���top blob��error gradient����õ�bottom��error gradient.
//����ʱoutput blobs,��output blobs�����diff�洢�ľ�������Ӧ��error gradients.
//����propagate_down���������Bottom�ĳ�����һ���ģ�ÿһ��index����ָ���Ƿ���Ҫ���򴫲�error gradients ����Ӧ��bottom blob��
//bottom�����diff�����ŵ���Backward���������Ӧ��gradient error
//��������� top ��������ݶȣ������������������ݶȣ������ݵ� bottom
 //�㡣һ���в����� layer ��Ҫ��������ڸ����������ݶ�ֵ���洢���ڲ���
  inline void Backward(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

  /**
   * @brief Returns the vector of learnable parameter blobs.
         ����ѧϰ������blob����
            * ���ؿ�ѧϰ�Ĳ���blobs
   */
   // ����layer�п�ѵ����Ȩֵ��ƫ�����blob����
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }

  /**
   * @brief Returns the layer parameter.
       ����layer�Ĳ���
   */
  const LayerParameter& layer_param() const { return layer_param_; }

  /**
   * @brief Writes the layer parameter to a protocol buffer
       д��layer��ʼ��������protobuffer��������
          * �Ѳ���д��protocol buffer
   */
  virtual void ToProto(LayerParameter* param, bool write_diff = false);

  /**
   * @brief Returns the scalar loss associated with a top blob at a given index.
   //����index������Ӧ��scalar loss
   */
     // ����index������Ӧ��scalar loss
  inline Dtype loss(const int top_index) const {
    return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
  }

  /**
   * @brief Sets the loss associated with a top blob at a given index.
        ������ĳ��top blob��صı���loss
   */
  inline void set_loss(const int top_index, const Dtype value) {
    if (loss_.size() <= top_index) {
      loss_.resize(top_index + 1, Dtype(0));
    }
    loss_[top_index] = value;
  }

  /**
   * @brief Returns the layer type.
      ���ز����ͣ��ַ�����ʽ������ʶ���������ฺ��
         * ���ز�����
   */
  virtual inline const char* type() const { return ""; }
//���漸��������Ҫ����bottom����top blob������״̬���Ƚϼ򵥣�ͨ����Ҫlayer���������
  //��д����Ϊ��ͬ��ָ�����������������ͬ
  /**
   * @brief Returns the exact number of bottom blobs required by the layer,
   *        or -1 if no exact number is required.
   *     ����layer��Ҫ������blob������-1��ʾ�����ģ��������ฺ��ʵ��
   * This method should be overridden to return a non-negative value 
      �������Ӧ�ñ����ǣ��Է���һ���Ǹ�ֵ
   * if your layer expects some exact number of bottom blobs.
      �������ҪһЩȷ�е� bottom blob
   */
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of bottom blobs required by the layer,
   *        or -1 if no minimum number is required.
   *    ����һ��layer�������С bottom blob
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of bottom blobs.
      �������Ӧ�ñ����ǣ��Է���һ���Ǹ���ֵ�����������һЩ��С��bottom blob
   */
  virtual inline int MinBottomBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of bottom blobs required by the layer,
   *        or -1 if no maximum number is required.
   *     ͬ��  ���bottom blob
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of bottom blobs.
   */
  virtual inline int MaxBottomBlobs() const { return -1; }
  /**
   * @brief Returns the exact number of top blobs required by the layer,
   *        or -1 if no exact number is required.
   *     ����һ����Ҫ��top blob
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of top blobs.
   */
  virtual inline int ExactNumTopBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of top blobs required by the layer,
   *        or -1 if no minimum number is required.
   *   ��Сtop blob
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of top blobs.
   */
  virtual inline int MinTopBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of top blobs required by the layer,
   *        or -1 if no maximum number is required.
   *   ��� top blob
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of top blobs.
   */
  virtual inline int MaxTopBlobs() const { return -1; }
  /**
   * @brief Returns true if the layer requires an equal number of bottom and top blobs.
   *        ���ظò��Ƿ�����ͬ���������blob
   *
   * This method should be overridden to return true if your layer expects an
   * equal number of bottom and top blobs.
   */
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }

  /**
   * @brief Return whether "anonymous" top blobs are created automatically by the layer.
   *        �����Ƿ��������� top blob���ɸò��Զ�������
          ��Ϊ�棬��Net::Init���������лᴴ���㹻���top blob
          �������Layer �� ExactNumTopBlobs() or MinTopBlobs().
   *
   * If this method returns true, Net::Init will create enough "anonymous" top
   * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
   * MinTopBlobs().
   */
  virtual inline bool AutoTopBlobs() const { return false; }

  /**
   * @brief Return whether to allow force_backward for a given bottom blob index.
   *        ����ĳЩbottom blob�Ƿ�����ǿ�Ʒ��򴫲���
           ���AllowForceBackward(i) == false ����������趨
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
   *     ָ��  ��Layer�Ƿ�������Ȩֵ��ƫ������ݶȣ��������˭����param_idָ��
   * You can safely ignore false values and always compute gradients for all parameters, 
   * but possibly with wasteful computation.
       ����Է��ĵĺ������false�����Ǽ������в������ݶȣ����ǿ��ܻ��˷Ѽ�����Դ
   */
  inline bool param_propagate_down(const int param_id) {
    return (param_propagate_down_.size() > param_id) ?
        param_propagate_down_[param_id] : false;
  }
  /**
   * @brief Sets whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
      *        �����Ƿ��ĳ��ѧϰ����blob�����ݶ�
        ����  ��Layer�Ƿ�������Ȩֵ��ƫ������ݶȣ��������˭����param_idָ��
   */
  inline void set_param_propagate_down(const int param_id, const bool value) {
    if (param_propagate_down_.size() <= param_id) {
      param_propagate_down_.resize(param_id + 1, true);
    }
    param_propagate_down_[param_id] = value;
  }


 protected:
  /** The protobuf that stores the layer parameters 
       //protobuf�ļ��д洢��layer����,��protocal buffers��ʽ������ṹ˵���ļ��ж�ȡ
  //protected���Ա�����캯���г�ʼ��
     ����layer������protobuffer����*/
  LayerParameter layer_param_;
  /** The phase: TRAIN or TEST 
  	    //��״̬�����������ѵ�����ǲ���
  	  �׶Σ���train����test*/
  Phase phase_;
  /** The vector that stores the learnable parameters as a set of blobs. 
  	    // ��ѧϰ������Ȩֵ��ƫ�ò�����ʹ����������ΪȨֵ������ƫ���Ƿֿ�����������blob�е�
  // �ڻ���layer�г�ʼ��(ֻ���������ļ������˵������)
  	  layer�ڲ���Ȩֵ��ƫ�����blob��ʽ��֯*/
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  /** Vector indicating whether to compute the diff of each param blob.
  	    //// ��־ÿ����ѧϰ����blob�Ƿ���Ҫ���㷴�򴫵ݵ��ݶ�ֵ
  	  ��־λ���Ƿ�����Ӧ����������ݶ� */
  vector<bool> param_propagate_down_;

  /** The vector that indicates whether each top blob has a non-zero weight in the objective function. 
   *  ��־λ����Ŀ�꺯���У��Ƿ�ÿ��top blob���з���Ȩ��*/
     // ��LossLayerΪ�㣬LossLayer�б�ʾÿ��top blob�����loss��Ȩ��
  vector<Dtype> loss_;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////�����ĸ������ᾭ����������Ҫ//////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /** @brief Using the CPU device, compute the layer output. 
  	  ʹ��CPUģʽ�������
  	     * ���麯�����������ʵ�֣�ʹ��cpu����ǰ�����*/
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  /**
   * @brief Using the GPU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
         ʹ��GPU�����������������У�����ʹ��CPU����
   */
   /* void��������void����
   * Ϊʲô��ô���ã���Ϊ��ģ���ͳһ��
   * template<class T>
   * T default_value()
   * {
    	return T();
   * }
   * ����T����Ϊvoid
   */
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

  /**
   * @brief Using the CPU device, compute the gradients for any parameters and for the bottom blobs if propagate_down is true.
   *       ʹ��CPU�������ݶȽ��м��㣬�����������ȷ�ģ����bottom blob���м���
      * ���麯�������������ʵ��
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) = 0;
  /**
   * @brief Using the GPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   *        Fall back to Backward_cpu() if unavailable.
        ʹ��GPU��������ݶȣ�ͬ��
        ��������ã��򷵻�ʹ��CPU
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
       �ɸ�layer����setup�����bottom��top blob��������Ϊ���룬ƥ��ָ������������
       {ExactNum,Min,Max}{Bottom,Top}Blobs() functions.
   */
     // ��������/��� Blob ��Ŀ�Ƿ����� Layer Ҫ��
       // �����������blobs�ĸ����Ƿ��ڸ�����Χ��
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
       ���� SetUp����������ʼ�����κ�top blob�й�����Ȩ�أ���loss function��
       ��diff blob�д洢�����loss weights
   */
     // �ú����� Layer �� SetUp �����б����ã���ҪĿ���ǳ�ʼ���� Top Blob ��ص� loss Ȩ��
  // �ŵ� Top Blob �� diff ��ʵ���� Forward() ���� loss ����
  // loss_weight == 0����ʾ��ǰ�㲻���� loss �������㣬�󲿷� Layer ������һ��
  // loss_weight == 1����ʾ��ǰ����� loss �������㣬��ʧ�㣨LossLayer��������һ��
  inline void SetLossWeights(const vector<Blob<Dtype>*>& top) {
  	   // ��protobuffer�����л��Layer������������Ҫ��loss_weight����
    const int num_loss_weights = layer_param_.loss_weight_size();
    if (num_loss_weights) {// ���protobuffer�д�������һ��loss_weight����
      // loss_weight��������Ӧ���� Top Blob ��Ŀ��ͬ�����߲�Ҫloss_weight����
      CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
          "unspecified or specified once per top blob.";
       // ����ÿ�� Top Blob
      for (int top_id = 0; top_id < top.size(); ++top_id) {
      	   // ��ProtoBuffer�����л�ȡloss_weightʵ��ֵ��0 ���� 1��
        const Dtype loss_weight = layer_param_.loss_weight(top_id);
        // ��Ϊ0������
        if (loss_weight == Dtype(0)) { continue; }
        // ����Ϊ0������������������
        this->set_loss(top_id, loss_weight);// ���ؼ�¼loss_weightֵ
        const int count = top[top_id]->count();
        Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();
        // �� loss_weight ֵд�� Top Blob �� diff �򣬴��ݵ�������Ҫʹ�õĵط���ʵ��Զ��ͬ��
        caffe_set(count, loss_weight, loss_multiplier);
      }
    }
  }

 private:
  DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer

// Forward and backward wrappers. You should implement the cpu and gpu specific implementations instead,
// Forward and backward�������̣���Ӧ��ʵ��cpu��gpu�ľ���ʵ�֡�
//  and should not change these functions.
// ��Ӧ�øı���Щ����
// ǰ�򴫲����������򴫲�������װ������Ҫ�޸�����������
// ʹ��ʱֻ��Ҫ���������и�д Forward_cpu��Forward_gpu��Backward_cpu��Backward_gpu ����
// ǰ�򴫲��ͷ��򴫲��ӿڡ� ÿ��Layer�������඼Ӧ��ʵ��Forward_cpu()
template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype loss = 0;
  Reshape(bottom, top);
  switch (Caffe::mode()) { // �жϼ����豸
  case Caffe::CPU: // ��CPU��ִ��ǰ������
    Forward_cpu(bottom, top); // ����CPU���Forward����
    // ���� loss ֵ������еĻ���
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      // ��Ϊ LossLayer�����Ѿ�ͨ�� Forward ���������ȫ����ʧ����ֵ������ Top Blob data ��
      const Dtype* data = top[top_id]->cpu_data(); 
      // �� loss_weight ��Ϊ0�����Ѿ��� SetLossWeights �����н� loss Ȩ�ط��� Top Blob diff ��
      const Dtype* loss_weights = top[top_id]->cpu_diff();
      loss += caffe_cpu_dot(count, data, loss_weights);
    }
    break;
  case Caffe::GPU:  //GPUģʽ������ͬ��
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

// ���򴫲�������ֱ�ӵ��ö�Ӧ���豸����
template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  switch (Caffe::mode()) {  //ѡһ��
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
// ��LayerParameter��������ò��������л���Э�黺����
//Layer�����л�����,��layer�Ĳ�˵������layer_param_��
//��Ȩֵ��ƫ�ò���blobs_���Ƶ�LayerParameter���󣬱���д������
template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
   // ���Ʋ�Ȩֵ��ƫ�ò���blobs_
  for (int i = 0; i < blobs_.size(); ++i) {// Ȩֵ��ƫ����Ҳ�ᱣ��
    blobs_[i]->ToProto(param->add_blobs(), write_diff); // param->add_blobs() �᷵��һ�� BlobProto ָ��
  }
}

}  // namespace caffe

#endif  // CAFFE_LAYER_H_
