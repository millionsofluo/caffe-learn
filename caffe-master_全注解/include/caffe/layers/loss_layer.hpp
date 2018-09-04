#ifndef CAFFE_LOSS_LAYER_HPP_
#define CAFFE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

/**
 * @brief An interface for Layer%s that take two Blob%s as input -- usually
 *        (1) predictions and (2) ground-truth labels -- and output a
 *        singleton Blob representing the loss.
 *    һ��������loss layer��Ҫ����blob��Ϊ����
         (1) Ԥ��ֵ (2) ��ʵ��ǩֵ
         ���һ��blob��Ϊloss
        
 * LossLayers are typically only capable of backpropagating to their first input
 * -- the predictions.
     losslayerͨ����Ϊ���򴫲��ĵ�һ������
 */
 
 //caffeʵ���˴�����loss function�����ǵĸ��඼�� LossLayer
 //  ��ʧ��İְ֣�������Layer
template <typename Dtype>
class LossLayer : public Layer<Dtype> {
 public:
 //  ��ʾ���캯��
  explicit LossLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
     //  �����ú���
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    //  ���κ���
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  //  ��������blob��Ϊ����
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  /**
   * @brief For convenience and backwards compatibility, instruct the Net to
   *        automatically allocate a single top Blob for LossLayers, into which
   *        they output their singleton loss, (even if the user didn't specify
   *        one in the prototxt, etc.).
   */
    //  Ϊ�˷��������ݣ�֪��netΪ��ʧ���Զ����䵥�����blob
    //  ��ʧ����Ὣ����ṹL(0)����������
  virtual inline bool AutoTopBlobs() const { return true; }
  //  ֻҪһ�����blob
  virtual inline int ExactNumTopBlobs() const { return 1; }
  /**
   * We usually cannot backpropagate to the labels; ignore force_backward for
   * these inputs.
      ����ͨ�����ܶԱ�ǩ���з�����㣬���Ժ���force_backward
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }
};

}  // namespace caffe

#endif  // CAFFE_LOSS_LAYER_HPP_
