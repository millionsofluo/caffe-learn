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
 *    一个这样的loss layer需要两个blob作为输入
         (1) 预测值 (2) 真实标签值
         输出一个blob作为loss
        
 * LossLayers are typically only capable of backpropagating to their first input
 * -- the predictions.
     losslayer通常作为反向传播的第一个输入
 */
 
 //caffe实现了大量的loss function，它们的父类都是 LossLayer
 //  损失层的爸爸，派生于Layer
template <typename Dtype>
class LossLayer : public Layer<Dtype> {
 public:
 //  显示构造函数
  explicit LossLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
     //  层配置函数
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    //  变形函数
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  //  接受两个blob作为输入
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  /**
   * @brief For convenience and backwards compatibility, instruct the Net to
   *        automatically allocate a single top Blob for LossLayers, into which
   *        they output their singleton loss, (even if the user didn't specify
   *        one in the prototxt, etc.).
   */
    //  为了方便向后兼容，知道net为损失层自动分配单个输出blob
    //  损失层则会将计算结构L(0)保存在这里
  virtual inline bool AutoTopBlobs() const { return true; }
  //  只要一个输出blob
  virtual inline int ExactNumTopBlobs() const { return 1; }
  /**
   * We usually cannot backpropagate to the labels; ignore force_backward for
   * these inputs.
      我们通常不能对标签进行反向计算，所以忽略force_backward
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }
};

}  // namespace caffe

#endif  // CAFFE_LOSS_LAYER_HPP_
