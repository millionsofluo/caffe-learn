#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
 //  基本数据层，派生于layer
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
 //  显示构造函数
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, 
  //  layersteup：实现通用数据层的setup
  //  and calls DataLayerSetUp to do special data layer setup for individual layer types.
  //  并且调用datalayersetup来为不同的层类型设置特殊的数据层设置
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  //  层配置，实现通用配置功能，之后调用DataLayerSetUp进行数据读取层的特别配置
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
//  这里的反向传播不需要做任何操作
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
 //  数据预处理变换器参数
  TransformationParameter transform_param_;
  //  数据预处理变换器
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  //  是否输出标签数据
  bool output_labels_;
};

//  批量数据，用于存放数据读取层的输出
template <typename Dtype>
class Batch {
 public:
 //  包含两个blob：data_用于存放图片数据，label_用于存放标签
  Blob<Dtype> data_, label_;
};

//  带预取功能的数据读取层，派生于BaseDataLayer和InternalThread
template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  //  同上
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  virtual void InternalThreadEntry();  //  内部线程入口
  virtual void load_batch(Batch<Dtype>* batch) = 0;  //  载入批量数据

  vector<shared_ptr<Batch<Dtype> > > prefetch_;  //  预取buffer
  BlockingQueue<Batch<Dtype>*> prefetch_free_;  //  空闲Batch 队列
  BlockingQueue<Batch<Dtype>*> prefetch_full_;  //  已加载batch 队列
  Batch<Dtype>* prefetch_current_;

  Blob<Dtype> transformed_data_;  //  变换后数据
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
