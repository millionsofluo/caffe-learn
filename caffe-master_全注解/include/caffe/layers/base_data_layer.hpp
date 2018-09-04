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
 //  �������ݲ㣬������layer
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
 //  ��ʾ���캯��
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, 
  //  layersteup��ʵ��ͨ�����ݲ��setup
  //  and calls DataLayerSetUp to do special data layer setup for individual layer types.
  //  ���ҵ���datalayersetup��Ϊ��ͬ�Ĳ�����������������ݲ�����
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  //  �����ã�ʵ��ͨ�����ù��ܣ�֮�����DataLayerSetUp�������ݶ�ȡ����ر�����
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
//  ����ķ��򴫲�����Ҫ���κβ���
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
 //  ����Ԥ����任������
  TransformationParameter transform_param_;
  //  ����Ԥ����任��
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  //  �Ƿ������ǩ����
  bool output_labels_;
};

//  �������ݣ����ڴ�����ݶ�ȡ������
template <typename Dtype>
class Batch {
 public:
 //  ��������blob��data_���ڴ��ͼƬ���ݣ�label_���ڴ�ű�ǩ
  Blob<Dtype> data_, label_;
};

//  ��Ԥȡ���ܵ����ݶ�ȡ�㣬������BaseDataLayer��InternalThread
template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  //  ͬ��
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  virtual void InternalThreadEntry();  //  �ڲ��߳����
  virtual void load_batch(Batch<Dtype>* batch) = 0;  //  ������������

  vector<shared_ptr<Batch<Dtype> > > prefetch_;  //  Ԥȡbuffer
  BlockingQueue<Batch<Dtype>*> prefetch_free_;  //  ����Batch ����
  BlockingQueue<Batch<Dtype>*> prefetch_full_;  //  �Ѽ���batch ����
  Batch<Dtype>* prefetch_current_;

  Blob<Dtype> transformed_data_;  //  �任������
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
