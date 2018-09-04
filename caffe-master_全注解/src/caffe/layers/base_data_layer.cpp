#include <boost/thread.hpp>
#include <vector>
//  ���ݶ�ȡ���ʵ��
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {
//  ���캯������ʼ��layer���������ݱ任������
template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

//  basedatalayer������
template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {  //  �ж����blob��������ֻΪ1���data��Ϊ2���data��label
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  //  ��ʼ�����ݱ任������
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();  //  �������������
  // The subclasses should setup the size of bottom and top
  //  ���ฺ������top blob��״
  DataLayerSetUp(bottom, top);
}

//  BasePrefetchingDataLayer���캯��
template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_(param.data_param().prefetch()),
      prefetch_free_(), prefetch_full_(), prefetch_current_() {
  for (int i = 0; i < prefetch_.size(); ++i) {
    prefetch_[i].reset(new Batch<Dtype>());
    prefetch_free_.push(prefetch_[i].get());//  ��Batch���������ж���
  }
}

//  BasePrefetchingDataLayer�����ú���
template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);

  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  //  �ڿ�������Ԥȡ�߳�ǰ��ͨ������Blob��Ӧ�ĺ�������cudaMalloc
  //  �����ڶ��߳������ͬʱcudaMalloc���ᵼ��CUDA API����ʧ��
  for (int i = 0; i < prefetch_.size(); ++i) {
    prefetch_[i]->data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i]->label_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < prefetch_.size(); ++i) {
      prefetch_[i]->data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i]->label_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();  //  �����ڲ�Ԥȡ�߳�
  DLOG(INFO) << "Prefetch initialized.";
}

//  �ڲ��߳����
template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
	//  ����cuda stream
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {  //  ѭ��������������
      Batch<Dtype>* batch = prefetch_free_.pop();   //  �õ�һ������Batch
      load_batch(batch);  //  ������������
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        if (this->output_labels_) {
          batch->label_.data().get()->async_gpu_push(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));  //  ͬ����gpu
      }
#endif
      prefetch_full_.push(batch);  //  ���뵽�����ص�batch����
    }
  } catch (boost::thread_interrupted&) {  //  �����쳣���˳�
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));  //  ����cuda stream
  }
#endif
}

//  ǰ�򴫲�����
template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    	//  �Ӵ����ص�batch������ȡ��һ��batch����
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
 //  top blob����batch��״����
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    //  top blob����batch�е�label_��״����
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
