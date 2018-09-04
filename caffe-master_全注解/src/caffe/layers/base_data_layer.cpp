#include <boost/thread.hpp>
#include <vector>
//  数据读取层的实现
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {
//  构造函数，初始化layer参数，数据变换器参数
template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

//  basedatalayer层设置
template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {  //  判断输出blob个数，若只为1输出data，为2输出data和label
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  //  初始化数据变换器对象
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();  //  生成随机种子数
  // The subclasses should setup the size of bottom and top
  //  子类负责设置top blob形状
  DataLayerSetUp(bottom, top);
}

//  BasePrefetchingDataLayer构造函数
template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_(param.data_param().prefetch()),
      prefetch_free_(), prefetch_full_(), prefetch_current_() {
  for (int i = 0; i < prefetch_.size(); ++i) {
    prefetch_[i].reset(new Batch<Dtype>());
    prefetch_free_.push(prefetch_[i].get());//  将Batch对象放入空闲队列
  }
}

//  BasePrefetchingDataLayer层配置函数
template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);

  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  //  在开启数据预取线程前，通过调用Blob相应的函数先行cudaMalloc
  //  避免在多线程情况下同时cudaMalloc，会导致CUDA API调用失败
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
  StartInternalThread();  //  开启内部预取线程
  DLOG(INFO) << "Prefetch initialized.";
}

//  内部线程入口
template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
	//  创建cuda stream
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {  //  循环载入批量数据
      Batch<Dtype>* batch = prefetch_free_.pop();   //  拿到一个空闲Batch
      load_batch(batch);  //  载入批量数据
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        if (this->output_labels_) {
          batch->label_.data().get()->async_gpu_push(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));  //  同步到gpu
      }
#endif
      prefetch_full_.push(batch);  //  加入到带负载的batch队列
    }
  } catch (boost::thread_interrupted&) {  //  捕获异常，退出
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));  //  销毁cuda stream
  }
#endif
}

//  前向传播函数
template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    	//  从带负载的batch队列中取出一个batch对象
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
 //  top blob根据batch形状变形
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    //  top blob根据batch中的label_形状变形
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
