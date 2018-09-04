// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/tanh_layer.hpp"

namespace caffe {

//  前向传播函数
template <typename Dtype>
void TanHLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    	// 只读  获得前向输入blob的data指针
  const Dtype* bottom_data = bottom[0]->cpu_data();
  	//读写  获得输出blob的data指针
  Dtype* top_data = top[0]->mutable_cpu_data();
  	// 只读  blob的总数
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = tanh(bottom_data[i]);
  }
}

template <typename Dtype>
void TanHLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
  	  //只读 获得后一层的data指针
    const Dtype* top_data = top[0]->cpu_data();
    // 只读  获得后一层的diff指针
    const Dtype* top_diff = top[0]->cpu_diff();
    // 读写  获得前一层的diff指针
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    //  得到blob总数
    const int count = bottom[0]->count();
    Dtype tanhx;
    for (int i = 0; i < count; ++i) {
    	//  结果复用
      tanhx = top_data[i];
      // 计算前一层的梯度，等于后一层的误差乘以导函数得到前一层的误差
      bottom_diff[i] = top_diff[i] * (1 - tanhx * tanhx);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TanHLayer);
#endif

INSTANTIATE_CLASS(TanHLayer);

}  // namespace caffe
