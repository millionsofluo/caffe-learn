#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {
 // 先定义sigmoid 函数的算法  这里调用了tanh函数
template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

//    前向传播函数
template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    	// 获得前向输入blob的data指针 只读
  const Dtype* bottom_data = bottom[0]->cpu_data();
	// 获得前向输出blob的data指针 读写
  Dtype* top_data = top[0]->mutable_cpu_data();
      	// 获得blob的元素个数
  const int count = bottom[0]->count();
  	// 计算sigmoid
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
}

// 反向传播函数
template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    	// 如果要进行反向传播
  if (propagate_down[0]) {
  	  // 得到后一层的data指针  只读
    const Dtype* top_data = top[0]->cpu_data();
    //获得后一层的diff指针  只读
    const Dtype* top_diff = top[0]->cpu_diff();
    // 获得前一层的diff指针  读写
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // 获得 blob的总数
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
    	// top_data 是前向传播阶段计算结果 发（x），这里重用，减少计算量
      const Dtype sigmoid_x = top_data[i];
      // 这里根据链式求导法则，后一层的误差乘上导函数，得到前一层的误差，这里就直接导好了乘
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidLayer);
#endif

INSTANTIATE_CLASS(SigmoidLayer);


}  // namespace caffe
