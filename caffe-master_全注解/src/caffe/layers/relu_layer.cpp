#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    	//只读  获取输入blob的data指针
    	//  p = a -> b  取出a中的b赋值给p
  const Dtype* bottom_data = bottom[0]->cpu_data();
      //读写  获取输出blob的data指针
  Dtype* top_data = top[0]->mutable_cpu_data();
      // 获取输入blob元素个数
  const int count = bottom[0]->count();
      // ReLU 参数， 从layer_param 中获得 ， 默认为0 ，即普通的ReLU
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
      // 执行ReLU操作， 可以认为 negative_slope=0 
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    	// 如果要反向传播
  if (propagate_down[0]) {
  	  // 获取前一层的data指针  只读
    const Dtype* bottom_data = bottom[0]->cpu_data();
        // 获取后一层的diff指针  只读
    const Dtype* top_diff = top[0]->cpu_diff();
    	//  获取前一层的diff指针  读写
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    	//  获取需要参与计算的元素总个数
    const int count = bottom[0]->count();
         // ReLU 参数， 从layer_param 中获得 negative_slope 默认为0 ，即普通的ReLU
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
    	// (bottom_data[i] > 0) 就是ReLU的导函数，根据链式求导法则，后一层的误差乘以导函数得到前一层的误差
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
