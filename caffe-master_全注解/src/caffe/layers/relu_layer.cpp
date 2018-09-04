#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    	//ֻ��  ��ȡ����blob��dataָ��
    	//  p = a -> b  ȡ��a�е�b��ֵ��p
  const Dtype* bottom_data = bottom[0]->cpu_data();
      //��д  ��ȡ���blob��dataָ��
  Dtype* top_data = top[0]->mutable_cpu_data();
      // ��ȡ����blobԪ�ظ���
  const int count = bottom[0]->count();
      // ReLU ������ ��layer_param �л�� �� Ĭ��Ϊ0 ������ͨ��ReLU
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
      // ִ��ReLU������ ������Ϊ negative_slope=0 
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    	// ���Ҫ���򴫲�
  if (propagate_down[0]) {
  	  // ��ȡǰһ���dataָ��  ֻ��
    const Dtype* bottom_data = bottom[0]->cpu_data();
        // ��ȡ��һ���diffָ��  ֻ��
    const Dtype* top_diff = top[0]->cpu_diff();
    	//  ��ȡǰһ���diffָ��  ��д
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    	//  ��ȡ��Ҫ��������Ԫ���ܸ���
    const int count = bottom[0]->count();
         // ReLU ������ ��layer_param �л�� negative_slope Ĭ��Ϊ0 ������ͨ��ReLU
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
    	// (bottom_data[i] > 0) ����ReLU�ĵ�������������ʽ�󵼷��򣬺�һ��������Ե������õ�ǰһ������
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
