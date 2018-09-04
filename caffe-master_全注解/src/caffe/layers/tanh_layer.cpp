// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/tanh_layer.hpp"

namespace caffe {

//  ǰ�򴫲�����
template <typename Dtype>
void TanHLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    	// ֻ��  ���ǰ������blob��dataָ��
  const Dtype* bottom_data = bottom[0]->cpu_data();
  	//��д  ������blob��dataָ��
  Dtype* top_data = top[0]->mutable_cpu_data();
  	// ֻ��  blob������
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
  	  //ֻ�� ��ú�һ���dataָ��
    const Dtype* top_data = top[0]->cpu_data();
    // ֻ��  ��ú�һ���diffָ��
    const Dtype* top_diff = top[0]->cpu_diff();
    // ��д  ���ǰһ���diffָ��
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    //  �õ�blob����
    const int count = bottom[0]->count();
    Dtype tanhx;
    for (int i = 0; i < count; ++i) {
    	//  �������
      tanhx = top_data[i];
      // ����ǰһ����ݶȣ����ں�һ��������Ե������õ�ǰһ������
      bottom_diff[i] = top_diff[i] * (1 - tanhx * tanhx);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TanHLayer);
#endif

INSTANTIATE_CLASS(TanHLayer);

}  // namespace caffe
