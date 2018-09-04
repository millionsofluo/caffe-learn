#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {
 // �ȶ���sigmoid �������㷨  ���������tanh����
template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

//    ǰ�򴫲�����
template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    	// ���ǰ������blob��dataָ�� ֻ��
  const Dtype* bottom_data = bottom[0]->cpu_data();
	// ���ǰ�����blob��dataָ�� ��д
  Dtype* top_data = top[0]->mutable_cpu_data();
      	// ���blob��Ԫ�ظ���
  const int count = bottom[0]->count();
  	// ����sigmoid
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
}

// ���򴫲�����
template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    	// ���Ҫ���з��򴫲�
  if (propagate_down[0]) {
  	  // �õ���һ���dataָ��  ֻ��
    const Dtype* top_data = top[0]->cpu_data();
    //��ú�һ���diffָ��  ֻ��
    const Dtype* top_diff = top[0]->cpu_diff();
    // ���ǰһ���diffָ��  ��д
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // ��� blob������
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
    	// top_data ��ǰ�򴫲��׶μ����� ����x�����������ã����ټ�����
      const Dtype sigmoid_x = top_data[i];
      // ���������ʽ�󵼷��򣬺�һ��������ϵ��������õ�ǰһ����������ֱ�ӵ����˳�
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidLayer);
#endif

INSTANTIATE_CLASS(SigmoidLayer);


}  // namespace caffe
