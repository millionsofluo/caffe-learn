
//  caffeԴ���ļ�/src/caffe/SycedMem.cpp�����ļ���Ҫʵ��cpu��gpu���ڴ�ͬ��,�Ͷ�д��������

#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
/*����ע�ͷ��룺����
  ��Cuda���ò�����GPUģʽ�£���cudaMallocHost���Է���õ��̶����ڴ档
  ��������õ��ڴ治�ᱻ����DMA�����ڴ��ȡ���ƶ�̬ռ�ã�
  ����������ڴ���ڵ�һ��GPU��˵������̫����ô���
  �����ڲ���ѵ����˵���������,���䣬����������ڴ���������ߴ���ģ���ڶ�GPU�����µ��ȶ���*/
// �����gpuģʽ�£�����ʹ��cuda����ô�����ڴ����ҳ�����ķ�ʽ����
// ʹ��cudaMallocHost()���������ڵ�gpu�������������ԣ����Ƕ��ڶ�gpu��ǳ�����
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));// GPUģʽ����cuda�ṩ�⺯�������ڴ�
    *use_cuda = true;
    return;
  }
#endif
// MKL�� intel ����ѧ��
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size:1, 64);
#else
  *ptr = malloc(size);//��CPUģʽ����ͨ��c��malloc��������
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

// ��CaffeMallocHost��Ӧ
inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));//GPUģʽ����cuda�⺯��cudaFreeHost�ͷ��ڴ�
    return;
  }
#endif
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);//��cpuģʽ��C�⺯���ͷ��ڴ�
#endif
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU) and device (GPU).
 *        ����ฺ��������CPU�����豸��GPU��֮����ڴ�����ͬ��
 *
 * TODO(dox): more thorough description.  ��ȫ�������
 */
class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);  ////��explicit�ؼ��ֵģ��е��������Ĺ��캯����explicit��ֹ���������캯������ʽת��
  ~SyncedMemory();
  const void* cpu_data();    // ֻ����ȡ cpu data ���ط����cpu���ڴ��ַ:cpu_ptr_***********
  void set_cpu_data(void* data);  //����cpu data
  const void* gpu_data();  //  ֻ����ȡ gpu data ���ط����gpu���ڴ��ַ:gpu_ptr_
  void set_gpu_data(void* data); //����gpu data
  void* mutable_cpu_data();  // ��д���cpu data
  void* mutable_gpu_data();  // ��д���gpu data
  // ״̬����������ʾ����״̬��δ��ʼ����cpu������Ч��gpu������Ч����ͬ��
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  // ��õ�ǰ��״̬������
  SyncedHead head() { return head_; }
  //  ��õ�ǰ�洢�ռ�ĳߴ�
  size_t size() { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream); //async: �첽����ͬ��
#endif

 private:  //����ඨ���һЩ��
  void check_device();

  void to_cpu();  // ����ͬ������gpu��cpu
  void to_gpu();  // ����ͬ������cpu��gpu
  void* cpu_ptr_;  // λ��cpu������ָ�룬�ڴ�ָ��
  void* gpu_ptr_; // λ��gpu������ָ�룬�Դ�ָ��
  size_t size_;  // �洢�ռ��С�����ݴ�С
  SyncedHead head_;  // ״̬����������ǰ����״̬��UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED
  bool own_cpu_data_;  // ��־�Ƿ�ӵ��cpu���ݵ�����Ȩ�����񣬾��Ǵӱ�Ķ�����
  bool cpu_malloc_use_cuda_;  //
  bool own_gpu_data_;  // ��־�Ƿ�ӵ��gpu���ݵ�����Ȩ
  int device_; //gpu���豸��

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);/*��common.cpp����*/
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
