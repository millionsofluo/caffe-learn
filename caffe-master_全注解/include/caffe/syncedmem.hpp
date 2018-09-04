
//  caffe源码文件/src/caffe/SycedMem.cpp，该文件主要实现cpu与gpu的内存同步,和读写操作？。

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
/*下面注释翻译：：：
  在Cuda可用并且在GPU模式下，用cudaMallocHost可以分配得到固定的内存。
  这样分配好的内存不会被例如DMA这种内存存取机制动态占用，
  这样分配的内存对于单一的GPU来说不会有太大的用处，
  但对于并行训练来说他会更有用,尤其，这样分配的内存能显著提高大型模型在多GPU情形下的稳定性*/
// 如果在gpu模式下，且能使用cuda，那么主机内存会以页锁定的方式分配
// 使用cudaMallocHost()函数，对于单gpu性能提升不明显，但是对于多gpu会非常明显
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));// GPU模式下用cuda提供库函数分配内存
    *use_cuda = true;
    return;
  }
#endif
// MKL是 intel 的数学库
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size:1, 64);
#else
  *ptr = malloc(size);//单CPU模式下则通过c的malloc函数分配
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

// 与CaffeMallocHost对应
inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));//GPU模式下用cuda库函数cudaFreeHost释放内存
    return;
  }
#endif
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);//单cpu模式用C库函数释放内存
#endif
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU) and device (GPU).
 *        这个类负责主机（CPU）和设备（GPU）之间的内存分配和同步
 *
 * TODO(dox): more thorough description.  更全面的描述
 */
class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);  ////带explicit关键字的，有单个参数的构造函数，explicit禁止单参数构造函数的隐式转换
  ~SyncedMemory();
  const void* cpu_data();    // 只读获取 cpu data 返回分配的cpu的内存地址:cpu_ptr_***********
  void set_cpu_data(void* data);  //设置cpu data
  const void* gpu_data();  //  只读获取 gpu data 返回分配的gpu的内存地址:gpu_ptr_
  void set_gpu_data(void* data); //设置gpu data
  void* mutable_cpu_data();  // 读写获得cpu data
  void* mutable_gpu_data();  // 读写获得gpu data
  // 状态机变量，表示四种状态：未初始化，cpu数据有效，gpu数据有效，已同步
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  // 获得当前的状态机变量
  SyncedHead head() { return head_; }
  //  获得当前存储空间的尺寸
  size_t size() { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream); //async: 异步，非同步
#endif

 private:  //这个类定义的一些量
  void check_device();

  void to_cpu();  // 数据同步，由gpu至cpu
  void to_gpu();  // 数据同步，由cpu至gpu
  void* cpu_ptr_;  // 位于cpu的数据指针，内存指针
  void* gpu_ptr_; // 位于gpu的数据指针，显存指针
  size_t size_;  // 存储空间大小，数据大小
  SyncedHead head_;  // 状态机变量，当前数据状态，UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED
  bool own_cpu_data_;  // 标志是否拥有cpu数据的所有权，（否，就是从别的对象共享）
  bool cpu_malloc_use_cuda_;  //
  bool own_gpu_data_;  // 标志是否拥有gpu数据的所有权
  int device_; //gpu的设备号

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);/*见common.cpp解析*/
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
