
//   include/caffe/blob.hpp
//   封装了SyncedMemory类，作为基本的计算单元

#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"//实例化caffe类，并且封装了boost和cuda随机数生成的函数，提供了统一接口
#include "caffe/proto/caffe.pb.h"//  由protoc生成的头文件，声明了BlobProto，BlobShape等遵循caffe.proto协议的数据结构
#include "caffe/syncedmem.hpp"   //cpu,gpu共享内存类，用于数据同步

/*
主要是分配内存和释放内存。class yncedMemory定义了内存分配管理和CPU与GPU之间同步的函数。
 Blob会使用SyncedMem自动决定什么时候去copy data以提高运行效率，通常情况是仅当gnu或cpu修改后有copy操作。
 */
const int kMaxBlobAxes = 32;   //blob的最大维数//在头文件中为它添加 extern 声明,以使其能被多个文件共享


namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic computational unit through which Layer%s, Net%s, and Solver%s interact.
 *        作为基本计算单元的封装器，他是一种基本的计算单元，通过 layer net solver相互作用
 *       
 * TODO(dox): more thorough description.  
 *      更全面的描述
 */
 //  这里就是类模板的使用了
template <typename Dtype>  //模板类，虚拟类型Dtype
class Blob {
 public:
 //  默认构造函数
  Blob()//构造函数：初始化列表 {空函数体}
       : data_(), diff_(), count_(0), capacity_(0) {}  //显示构造函数，避免隐式数据转换 capacity（容量）

  /// @brief Deprecated（弃用）; use <code>Blob(const vector<int>& shape)</code>.
  explicit Blob(const int num, const int channels, const int height,
      const int width);//可以通过设置数据维度（N,C,H,W）初始化
  //const 传递过来的参数在函数内不可以改变(无意义，因为本身就是形参)
   //const引用参数在函数内为常量不可变
  explicit Blob(const vector<int>& shape);  //也可以通过传入vector<int>直接传入维数

  /// @brief Deprecated（弃用）;; use <code>Reshape(const vector<int>& shape)</code>.
   // 变形函数，根据输入参数重新设置当前Blob形状，必要时重新分配内存
  void Reshape(const int num, const int channels, const int height,
      const int width);
  /**
   * @brief Change the dimensions of the blob, allocating new memory if necessary.
   *        改变blob的大小，如果需要的话，重新分配内存
   *
   * This function can be called both to create an initial allocation of memory, 
   * 可以调用这个函数在创建时分配初始内存
   * and to adjust the dimensions of a top blob during Layer::Reshape or Layer::Forward. 
   * 在Layer::Reshape 或 Layer::Forward时调整top blob的尺寸
   * When changing the size of blob, memory will only be reallocated if sufficient memory does not already exist,
   * 当改变blob的尺寸时，内存不足就重新分配
   *  and excess memory will never be freed.
   *  溢出的内存则不会被释放
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is an error; 
   * 注意，调整输入blob的尺寸后立即调用Net::Backward是错误的
   * either Net::Forward or Net::Reshape need to be called to propagate the new input shape to higher layers.
   * 要么Net::Forward or Net::Reshape被调用，将新的输入图形传播到更高的层次
   */
   // 变形函数，根据输入的参数重新设置当前的blob，必要时重新分配内存
  void Reshape(const vector<int>& shape);  
  void Reshape(const BlobShape& shape);
  void ReshapeLike(const Blob& other);
  //  得到blob的形状字符串用于打印log，类似于“Top shape ：100  1  28  28  （78400）”
  //内联函数 通过内联函数，编译器不需要跳转到内存其他地址去执行函数调用，也不需要保留函数调用时的现场数据。
  // const 成员函数，任何不会修改数据成员的函数都应该声明为const 类型。

  // 输出blob的形状
  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }
  //  返回blob形状
  inline const vector<int>& shape() const { return shape_; }
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th axis from the end, if index is negative).
   *        返回index-th的尺寸，
   *
   * @param index the axis index, which may be negative as it will be"canonicalized" using CanonicalAxisIndex.
   *        维数可能为负，就将他规范化
   *        Dies on out of range index.
   */
   //  返回某一维度的尺寸
  inline int shape(int index) const { //根据索引返回维数，对于维数(N,C,H,W),shape(0)返回N,shape(-1)返回W。
    return shape_[CanonicalAxisIndex(index)];//Canonical(规范化) Axis（维）
  }
  //  返回维度数目
   //返回Blob维度数，对于维数(N,C,H,W)，返回4
  inline int num_axes() const { return shape_.size(); }
  //  返回blob中的元素总数，  
  //返回Blob维度数，对于维数(N,C,H,W)，返回N×C×H×W
  inline int count() const { return count_; }

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions among a range of axes.
   *        计算维度乘积
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
   //  返回blob中某几维自己的元素总数
     //对于维数(N,C,H,W)，count(0, 3)返回N×C×H
       // 计算从第start_axis维到第end_axis维的slice的volume
  inline int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);    //  保证start_axis <= end_axis
    CHECK_GE(start_axis, 0);    //  保证start_axis >= 0
    CHECK_GE(end_axis, 0);    //  保证 end_axis >= 0
    CHECK_LE(start_axis, num_axes());    //  保证 start_axis <= 总维数
    CHECK_LE(end_axis, num_axes());    //  保证 end_axis <= 总维数
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }
  /**
   * @brief Compute the volume of a slice spanning from a particular first axis to the final axis.
   *       从某一维度开始计算元素总数
   *
   * @param start_axis The first axis to include in the slice.
   */
     //对于维数(N,C,H,W)，count(1)返回C×H×W
  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
   //  转换坐标轴索引[ -N , N ) 为普通索引[ 0 , N )
  inline int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes())   //GE:  左边 <= 右边
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())    //LT:  左边 < 右边
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

    //  获取形状中某一维的尺寸
  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  inline int num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int channels() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int height() const { return LegacyShape(2); }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int width() const { return LegacyShape(3); }
  inline int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4)
        << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse indexing) 
      // -- this special case simulates the one-padding used to fill extraneous axes of legacy blobs.
      // 
      return 1;
    }
    return shape(index);
  }
  // 下面几个函数都是计算偏移量的
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {  //计算物理偏移量，(n,c,h,w)的偏移量为((n∗C+c)∗H+h)∗W+w
    CHECK_GE(n, 0);  //  GE  左边 <= 右边
    CHECK_LE(n, num());  //  LE  左边 >= 右边
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }

  inline int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }
  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape of other (and die otherwise);
   *         if true, Reshape this Blob to other's shape if necessary
   *        
   */
   //  按值拷贝blob到当前blob
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);
  //  这几个函数都是存取器
    // 以下几个函数都是获取元素函数
  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }

  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }

  inline Dtype data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }

  inline Dtype diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }

  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

  /*
  	// 假定数据在 CPU 上进行初始化，我们有一个 blob
	const Dtype* foo;
	Dtype* bar;
	foo = blob.gpu_data(); // 数据从 CPU 复制到 GPU
	foo = blob.cpu_data(); // 没有数据复制，两者都有最新的内容
	bar = blob.mutable_gpu_data(); // 没有数据复制
	// ... 一些操作 ...
	bar = blob.mutable_gpu_data(); // 仍在 GPU，没有数据复制
	foo = blob.cpu_data(); // 由于 GPU 修改了数值，数据从 GPU 复制到 CPU
	foo = blob.gpu_data(); //没有数据复制，两者都有最新的内容
	bar = blob.mutable_cpu_data(); // 依旧没有数据复制
	bar = blob.mutable_gpu_data(); //数据从 CPU 复制到 GPU
	bar = blob.mutable_cpu_data(); //数据从 GPU 复制到 CPU

   */

  const Dtype* cpu_data() const;  // 只访问cpu data  //数据访问，const方式只读，不允许改写数据
  void set_cpu_data(Dtype* data);  // 设置 cpu data 
  const int* gpu_shape() const;  // 
  const Dtype* gpu_data() const;  // 只访问 gpu data
  void set_gpu_data(Dtype* data);  // 设置 gpu data
  const Dtype* cpu_diff() const;  // 只访问 cpu diff
  const Dtype* gpu_diff() const;  // 只访问 gpu diff
  Dtype* mutable_cpu_data();  //  读写访问 cup data //mutable(易变)方式可改写数据（对diff_的访问也是类似的）
  Dtype* mutable_gpu_data();  //同
  Dtype* mutable_cpu_diff();  // 读写访问 cpu diff
  Dtype* mutable_gpu_diff();  //同
  void Update();   //blob 更新运算，可简单理解为data与diff的merge过程
  //  反序列化函数，从blobproto中恢复一个blob对象
  //从proto读数据进来，其实就是反序列化
  void FromProto(const BlobProto& proto, bool reshape = true);
  //  序列化函数，将内存中的blob对象保存到blobproto中
   //blob数据保存到proto中
  void ToProto(BlobProto* proto, bool write_diff = false) const;

  /// @brief Compute the sum of absolute values (L1 norm) of the data.  计算data的L-1范数
  Dtype asum_data() const;
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.  diff   L-1范数
  Dtype asum_diff() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the data.  L-2范数
  Dtype sumsq_data() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.  L-2范数
  Dtype sumsq_diff() const;

  /// @brief Scale the blob data by a constant factor.  data乘以一个标量
  void scale_data(Dtype scale_factor);
  /// @brief Scale the blob diff by a constant factor.  diff 乘以一个标量
  void scale_diff(Dtype scale_factor);

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);  //  共享另一个blob的data
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
   //Blob& other 赋值给diff_
  void ShareDiff(const Blob& other);  //共享另一个blob的diff

  bool ShapeEquals(const BlobProto& other);
///////////////////////////////重要//////////////////////////////////////////
 protected:
  shared_ptr<SyncedMemory> data_;    // 存放指向data的指针//存储前向传递数据
  shared_ptr<SyncedMemory> diff_;    // 存放指向diff的指针//存储反向传递梯度
  shared_ptr<SyncedMemory> shape_data_;    // 存放指向shape_data的指针
  vector<int> shape_;    // //参数维度
  int count_;    // 存放有效元素数目信息//Blob存储的元素个数（shape_所有元素乘积）
  int capacity_;    // 存放blob容器的容量信息//当前Blob的元素个数（控制动态分配）

  DISABLE_COPY_AND_ASSIGN(Blob);  //禁止拷贝构造函数，赋值运算符重载
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
