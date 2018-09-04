#include "caffe/layer.hpp"

namespace caffe {
//模板显示实例化
INSTANTIATE_CLASS(Layer);

}  // namespace caffe


/*
	可见Layer大部分函数都没有实现，只是虚函数，真正实现实在派生类中，
	具体的代码再  src/caffe/layers/*.cpp   */
	
/*
	注意
	在使用layer之前，需要包含头文件caffe/layer.hpp  
	再通过using namespace caffe 来使用命名空间caffe
	否则编译时会报错*/