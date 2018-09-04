.cpp 文件是用CPU计算是执行的文件

每一个.cpp都会判断要不要用GPU

#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

如果要，则执行相应的 .cu文件