# ZQ_SampleMTCNNCuda

这是一个非完全优化的CUDA版MTCNN，可能存在BUG

从经济效益角度考虑，请使用[ZQCNN](https://github.com/zuoqing1988/ZQCNN)中的[ZQ_CNN_MTCNN](https://github.com/zuoqing1988/ZQCNN/tree/master/SamplesZQCNN/SampleMTCNN)

依赖：

	OpenCV3.1.0，其他版本需要自己更改项目include、lib路径
	CUDA 9.0， 其他版本需要用文本方式打开.vcxproj来更改CUDA版本
	cudnn 7， 请根据自己的cudnn7路径来更改项目include、lib路径