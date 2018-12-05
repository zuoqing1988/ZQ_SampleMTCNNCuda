#ifndef _ZQ_CNN_FORWARD_CUDA_H_
#define _ZQ_CNN_FORWARD_CUDA_H_
#pragma once

#include "ZQ_CPU_Tensor4D_NCHW.h"
#include "ZQ_GPU_Tensor4D_NCHW.h"
#include "ZQ_CNN_CUDA.h"
#include <math.h>
#include <vector>

#define ZQ_CNN_USE_CUDNN

namespace ZQ
{
	class ZQ_CNN_Forward_CUDA
	{
	public:
		static void InitCUDNN();
		
		static void ShutDownCUDNN();

		static bool Upload(const ZQ_CPU_Tensor4D_NCHW<float>& cpu_tensor, ZQ_GPU_Tensor4D_NCHW& gpu_tensor);
		
		static bool Upload(const ZQ_CPU_ConvolutionFilters_NCHW<float>& cpu_filters, ZQ_GPU_ConvolutionFilters_NCHW& gpu_filters);

		static bool Download(const ZQ_GPU_Tensor4D_NCHW& gpu_tensor, ZQ_CPU_Tensor4D_NCHW<float>& cpu_tensor);

		static bool Download(const ZQ_GPU_ConvolutionFilters_NCHW& gpu_filters, ZQ_CPU_ConvolutionFilters_NCHW<float>& cpu_filters);

		static bool Resize(const ZQ_GPU_Tensor4D_NCHW& input, int dst_H, int dst_W, ZQ_GPU_Tensor4D_NCHW& output, float& cudacosttime);

		static bool ResizeRect(const ZQ_GPU_Tensor4D_NCHW& input, const std::vector<int>& rects, int dst_H, int dst_W, ZQ_GPU_Tensor4D_NCHW& output, float& cudacosttime);

		static bool Convolution(const ZQ_GPU_Tensor4D_NCHW& input, const ZQ_GPU_ConvolutionFilters_NCHW& filters, ZQ_GPU_Tensor4D_NCHW& output, float& cudacosttime);

		static bool ConvolutionCUDNN(const ZQ_GPU_Tensor4D_NCHW& input, const ZQ_GPU_ConvolutionFilters_NCHW& filters, ZQ_GPU_Tensor4D_NCHW& output, float& cudacosttime);

		static bool FullConnect(const ZQ_GPU_Tensor4D_NCHW &input, const ZQ_GPU_ConvolutionFilters_NCHW &weight, ZQ_GPU_Tensor4D_NCHW &output, float& cudacosttime);

		static void MaxPooling(const ZQ_GPU_Tensor4D_NCHW &input, ZQ_GPU_Tensor4D_NCHW &output, int kernel_H, int kernel_W, int stride_H, int stride_W, float& cudacosttime);

		static bool PReLU(ZQ_GPU_Tensor4D_NCHW &input, const ZQ_GPU_Tensor4D_NCHW& bias, const ZQ_GPU_Tensor4D_NCHW& para, float& cudacosttime);

		static bool AddBias(ZQ_GPU_Tensor4D_NCHW &input, const ZQ_GPU_Tensor4D_NCHW& bias, float& cudacosttime);

		static void SoftmaxChannel(ZQ_GPU_Tensor4D_NCHW &input, float& cudacosttime);
	};
}
#endif
