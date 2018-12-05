#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ZQ_CNN_Forward_CUDA.h"
#include <omp.h>
#include "cudnn.h"


cudnnHandle_t cudnn;

namespace ZQ
{
	void ZQ_CNN_Forward_CUDA::InitCUDNN()
	{
		cudnnCreate(&cudnn);
	}

	void ZQ_CNN_Forward_CUDA::ShutDownCUDNN()
	{
		cudnnDestroy(cudnn);
	}

	bool ZQ_CNN_Forward_CUDA::Upload(const ZQ_CPU_Tensor4D_NCHW<float>& cpu_tensor, ZQ_GPU_Tensor4D_NCHW& gpu_tensor)
	{
		int size = cpu_tensor.N*cpu_tensor.C*cpu_tensor.H*cpu_tensor.W;
		if (size > 0)
		{
			gpu_tensor.ChangeSize(cpu_tensor.N, cpu_tensor.C, cpu_tensor.H, cpu_tensor.W);
			cudaMemcpy(gpu_tensor.pData, cpu_tensor.pData, sizeof(float)*size, cudaMemcpyHostToDevice);
			return true;
		}
		return false;
	}

	bool ZQ_CNN_Forward_CUDA::Upload(const ZQ_CPU_ConvolutionFilters_NCHW<float>& cpu_filters, ZQ_GPU_ConvolutionFilters_NCHW& gpu_filters)
	{
		gpu_filters.pad_H = cpu_filters.pad_H;
		gpu_filters.pad_W = cpu_filters.pad_W;
		gpu_filters.stride_H = cpu_filters.stride_H;
		gpu_filters.stride_W = cpu_filters.stride_W;
		return Upload(cpu_filters.filters, gpu_filters.filters);
	}

	bool ZQ_CNN_Forward_CUDA::Download(const ZQ_GPU_Tensor4D_NCHW& gpu_tensor, ZQ_CPU_Tensor4D_NCHW<float>& cpu_tensor)
	{
		int size = gpu_tensor.N*gpu_tensor.C*gpu_tensor.H*gpu_tensor.W;
		if (size > 0)
		{
			cpu_tensor.ChangeSize(gpu_tensor.N, gpu_tensor.C, gpu_tensor.H, gpu_tensor.W);
			cudaMemcpy(cpu_tensor.pData, gpu_tensor.pData, sizeof(float)*size, cudaMemcpyDeviceToHost);
			return true;
		}
		return false;
	}

	bool ZQ_CNN_Forward_CUDA::Download(const ZQ_GPU_ConvolutionFilters_NCHW& gpu_filters, ZQ_CPU_ConvolutionFilters_NCHW<float>& cpu_filters)
	{
		cpu_filters.pad_H = gpu_filters.pad_H;
		cpu_filters.pad_W = gpu_filters.pad_W;
		cpu_filters.stride_H = gpu_filters.stride_H;
		cpu_filters.stride_W = gpu_filters.stride_W;
		return Download(gpu_filters.filters, cpu_filters.filters);
	}

	bool ZQ_CNN_Forward_CUDA::Resize(const ZQ_GPU_Tensor4D_NCHW& input, int dst_H, int dst_W, ZQ_GPU_Tensor4D_NCHW& output, float& cudacosttime)
	{
		int N = input.N;
		int C = input.C;
		if (!(N > 0 && C > 0 && input.H >= 2 && input.W >= 2 && dst_H > 0 && dst_W > 0))
			return false;
		output.ChangeSize(N, C, dst_H, dst_W);
		cudacosttime = ZQ_CNN_CUDA::cuResizeBilinear(N, C, input.H, input.W, input.pData, N, C, dst_H, dst_W, output.pData);
		return true;
	}

	bool ZQ_CNN_Forward_CUDA::ResizeRect(const ZQ_GPU_Tensor4D_NCHW& input, const std::vector<int>& rects, int dst_H, int dst_W, ZQ_GPU_Tensor4D_NCHW& output, float& cudacosttime)
	{
		if (rects.size() % 4 != 0)
			return false;
		int num_rects = rects.size() / 4;
		if (num_rects == 0)
			return false;
		if (input.N != 1 || dst_H < 2 || dst_W < 2)
			return false;
		for (int n = 0; n < num_rects; n++)
		{
			if (rects[n * 4 + 0] < 0 || rects[n * 4 + 1] < 0 || rects[n * 4 + 0] + rects[n * 4 + 2] > input.W || rects[n * 4 + 1] + rects[n * 4 + 3] > input.H)
				return false;
		}
		int* d_rects = 0;
		cudaMalloc((void**)&d_rects, sizeof(int)*num_rects * 4);
		cudaMemcpy(d_rects, &rects[0], sizeof(int)*num_rects * 4, cudaMemcpyHostToDevice);
		output.ChangeSize(num_rects, input.C, dst_H, dst_W);
		cudacosttime = ZQ_CNN_CUDA::cuResizeRectBilinear(input.N, input.C, input.H, input.W, input.pData, num_rects, d_rects, num_rects, input.C, dst_H, dst_W, output.pData);
		cudaFree(d_rects);
		return true;
	}

	bool ZQ_CNN_Forward_CUDA::Convolution(const ZQ_GPU_Tensor4D_NCHW& input, const ZQ_GPU_ConvolutionFilters_NCHW& filters, ZQ_GPU_Tensor4D_NCHW& output, float& cudacosttime)
	{
		if (input.C != filters.filters.C)
			return false;

		if (filters.pad_H != 0 || filters.pad_W != 0)
		{
			ZQ_GPU_Tensor4D_NCHW other;
			input.Padding(filters.pad_H, filters.pad_W, other);
			int out_W = (other.W + 2 * filters.pad_W - filters.filters.W) / filters.stride_W + 1;
			int out_H = (other.H + 2 * filters.pad_H - filters.filters.H) / filters.stride_H + 1;
			int out_C = filters.filters.N;
			int out_N = other.N;
			output.ChangeSize(out_N, out_C, out_H, out_W);
			cudacosttime = ZQ_CNN_CUDA::cuConvolutionNopadding(other.N, other.C, other.H, other.W, other.pData,
				filters.filters.N, filters.filters.C, filters.filters.H, filters.filters.W, filters.stride_H, filters.stride_W, filters.filters.pData,
				out_N, out_C, out_H, out_W, output.pData);
			return true;
		}
		else
		{
			int out_W = (input.W + 2 * filters.pad_W - filters.filters.W) / filters.stride_W + 1;
			int out_H = (input.H + 2 * filters.pad_H - filters.filters.H) / filters.stride_H + 1;
			int out_C = filters.filters.N;
			int out_N = input.N;
			output.ChangeSize(out_N, out_C, out_H, out_W);

			cudacosttime = ZQ_CNN_CUDA::cuConvolutionNopadding(input.N, input.C, input.H, input.W, input.pData,
				filters.filters.N, filters.filters.C, filters.filters.H, filters.filters.W, filters.stride_H, filters.stride_W, filters.filters.pData,
				out_N, out_C, out_H, out_W, output.pData);
			return true;
		}
	}

	bool ZQ_CNN_Forward_CUDA::ConvolutionCUDNN(const ZQ_GPU_Tensor4D_NCHW& input, const ZQ_GPU_ConvolutionFilters_NCHW& filters, ZQ_GPU_Tensor4D_NCHW& output, float& cudacosttime)
	{
#ifdef ZQ_CNN_USE_CUDNN
		if (input.C != filters.filters.C)
			return false;

		if (filters.pad_H != 0 || filters.pad_W != 0)
		{
			ZQ_GPU_Tensor4D_NCHW other;
			input.Padding(filters.pad_H, filters.pad_W, other);
			int out_W = (other.W + 2 * filters.pad_W - filters.filters.W) / filters.stride_W + 1;
			int out_H = (other.H + 2 * filters.pad_H - filters.filters.H) / filters.stride_H + 1;
			int out_C = filters.filters.N;
			int out_N = other.N;
			output.ChangeSize(out_N, out_C, out_H, out_W);
		}
		else
		{
			int out_W = (input.W + 2 * filters.pad_W - filters.filters.W) / filters.stride_W + 1;
			int out_H = (input.H + 2 * filters.pad_H - filters.filters.H) / filters.stride_H + 1;
			int out_C = filters.filters.N;
			int out_N = input.N;
			output.ChangeSize(out_N, out_C, out_H, out_W);
		}


		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		cudnnTensorDescriptor_t input_descriptor;
		cudnnCreateTensorDescriptor(&input_descriptor);
		cudnnSetTensor4dDescriptor(input_descriptor,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*batch_size=*/input.N,
			/*channels=*/input.C,
			/*image_height=*/input.H,
			/*image_width=*/input.W);

		cudnnFilterDescriptor_t kernel_descriptor;
		cudnnCreateFilterDescriptor(&kernel_descriptor);
		cudnnSetFilter4dDescriptor(kernel_descriptor,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*out_channels=*/filters.filters.N,
			/*in_channels=*/filters.filters.C,
			/*kernel_height=*/filters.filters.H,
			/*kernel_width=*/filters.filters.W);

		cudnnConvolutionDescriptor_t convolution_descriptor;
		cudnnCreateConvolutionDescriptor(&convolution_descriptor);
		cudnnSetConvolution2dDescriptor(convolution_descriptor,
			/*pad_height=*/filters.pad_H,
			/*pad_width=*/filters.pad_W,
			/*vertical_stride=*/filters.stride_H,
			/*horizontal_stride=*/filters.stride_W,
			/*dilation_height=*/1,
			/*dilation_width=*/1,
			/*mode=*/CUDNN_CROSS_CORRELATION,
			/*computeType=*/CUDNN_DATA_FLOAT);

		int batch_size = input.N, channels = input.C, height = input.H, width = input.W;
		cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
			input_descriptor,
			kernel_descriptor,
			&batch_size,
			&channels,
			&height,
			&width);

		cudnnTensorDescriptor_t output_descriptor;
		cudnnCreateTensorDescriptor(&output_descriptor);
		cudnnSetTensor4dDescriptor(output_descriptor,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*batch_size=*/batch_size,
			/*channels=*/channels,
			/*image_height=*/output.H,
			/*image_width=*/output.W);

		cudnnConvolutionFwdAlgo_t convolution_algorithm;
		cudnnGetConvolutionForwardAlgorithm(cudnn,
			input_descriptor,
			kernel_descriptor,
			convolution_descriptor,
			output_descriptor,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			/*memoryLimitInBytes=*/0,
			&convolution_algorithm);

		size_t workspace_bytes = 0;
		cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			input_descriptor,
			kernel_descriptor,
			convolution_descriptor,
			output_descriptor,
			convolution_algorithm,
			&workspace_bytes);

		void* d_workspace{ nullptr };
		cudaMalloc(&d_workspace, workspace_bytes);

		float alpha = 1, beta = 0;
		cudnnConvolutionForward(cudnn,
			&alpha,
			input_descriptor,
			input.pData,
			kernel_descriptor,
			filters.filters.pData,
			convolution_descriptor,
			convolution_algorithm,
			d_workspace,
			workspace_bytes,
			&beta,
			output_descriptor,
			output.pData);


		cudaFree(d_workspace);

		cudnnDestroyTensorDescriptor(input_descriptor);
		cudnnDestroyTensorDescriptor(output_descriptor);
		cudnnDestroyFilterDescriptor(kernel_descriptor);
		cudnnDestroyConvolutionDescriptor(convolution_descriptor);
		
		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&cudacosttime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
#else
		Convolution(input, const filters, output, cudacosttime);
#endif
		return true;
	}

	bool ZQ_CNN_Forward_CUDA::FullConnect(const ZQ_GPU_Tensor4D_NCHW &input, const ZQ_GPU_ConvolutionFilters_NCHW &weight, ZQ_GPU_Tensor4D_NCHW &output, float& cudacosttime)
	{
		return Convolution(input, weight, output,cudacosttime);
	}

	void ZQ_CNN_Forward_CUDA::MaxPooling(const ZQ_GPU_Tensor4D_NCHW &input, ZQ_GPU_Tensor4D_NCHW &output, int kernel_H, int kernel_W, int stride_H, int stride_W, float& cudacosttime)
	{
		int out_N = input.N;
		int out_C = input.C;
		int out_H = ceil((float)(input.H - kernel_H) / stride_H + 1);
		int out_W = ceil((float)(input.W - kernel_W) / stride_W + 1);
		output.ChangeSize(out_N, out_C, out_H, out_W);
		cudacosttime = ZQ_CNN_CUDA::cuMaxpooling(input.N, input.C, input.H, input.W, input.pData, kernel_H, kernel_W, stride_H, stride_W,
			out_N, out_C, out_H, out_W, output.pData);
	}

	bool ZQ_CNN_Forward_CUDA::PReLU(ZQ_GPU_Tensor4D_NCHW &input, const ZQ_GPU_Tensor4D_NCHW& bias, const ZQ_GPU_Tensor4D_NCHW& para, float& cudacosttime)
	{
		if (bias.N != 1 || bias.C != input.C || bias.H != 1 || bias.W != 1
			|| para.N != 1 || para.C != input.C || para.H != 1 || para.W != 1)
			return false;
		cudacosttime = ZQ_CNN_CUDA::cuAddBiasPReLU(input.N, input.C, input.H, input.W, input.pData, bias.pData, para.pData);
		return true;
	}

	bool ZQ_CNN_Forward_CUDA::AddBias(ZQ_GPU_Tensor4D_NCHW &input, const ZQ_GPU_Tensor4D_NCHW& bias, float& cudacosttime)
	{
		if (bias.N != 1 || bias.C != input.C || bias.H != 1 || bias.W != 1)
			return false;
		cudacosttime = ZQ_CNN_CUDA::cuAddBias(input.N, input.C, input.H, input.W, input.pData, bias.pData);
		return true;
	}

	void ZQ_CNN_Forward_CUDA::SoftmaxChannel(ZQ_GPU_Tensor4D_NCHW &input, float& cudacosttime)
	{
		cudacosttime = ZQ_CNN_CUDA::cuSoftmaxChannel(input.N, input.C, input.H, input.W, input.pData);
	}
}