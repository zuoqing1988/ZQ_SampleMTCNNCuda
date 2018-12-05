#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include "ZQ_CNN_CUDA.cuh"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

namespace ZQ_CNN_CUDA
{
	/*make sure:
	pBias 's dim: (1,C,1,1)
	*/
	__global__ void add_bias_kernel(int N, int C, int H, int W, float* pData, const float* pBias)
	{
		int id = blockIdx.x*blockDim.x + threadIdx.x;

		int HW = H*W;
		int NHW = N*HW;
		int CHW = C*HW;
		if (id >= NHW)
			return;
		int n = id / HW;
		int p = id%HW;
		for (int c = 0; c < C; c++)
		{
			pData[n*CHW + c*HW + p] += pBias[c];
		}
	}


	/*make sure:
	pBias 's dim: (1,C,1,1)
	pPara 's dim: (1,C,1,1)
	*/
	__global__ void add_bias_PReLU_kernel(int N, int C, int H, int W, float* pData, const float* pBias, const float* pPara)
	{
		int id = blockIdx.x*blockDim.x + threadIdx.x;

		int HW = H*W;
		int NHW = N*HW;
		int CHW = C*HW;
		if (id >= NHW)
			return;
		int n = id / HW;
		int p = id%HW;
		for (int c = 0; c < C; c++)
		{
			float tmp = pData[n*CHW + c*HW + p] + pBias[c];
			if (tmp < 0)
				tmp *= pPara[c];
			pData[n*CHW + c*HW + p] = tmp;
		}
	}

	/*
	over all channels, for each N,H,W
	*/
	__global__ void softmax_channel_kernel(int N, int C, int H, int W, float* pData)
	{
		int id = blockIdx.x*blockDim.x + threadIdx.x;

		int HW = H*W;
		int NHW = N*HW;
		int CHW = C*HW;
		if (id >= NHW)
			return;
		int n = id / HW;
		int p = id%HW;
		float sum = 0;
		for (int c = 0; c < C; c++)
		{
			float tmp = pData[n*CHW + c*HW + p];
			tmp = exp(tmp);
			sum += tmp;
		}

		if (sum != 0)
		{
			for (int c = 0; c < C; c++)
			{
				float tmp = pData[n*CHW + c*HW + p];
				tmp = exp(tmp);
				pData[n*CHW + c*HW + p] = tmp / sum;
			}
		}
	}

	/*make sure:
	dst_W = (src_W - filter_W) / stride_W + 1;
	dst_H = (src_H - filter_H) / stride_H + 1;
	dst_C = filter_N;
	dst_N = src_N;
	filter_C = src_C;
	*/
	__global__ void convolution_nopadding_kernel(int src_N, int src_C, int src_H, int src_W, const float* src,
		int filter_N, int filter_C, int filter_H, int filter_W, int stride_H, int stride_W, const float* filters,
		int dst_N, int dst_C, int dst_H, int dst_W, float* dst)
	{
		int id = blockIdx.x*blockDim.x + threadIdx.x;
		int dst_HW = dst_H*dst_W;
		int dst_CHW = dst_C*dst_HW;
		int dst_NCHW = dst_N*dst_CHW;
		if (id >= dst_NCHW)
			return;
		int dst_n = id / dst_CHW;
		int id_1 = id%dst_CHW;
		int dst_c = id_1 / dst_HW;
		int id_2 = id_1%dst_HW;
		int dst_h = id_2 / dst_W;
		int dst_w = id_2%dst_W;
		int filter_CHW = filter_C*filter_H*filter_W;
		int filter_HW = filter_H*filter_W;
		int src_CHW = src_C*src_H*src_W;
		int src_HW = src_H*src_W;
		int src_h = dst_h*stride_H;
		int src_w = dst_w*stride_W;

		const float* cur_filter = filters + dst_c*filter_CHW;
		const float* cur_src = src + dst_n*src_CHW;
		float* cur_dst = dst + dst_n*dst_CHW;
		float sum = 0;
		for (int c = 0; c < src_C; c++)
		{
			for (int h = 0; h < filter_H; h++)
			{
				for (int w = 0; w < filter_W; w++)
				{
					sum += cur_filter[c*filter_HW + h*filter_W + w] * cur_src[c*src_HW + (src_h + h)*src_W + src_w + w];
				}
			}
		}
		cur_dst[dst_c*dst_HW+dst_h*dst_W+dst_w] = sum;
	}

	/*make sure:
	dst_W = ceil((float)(src_W - kernel_W) / stride_W + 1);
	dst_H = ceil((float)(src_H - kernel_H) / stride_H + 1);
	dst_C = src_C;
	dst_N = src_N;
	*/
	__global__ void maxpooling_good_divide_kernel(int src_N, int src_C, int src_H, int src_W, const float* src,
		int kernel_H, int kernel_W, int stride_H, int stride_W,
		int dst_N, int dst_C, int dst_H, int dst_W, float* dst)
	{
		int id = blockIdx.x*blockDim.x + threadIdx.x;
		int dst_HW = dst_H*dst_W;
		int dst_CHW = dst_C*dst_HW;
		int dst_NCHW = dst_N*dst_CHW;
		if (id >= dst_NCHW)
			return;
		int dst_n = id / dst_CHW;
		int id_1 = id%dst_CHW;
		int dst_c = id_1 / dst_HW;
		int id_2 = id_1%dst_HW;
		int dst_h = id_2 / dst_W;
		int dst_w = id_2%dst_W;
		int src_CHW = src_C*src_H*src_W;
		int src_HW = src_H*src_W;
		int src_h = dst_h*stride_H;
		int src_w = dst_w*stride_W;

		const float* cur_src = src + dst_n*src_CHW + dst_c*src_HW + src_h*src_W +src_w;
		float* cur_dst = dst + dst_n*dst_CHW + dst_c*dst_HW + dst_h*dst_W + dst_w;
		float max_val = cur_src[0];
		for (int w = 1; w < kernel_W; w++)
		{
			max_val = __max(max_val, cur_src[w]);
		}
		for (int h = 1; h < kernel_H; h++)
		{
			for (int w = 0; w < kernel_W; w++)
			{
				max_val = __max(max_val, cur_src[h*src_W + w]);
			}
		}
		cur_dst[0] = max_val;
	}

	/*make sure:
	dst_W = ceil((float)(src_W - kernel_W) / stride_W + 1);
	dst_H = ceil((float)(src_H - kernel_H) / stride_H + 1);
	dst_C = src_C;
	dst_N = src_N;
	*/
	__global__ void maxpooling_bad_divide_kernel(int src_N, int src_C, int src_H, int src_W, const float* src,
		int kernel_H, int kernel_W, int stride_H, int stride_W,
		int dst_N, int dst_C, int dst_H, int dst_W, float* dst)
	{
		int id = blockIdx.x*blockDim.x + threadIdx.x;
		int dst_HW = dst_H*dst_W;
		int dst_CHW = dst_C*dst_HW;
		int dst_NCHW = dst_N*dst_CHW;
		if (id >= dst_NCHW)
			return;
		int dst_n = id / dst_CHW;
		int id_1 = id%dst_CHW;
		int dst_c = id_1 / dst_HW;
		int id_2 = id_1%dst_HW;
		int dst_h = id_2 / dst_W;
		int dst_w = id_2%dst_W;
		int src_CHW = src_C*src_H*src_W;
		int src_HW = src_H*src_W;
		int src_h = dst_h*stride_H;
		int src_w = dst_w*stride_W;

		const float* cur_src = src + dst_n*src_CHW + dst_c*src_HW + src_h*src_W + src_w;
		float* cur_dst = dst + dst_n*dst_CHW + dst_c*dst_HW + dst_h*dst_W + dst_w;
		float max_val = cur_src[0];

		int max_kH = __min(kernel_H, src_H - dst_h*stride_H);
		int max_kW = __min(kernel_W, src_W - dst_w*stride_W);
		for (int w = 1; w < max_kW; w++)
		{
			max_val = __max(max_val, cur_src[w]);
		}
		for (int h = 1; h < max_kH; h++)
		{
			for (int w = 0; w < max_kW; w++)
			{
				max_val = __max(max_val, cur_src[h*src_W + w]);
			}
		}
		cur_dst[0] = max_val;
	}
	
	/*make sure:
	dst_N = src_N;
	dst_C = dst_C;
	*/
	__global__ void resize_bilinear_kernel(int src_N, int src_C, int src_H, int src_W, const float* src,
		int dst_N, int dst_C, int dst_H, int dst_W, float* dst)
	{
		int id = blockIdx.x*blockDim.x + threadIdx.x;
		int dst_HW = dst_H*dst_W;
		int dst_CHW = dst_C*dst_HW;
		int dst_NCHW = dst_N*dst_CHW;
		if (id >= dst_NCHW)
			return;
		int dst_n = id / dst_CHW;
		int id_1 = id%dst_CHW;
		int dst_c = id_1 / dst_HW;
		int id_2 = id_1%dst_HW;
		int dst_h = id_2 / dst_W;
		int dst_w = id_2%dst_W;
		int src_CHW = src_C*src_H*src_W;
		int src_HW = src_H*src_W;

		float coord_x = (dst_w + 0.5f) / dst_W*src_W - 0.5f;
		float coord_y = (dst_h + 0.5f) / dst_H*src_H - 0.5f;

		int x0 = floor(coord_x);
		int x1 = x0 + 1;
		int y0 = floor(coord_y);
		int y1 = y0 + 1;

		float sx = coord_x - x0;
		float sy = coord_y - y0;
		int real_x0 = __max(0, __min(src_W - 1, x0));
		int real_x1 = __max(0, __min(src_W - 1, x1));
		int real_y0 = __max(0, __min(src_H - 1, y0));
		int real_y1 = __max(0, __min(src_H - 1, y1));

		const float* cur_src = src + dst_n*src_CHW + dst_c*src_HW;
		float* cur_dst = dst + dst_n*dst_CHW + dst_c*dst_HW + dst_h*dst_W + dst_w;

		float val = 0;
		val += cur_src[real_y0*src_W + real_x0] * (1.0f - sx)*(1.0f - sy);
		val += cur_src[real_y0*src_W + real_x1] * sx *(1.0f - sy);
		val += cur_src[real_y1*src_W + real_x0] * (1.0f - sx)*      sy;
		val += cur_src[real_y1*src_W + real_x1] * sx *      sy;

		cur_dst[0] = val;
	}

	/*make sure:
	src_N = 1;
	dst_N = rect_N;
	dst_C = dst_C;
	rects: 4*rectN, arranged as [off_x, off_y, rect_W, rect_H,...]
	*/
	__global__ void resize_rect_bilinear_kernel(int src_N, int src_C, int src_H, int src_W, const float* src,
		int rect_N, const int* rects,
		int dst_N, int dst_C, int dst_H, int dst_W, float* dst)
	{
		int id = blockIdx.x*blockDim.x + threadIdx.x;
		int dst_HW = dst_H*dst_W;
		int dst_CHW = dst_C*dst_HW;
		int dst_NCHW = dst_N*dst_CHW;
		if (id >= dst_NCHW)
			return;
		int dst_n = id / dst_CHW;
		int id_1 = id%dst_CHW;
		int dst_c = id_1 / dst_HW;
		int id_2 = id_1%dst_HW;
		int dst_h = id_2 / dst_W;
		int dst_w = id_2%dst_W;
		int src_HW = src_H*src_W;

		int rect_offX = rects[dst_n * 4 + 0];
		int rect_offY = rects[dst_n * 4 + 1];
		int rect_W = rects[dst_n * 4 + 2];
		int rect_H = rects[dst_n * 4 + 3];

		float coord_x = (dst_w + 0.5f) / dst_W*rect_W - 0.5f;
		float coord_y = (dst_h + 0.5f) / dst_H*rect_H - 0.5f;

		int x0 = floor(coord_x);
		int x1 = x0 + 1;
		int y0 = floor(coord_y);
		int y1 = y0 + 1;

		float sx = coord_x - x0;
		float sy = coord_y - y0;
		int real_x0 = __max(0, __min(src_W - 1, x0)) + rect_offX;
		int real_x1 = __max(0, __min(src_W - 1, x1)) + rect_offX;
		int real_y0 = __max(0, __min(src_H - 1, y0)) + rect_offY;
		int real_y1 = __max(0, __min(src_H - 1, y1)) + rect_offY;

		const float* cur_src = src + dst_c*src_HW;
		float* cur_dst = dst + dst_n*dst_CHW + dst_c*dst_HW + dst_h*dst_W + dst_w;

		float val = 0;
		val += cur_src[real_y0*src_W + real_x0] * (1.0f - sx)*(1.0f - sy);
		val += cur_src[real_y0*src_W + real_x1] * sx *(1.0f - sy);
		val += cur_src[real_y1*src_W + real_x0] * (1.0f - sx)*      sy;
		val += cur_src[real_y1*src_W + real_x1] * sx *      sy;

		cur_dst[0] = val;
	}

	/*****************************************************************/

	/*make sure:
	pBias 's dim: (1,C,1,1)
	*/
	float cuAddBias(int N, int C, int H, int W, float* pData, const float* pBias)
	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE, 1);
		dim3 gridSize((N*H*W - 1 + blockSize.x) / blockSize.x, 1);
		add_bias_kernel << <gridSize, blockSize >> > (N, C, H, W, pData, pBias);

		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		float costtime;
		cudaEventElapsedTime(&costtime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return costtime;
	}

	
	/*make sure:
	pBias 's dim: (1,C,1,1)
	pPara 's dim: (1,C,1,1)
	*/
	float cuAddBiasPReLU(int N, int C, int H, int W, float* pData, const float* pBias, const float* pPara)
	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE, 1);
		dim3 gridSize((N*H*W - 1 + blockSize.x) / blockSize.x, 1);
		add_bias_PReLU_kernel << <gridSize, blockSize >> > (N, C, H, W, pData, pBias, pPara);

		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		float costtime;
		cudaEventElapsedTime(&costtime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return costtime;
	}

	/*
	over all channels, for each N,H,W
	*/
	float cuSoftmaxChannel(int N, int C, int H, int W, float* pData)
	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE, 1);
		dim3 gridSize((N*H*W - 1 + blockSize.x) / blockSize.x, 1);
		softmax_channel_kernel << <gridSize, blockSize >> > (N,C,H,W, pData);

		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		float costtime;
		cudaEventElapsedTime(&costtime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return costtime;
	}

	/*make sure:
	dst_W = (src_W - filter_W) / stride_W + 1;
	dst_H = (src_H - filter_H) / stride_H + 1;
	dst_C = filter_N;
	dst_N = src_N;
	filter_C = src_C;
	*/
	float cuConvolutionNopadding(int src_N, int src_C, int src_H, int src_W, const float* src,
		int filter_N, int filter_C, int filter_H, int filter_W, int stride_H, int stride_W, const float* filters,
		int dst_N, int dst_C, int dst_H, int dst_W, float* dst)
	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE, 1);
		dim3 gridSize((dst_N*dst_C*dst_H*dst_W - 1 + blockSize.x) / blockSize.x, 1);
		convolution_nopadding_kernel << <gridSize, blockSize >> > (src_N, src_C, src_H, src_W, src, 
			filter_N, filter_C, filter_H, filter_W, stride_H, stride_W, filters, 
			dst_N, dst_C, dst_H, dst_W, dst);
		
		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		float costtime;
		cudaEventElapsedTime(&costtime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return costtime;
	}

	/*make sure:
	dst_W = ceil((float)(src_W - kernel_W) / stride_W + 1);
	dst_H = ceil((float)(src_H - kernel_H) / stride_H + 1);
	dst_C = src_C;
	dst_N = src_N;
	*/
	float cuMaxpooling(int src_N, int src_C, int src_H, int src_W, const float* src,
		int kernel_H, int kernel_W, int stride_H, int stride_W,
		int dst_N, int dst_C, int dst_H, int dst_W, float* dst)
	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE, 1);
		dim3 gridSize((dst_N*dst_C*dst_H*dst_W - 1 + blockSize.x) / blockSize.x, 1);
		if ((src_W - kernel_W) % stride_W == 0 && (src_H - kernel_H) % stride_H == 0)
		{
			maxpooling_good_divide_kernel << <gridSize, blockSize >> > (src_N,src_C,src_H,src_W,src,
				kernel_H,kernel_W,stride_H,stride_W,dst_N,dst_C,dst_H,dst_W,dst);
		}
		else
		{
			maxpooling_bad_divide_kernel << <gridSize, blockSize >> > (src_N, src_C, src_H, src_W, src,
				kernel_H, kernel_W, stride_H, stride_W, dst_N, dst_C, dst_H, dst_W, dst);
		}
		
		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		float costtime;
		cudaEventElapsedTime(&costtime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return costtime;
	}
	
	/*make sure:
	dst_N = src_N;
	dst_C = dst_C;
	*/
	float cuResizeBilinear(int src_N, int src_C, int src_H, int src_W, const float* src,
		int dst_N, int dst_C, int dst_H, int dst_W, float* dst)
	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE, 1);
		dim3 gridSize((dst_N*dst_C*dst_H*dst_W - 1 + blockSize.x) / blockSize.x, 1);
		resize_bilinear_kernel << <gridSize, blockSize >> > (src_N, src_C, src_H, src_W, src, dst_N, dst_C, dst_H, dst_W, dst);
		
		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		float costtime;
		cudaEventElapsedTime(&costtime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return costtime;
	}

	/*make sure:
	src_N = 1;
	dst_N = rect_N;
	dst_C = dst_C;
	rects: 4*rectN, arranged as [off_x, off_y, rect_W, rect_H,...]
	*/
	float cuResizeRectBilinear(int src_N, int src_C, int src_H, int src_W, const float* src,
		int rect_N, const int* rects,
		int dst_N, int dst_C, int dst_H, int dst_W, float* dst)
	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE, 1);
		dim3 gridSize((dst_N*dst_C*dst_H*dst_W - 1 + blockSize.x) / blockSize.x, 1);
		resize_rect_bilinear_kernel << <gridSize, blockSize >> > (src_N, src_C, src_H, src_W, src, rect_N, rects, dst_N, dst_C, dst_H, dst_W, dst);

		cudaDeviceSynchronize();
		
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		float costtime;
		cudaEventElapsedTime(&costtime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return costtime;
	}
}