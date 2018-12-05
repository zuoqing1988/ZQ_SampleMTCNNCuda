#include "ZQ_GPU_Tensor4D_NCHW.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace ZQ
{
	ZQ_GPU_Tensor4D_NCHW::ZQ_GPU_Tensor4D_NCHW()
	{
		N = C = H = W = 0;
		pData = 0;
	}

	ZQ_GPU_Tensor4D_NCHW::ZQ_GPU_Tensor4D_NCHW(const ZQ_GPU_Tensor4D_NCHW& other)
	{
		CopyData(other);
	}

	ZQ_GPU_Tensor4D_NCHW::~ZQ_GPU_Tensor4D_NCHW()
	{
		if (pData)
		{
			cudaFree(pData);
			pData = 0;
		}
	}

	void ZQ_GPU_Tensor4D_NCHW::ChangeSize(int dst_N, int dst_C, int dst_H, int dst_W)
	{
		int cur_data_len = W*H*C*N;
		int dst_data_len = dst_W*dst_H*dst_C*dst_N;
		bool need_reallocate_pData = cur_data_len != dst_data_len;

		W = dst_W;
		H = dst_H;
		C = dst_C;
		N = dst_N;
		if (need_reallocate_pData)
		{
			if (pData)
				cudaFree(pData);
			if (dst_data_len != 0)
				cudaMalloc((void**)&pData, sizeof(float)*W*H*C*N);
			else
				pData = 0;
		}
	}

	void ZQ_GPU_Tensor4D_NCHW::CopyData(const ZQ_GPU_Tensor4D_NCHW& other)
	{
		ChangeSize(other.N, other.C, other.H, other.W);
		int data_len = GetDataLength();
		if (data_len > 0)
			cudaMemcpy(pData, other.pData, sizeof(float)*data_len, cudaMemcpyDeviceToDevice);
	}

	bool ZQ_GPU_Tensor4D_NCHW::Padding(int pad_H, int pad_W, ZQ_GPU_Tensor4D_NCHW& other) const
	{
		if (pad_H <= 0 && pad_W <= 0)
			return false;

		if (pad_W > 0)
		{
			int dst_H = H + pad_H * 2;
			int dst_W = W + pad_W * 2;
			other.ChangeSize(N, C, dst_H, dst_W);
			cudaMemset(other.pData, 0, sizeof(float)*N*C*dst_H*dst_W);
			for (int n = 0; n < N; n++)
			{
				for (int c = 0; c < C; c++)
				{
					for (int h = 0; h < H; h++)
					{
						float* src = pData + n*(C*H*W) + c*H*W + h*W;
						float* dst = other.pData + n*(C*dst_H*dst_W) + c*dst_H*dst_W + (h + pad_H)*dst_W + pad_W;
						cudaMemcpy(dst, src, sizeof(float)*W, cudaMemcpyDeviceToDevice);
					}
				}
			}
		}
		else
		{
			int dst_H = H + pad_H * 2;
			int dst_W = W + pad_W * 2;
			other.ChangeSize(N, C, dst_H, dst_W);
			cudaMemset(other.pData, 0, sizeof(float)*N*C*dst_H*dst_W);
			for (int n = 0; n < N; n++)
			{
				for (int c = 0; c < C; c++)
				{
					float* src = pData + n*(C*H*W) + c*H*W;
					float* dst = other.pData + n*(C*dst_H*dst_W) + c*dst_H*dst_W + pad_H*dst_W;
					cudaMemcpy(dst, src, sizeof(float)*H*W, cudaMemcpyDeviceToDevice);
				}
			}
		}
		return true;
	}

	/*******************************************/
	ZQ_GPU_ConvolutionFilters_NCHW::ZQ_GPU_ConvolutionFilters_NCHW()
	{
		pad_H = pad_W = stride_H = stride_W = 0;
	}
	ZQ_GPU_ConvolutionFilters_NCHW::ZQ_GPU_ConvolutionFilters_NCHW(const ZQ_GPU_ConvolutionFilters_NCHW& other)
	{
		CopyData(other);
	}
	ZQ_GPU_ConvolutionFilters_NCHW::~ZQ_GPU_ConvolutionFilters_NCHW()
	{

	}

	void ZQ_GPU_ConvolutionFilters_NCHW::ChangeSize(int dst_pad_H, int dst_pad_W, int dst_stride_H, int dst_stride_W, int dst_N, int dst_C, int dst_H, int dst_W)
	{
		pad_H = dst_pad_H;
		pad_W = dst_pad_W;
		stride_H = dst_stride_H;
		stride_W = dst_stride_W;
		filters.ChangeSize(dst_N, dst_C, dst_H, dst_W);
	}

	void ZQ_GPU_ConvolutionFilters_NCHW::CopyData(const ZQ_GPU_ConvolutionFilters_NCHW& other)
	{
		pad_H = other.pad_H;
		pad_W = other.pad_W;
		stride_H = other.stride_H;
		stride_W = other.stride_W;
		filters.CopyData(other.filters);
	}
}