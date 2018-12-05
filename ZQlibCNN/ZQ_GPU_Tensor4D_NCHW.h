#ifndef _ZQ_GPU_TENSOR4D_NCHW_H_
#define _ZQ_GPU_TENSOR4D_NCHW_H_
#pragma once

namespace ZQ
{
	class ZQ_GPU_Tensor4D_NCHW
	{
		friend class ZQ_CNN_Forward_CUDA;
		friend class ZQ_GPU_ConvolutionFilters_NCHW;
	public:
		ZQ_GPU_Tensor4D_NCHW();
		ZQ_GPU_Tensor4D_NCHW(const ZQ_GPU_Tensor4D_NCHW& other);
		~ZQ_GPU_Tensor4D_NCHW();
		int GetN()const { return N; }
		int GetC()const { return C; }
		int GetH()const { return H; }
		int GetW()const { return W; }
		int GetDataLength() const { return N*C*H*W; }
		void ChangeSize(int dst_N, int dst_C, int dst_H, int dst_W);
		void CopyData(const ZQ_GPU_Tensor4D_NCHW& other);
		bool Padding(int pad_H, int pad_W, ZQ_GPU_Tensor4D_NCHW& other) const;
	protected:
		int N, C, H, W;
		float* pData;
	};

	class ZQ_GPU_ConvolutionFilters_NCHW
	{
		friend class ZQ_CNN_Forward_CUDA;
	public:
		ZQ_GPU_ConvolutionFilters_NCHW();
		ZQ_GPU_ConvolutionFilters_NCHW(const ZQ_GPU_ConvolutionFilters_NCHW& other);
		~ZQ_GPU_ConvolutionFilters_NCHW();
		void ChangeSize(int dst_pad_H, int dst_pad_W, int dst_stride_H, int dst_stride_W, int dst_N, int dst_C, int dst_H, int dst_W);
		void CopyData(const ZQ_GPU_ConvolutionFilters_NCHW& other);
	protected:
		ZQ_GPU_Tensor4D_NCHW filters;
		int pad_H;
		int pad_W;
		int stride_H;
		int stride_W;
	};
}

#endif
