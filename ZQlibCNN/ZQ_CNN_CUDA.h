#ifndef _ZQ_CNN_CUDA_H_
#define _ZQ_CNN_CUDA_H_
#pragma once


namespace ZQ_CNN_CUDA
{

	/*****************************************************************************/

	/*make sure:
	pBias 's dim: (1,C,1,1)
	*/
	float cuAddBias(int N, int C, int H, int W, float* pData, const float* pBias);

	/*make sure:
	pBias 's dim: (1,C,1,1)
	pPara 's dim: (1,C,1,1)
	*/
	float cuAddBiasPReLU(int N, int C, int H, int W, float* pData, const float* pBias, const float* pPara);

	/*
	over all channels, for each N,H,W
	*/
	float cuSoftmaxChannel(int N, int C, int H, int W, float* pData);

	/*make sure:
	dst_W = (src_W - filter_W) / stride_W + 1;
	dst_H = (src_H - filter_H) / stride_H + 1;
	dst_C = filter_N;
	dst_N = src_N;
	filter_C = src_C;
	*/
	float cuConvolutionNopadding(int src_N, int src_C, int src_H, int src_W, const float* src,
		int filter_N, int filter_C, int filter_H, int filter_W, int stride_H, int stride_W, const float* filters,
		int dst_N, int dst_C, int dst_H, int dst_W, float* dst);

	/*make sure:
	dst_W = ceil((float)(src_W - kernel_W) / stride_W + 1);
	dst_H = ceil((float)(src_H - kernel_H) / stride_H + 1);
	dst_C = src_C;
	dst_N = src_N;
	*/
	float cuMaxpooling(int src_N, int src_C, int src_H, int src_W, const float* src,
		int kernel_H, int kernel_W, int stride_H, int stride_W,
		int dst_N, int dst_C, int dst_H, int dst_W, float* dst);

	/*make sure:
	dst_N = src_N;
	dst_C = dst_C;
	*/
	float cuResizeBilinear(int src_N, int src_C, int src_H, int src_W, const float* src,
		int dst_N, int dst_C, int dst_H, int dst_W, float* dst);

	/*make sure:
	src_N = 1;
	dst_N = rect_N;
	dst_C = dst_C;
	rects: 4*rectN, arranged as [off_x, off_y, rect_W, rect_H,...]
	*/
	float cuResizeRectBilinear(int src_N, int src_C, int src_H, int src_W, const float* src,
		int rect_N, const int* rects,
		int dst_N, int dst_C, int dst_H, int dst_W, float* dst);
}

#endif
