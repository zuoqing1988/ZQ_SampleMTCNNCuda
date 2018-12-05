#ifndef _ZQ_CPU_TENSOR4D_NCHW_H_
#define _ZQ_CPU_TENSOR4D_NCHW_H_
#pragma once

namespace ZQ
{
	template<class BaseType>
	class ZQ_CPU_Tensor4D_NCHW
	{
	public:
		BaseType* pData;
		int N, C, H, W;

		ZQ_CPU_Tensor4D_NCHW()
		{
			pData = 0;
			N = C = H = W = 0;
		}
		ZQ_CPU_Tensor4D_NCHW(const ZQ_CPU_Tensor4D_NCHW& other)
		{
			CopyData(other);
		}

		~ZQ_CPU_Tensor4D_NCHW()
		{
			if (pData)
			{
				delete[]pData;
				pData = 0;
			}
		}


		void CopyData(const ZQ_CPU_Tensor4D_NCHW& other)
		{
			ChangeSize(other.N, other.C, other.H, other.W);
			memcpy(pData, other.pData, sizeof(BaseType)*other.N*other.C*other.H*other.W);
		}

		int GetDataLength() const
		{
			return N*C*H*W;
		}

		void ChangeSize(int dst_N, int dst_C, int dst_H, int dst_W)
		{
			int cur_data_len = N*C*H*W;
			int dst_data_len = dst_N*dst_C*dst_H*dst_W;
			bool need_reallocate_pData = cur_data_len != dst_data_len;
			N = dst_N;
			C = dst_C;
			H = dst_H;
			W = dst_W;
			if (need_reallocate_pData)
			{
				if (pData)
					delete[]pData;
				pData = new BaseType[N*C*H*W];
			}
		}

		void Padding(int pad_H, int pad_W)
		{
			ZQ_CPU_Tensor4D_NCHW other;
			Padding(pad_H, pad_W, other);
			CopyData(other);
		}

		void Padding(int pad_H, int pad_W, ZQ_CPU_Tensor4D_NCHW& out) const
		{
			int pWidth = W + 2 * pad_W;
			int pHeight = H + 2 * pad_H;
			out.ChangeSize(N,C,pHeight, pWidth);

			BaseType *&p = out.pData;
			memset(p, 0, sizeof(BaseType)*N*C*pHeight*pWidth);

			for (int n = 0; n < N; n++)
			{
				for (int c = 0; c < C; c++)
				{
					for (int h = 0; h < H; h++)
					{
						const BaseType* src = pData + n*(C*H*W) + c*H*W + h*W;
						BaseType* dst = p + n*(C*pHeight*pWidth) + c*(pHeight*pWidth) + (h + pad_H)*pWidth + pad_W;
						memcpy(dst, src, sizeof(BaseType)*W);
					}
				}
			}	
		}

		/*make sure:
		bias's dim: (1,C,1,1)
		*/
		bool AddBias(const ZQ_CPU_Tensor4D_NCHW& bias)
		{
			if (bias.N != 1 || bias.C != C || bias.H != 1 || bias.W != 1)
				return false;
			
			for (int n = 0; n < N; n++)
			{
				BaseType *op = pData + n*C*H*W;
				const BaseType *pb = bias.pData;
				for (int c = 0; c < C; c++)
				{
					for (int i = 0; i < H*W; i++)
					{
						op[c*H*W + i] += pb[c];
					}
				}
			}
			return true;
		}

	};

	template<class BaseType>
	class ZQ_CPU_ConvolutionFilters_NCHW
	{
	public:
		ZQ_CPU_Tensor4D_NCHW<BaseType> filters;
		int pad_H;
		int pad_W;
		int stride_H;
		int stride_W;

		void ChangeSize(int dst_pad_H, int dst_pad_W, int dst_stride_H, int dst_stride_W, int dst_N, int dst_C, int dst_H, int dst_W)
		{
			pad_H = dst_pad_H;
			pad_W = dst_pad_W;
			stride_H = dst_stride_H;
			stride_W = dst_stride_W;
			filters.ChangeSize(dst_N, dst_C, dst_H, dst_W);
		}

		void CopyData(const ZQ_CPU_ConvolutionFilters_NCHW& other)
		{
			pad_H = other.pad_H;
			pad_W = other.pad_W;
			stride_H = other.stride_H;
			stride_W = other.stride_W;
			filters.CopyData(other.filters);
		}
	};
}
#endif
