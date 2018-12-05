#ifndef _ZQ_CNN_FORWARD_H_
#define _ZQ_CNN_FORWARD_H_
#pragma once
#include "ZQ_CPU_Tensor4D_NCHW.h"
#include "ZQ_CNN_BBox.h"
#include <vector>
#include <algorithm>
#include <fstream>
#include <stdio.h>

namespace ZQ
{
	class ZQ_CNN_Forward
	{
	public:
		template<class T>
		static bool Convolution(const ZQ_CPU_Tensor4D_NCHW<T>& input, const ZQ_CPU_ConvolutionFilters_NCHW<T>& filters, ZQ_CPU_Tensor4D_NCHW<T>& output)
		{
			if (filters.pad_H != 0 || filters.pad_W != 0)
			{
				ZQ_CPU_Tensor4D_NCHW<T> padded;
				input.Padding(filters.pad_H, filters.pad_W, padded);
				return _conv_no_padding(padded, filters, output);
			}
			else
				return _conv_no_padding(input, filters, output);
		}

		template<class T>
		static bool FullConnect(const ZQ_CPU_Tensor4D_NCHW<T>& input, const ZQ_CPU_ConvolutionFilters_NCHW<T>& filters, ZQ_CPU_Tensor4D_NCHW<T>& output)
		{
			return _full_connect(input, filters, output);
		}

		template<class T>
		static void MaxPooling(const ZQ_CPU_Tensor4D_NCHW<T> &input, ZQ_CPU_Tensor4D_NCHW<T> &output, int kernel_H, int kernel_W, int stride_H, int stride_W)
		{
			_max_pooling(input, output, kernel_H, kernel_W, stride_H, stride_W);
		}


		template<class T>
		static bool PReLU(ZQ_CPU_Tensor4D_NCHW<T> &input, const ZQ_CPU_Tensor4D_NCHW<T>& bias, const ZQ_CPU_Tensor4D_NCHW<T>& para)
		{
			return _prelu(input, bias, para);
		}

		template<class T>
		static void SoftmaxChannel(ZQ_CPU_Tensor4D_NCHW<T> &input)
		{
			_softmax_channel(input);
		}

		static void NMS(std::vector<ZQ_CNN_BBox> &boundingBox, std::vector<ZQ_CNN_OrderScore> &bboxScore, const float overlap_threshold, const std::string& modelname = "Union", int overlap_num_thresh = 0)
		{
			_nms(boundingBox, bboxScore, overlap_threshold, modelname, overlap_num_thresh);
		}

		static void RefineAndSquareBbox(std::vector<ZQ_CNN_BBox> &vecBbox, const int width, const int height)
		{
			_refine_and_square_bbox(vecBbox, width, height);
		}

		template<class T>
		static bool ReadData(const std::string& filename, int num, const long dataNumber[], T * const pTeam[])
		{
			return _read_data(filename, num, dataNumber, pTeam);
		}

		template<class T>
		static bool ReadDataBinary(const std::string& filename, int num, long dataNumber[], T *pTeam[])
		{
			return _read_data_binary(filename, num, dataNumber, pTeam);
		}

		template<class T>
		static bool SaveDataBinary(const std::string& filename, int num, long dataNumber[], T* pTeam[])
		{
			return _save_data_binary(filename, num, dataNumber, pTeam);
		}

	private:

		template<class T>
		static bool _conv_no_padding(const ZQ_CPU_Tensor4D_NCHW<T>& input, const ZQ_CPU_ConvolutionFilters_NCHW<T>& filters, ZQ_CPU_Tensor4D_NCHW<T>& output)
		{
			if (filters.filters.C != input.C)
				return false;

			int out_W = (input.W + 2 * filters.pad_W - filters.filters.W) / filters.stride_W + 1;
			int out_H = (input.H + 2 * filters.pad_H - filters.filters.H) / filters.stride_H + 1;
			int out_C = filters.filters.N;
			int out_N = input.N;
			output.ChangeSize(out_N, out_C, out_H, out_W);

			int in_HW = input.H*input.W;
			int in_CHW = input.C*in_HW;
			int out_HW = output.H*output.W;
			int out_CHW = output.C*out_HW;

			for (int n = 0; n < input.N; n++)
			{
				T* cur_src = input.pData + n*in_CHW;
				T* cur_dst = output.pData + n*out_CHW;
				for (int out_c = 0; out_c < out_C; out_c++)
				{
					const T* wData = filters.filters.pData + out_c*input.C* filters.filters.H*filters.filters.W;

					for (int out_h = 0; out_h < out_H; out_h++)
					{
						for (int out_w = 0; out_w < out_W; out_w++)
						{
							const T* in_data = cur_src + (out_h*filters.stride_H)*input.W + out_w*filters.stride_W;
							T* out_data = cur_dst + out_c*out_HW + out_h*out_W + out_w;
							T out_sum = 0;

							int w_idx = 0;
							for (int in_c = 0; in_c < input.C; in_c++)
							{
								for (int in_h = 0; in_h < filters.filters.H; in_h++)
								{
									for (int in_w = 0; in_w < filters.filters.W; in_w++)
									{
									//	printf("(%d %d %d)%f %f\n", in_c, in_h, in_w, wData[w_idx], in_data[in_c*in_HW + in_h*input.W + in_w]);
										out_sum += wData[w_idx++] * in_data[in_c*in_HW + in_h*input.W + in_w];
									}
								}
							}
							out_data[0] = out_sum;
							//printf("[%d,%d,%d]%f\n", out_h, out_w, out_c, out_sum);
						}
					}
				}
			}
			
			return true;
		}

		template<class T>
		static bool _full_connect(const ZQ_CPU_Tensor4D_NCHW<T> &input, const ZQ_CPU_ConvolutionFilters_NCHW<T> &filters, ZQ_CPU_Tensor4D_NCHW<T> &output)
		{
			if (filters.filters.C != input.C || filters.filters.H != input.H || filters.filters.W != input.W)
				return false;

			return _conv_no_padding(input, filters, output);
		}

		template<class T>
		static void _max_pooling(const ZQ_CPU_Tensor4D_NCHW<T> &input, ZQ_CPU_Tensor4D_NCHW<T> &output, int kernel_H, int kernel_W, int stride_H, int stride_W)
		{
			int N = input.N;
			int C = input.C;
			int H = input.H;
			int W = input.W;
			int out_W = ceil((float)(W - kernel_W) / stride_W + 1);
			int out_H = ceil((float)(H - kernel_H) / stride_H + 1);
			output.ChangeSize(N, C, out_H, out_W);

			int HW = H*W;
			int CHW = C*HW;
			int out_HW = out_H*out_W;
			int out_CHW = C*out_HW;

			if ((W - kernel_W) % stride_W == 0 && (H - kernel_H) % stride_H == 0)
			{
				for (int n = 0; n < N; n++)
				{
					const T* cur_src = input.pData + n*CHW;
					T* cur_dst = output.pData + n*out_CHW;
					for (int c = 0; c < C; c++)
					{
						for (int out_h = 0; out_h < out_H; out_h++)
						{
							for (int out_w = 0; out_w < out_W; out_w++)
							{
								const T* pIn = cur_src + c*HW + out_h*stride_H*W + out_w*stride_W;
								T* pOut = cur_dst + c*out_HW + out_h*out_W + out_w;
								pOut[0] = pIn[0];
								for (int kh = 0; kh < kernel_H; kh++)
								{
									for (int kw = 0; kw < kernel_W; kw++)
									{
										pOut[0] = __max(pOut[0], pIn[kh*W + kw]);
									}
								}
							//	printf("[%d,%d,%d]%f\n", out_h, out_w, c, pOut[0]);
							}
						}
					}
				}
				
			}
			else
			{
				for (int n = 0; n < N; n++)
				{
					const T* cur_src = input.pData + n*CHW;
					T* cur_dst = output.pData + n*out_CHW;
					for (int c = 0; c < C; c++)
					{
						for (int out_h = 0; out_h < out_H; out_h++)
						{
							for (int out_w = 0; out_w < out_W; out_w++)
							{
								const T* pIn = cur_src + c*HW + out_h*stride_H*W + out_w*stride_W;
								T* pOut = cur_dst + c*out_HW + out_h*out_W + out_w;
								pOut[0] = pIn[0];
								int max_kh = __min(kernel_H, H - out_h*stride_H);
								int max_kw = __min(kernel_W, W - out_w*stride_W);
								for (int kh = 0; kh < max_kh; kh++)
								{
									for (int kw = 0; kw < max_kw; kw++)
									{
										pOut[0] = __max(pOut[0], pIn[kh*W + kw]);
										
									}
								}
							//	printf("[%d,%d,%d]%f\n", out_h, out_w, c, pOut[0]);
							}
						}
					}
				}
			}
		}

		template<class T>
		static bool _prelu(const ZQ_CPU_Tensor4D_NCHW<T> &input, const ZQ_CPU_Tensor4D_NCHW<T>& bias, const ZQ_CPU_Tensor4D_NCHW<T>& para)
		{
			if (bias.N != 1 || bias.C != input.C || bias.H != 1 || bias.W != 1
				|| para.N != 1 || para.C != input.C || para.H != 1 || para.W != 1)
				return false;

			int HW = input.H*input.W;
			int CHW = input.C*HW;
			for (int c = 0; c < input.C; c++)
			{
				
				for (int n = 0; n < input.N; n++)
				{
					T* cur_data = input.pData + n*CHW + c*HW;
					for(int h = 0;h < input.H;h++)
					{
						for (int w = 0; w < input.W; w++)
						{
							int i = h*input.W + w;
							cur_data[i] += bias.pData[c];
							if (cur_data[i] < 0)
								cur_data[i] *= para.pData[c];
							//printf("[%d %d %d %d] %f\n", c, n, h, w, cur_data[i]);
						}
					
					}
				}
			}
		
			return true;
		}

		template<class T>
		static void _softmax_channel(ZQ_CPU_Tensor4D_NCHW<T> &input)
		{
			// value = exp( value - global max value )
			// sum all value
			// value = value / sum
			int HW = input.H*input.W;
			int CHW = input.C*HW;
			for (int n = 0; n < input.N; n++)
			{
				for (int h = 0; h < input.H; h++)
				{
					for (int w = 0; w < input.W; w++)
					{
						T* cur_data = input.pData + n*CHW + h*input.W + w;
						T sum = 0;
						T max_val = -FLT_MAX;
						for (int c = 0; c < input.C; c++)
						{
							max_val = __max(max_val, cur_data[c*HW]);
						}
						for (int c = 0; c < input.C; c++)
						{
							T tmp_val = exp(cur_data[c*HW] - max_val);
							sum += tmp_val;
							cur_data[c*HW] = tmp_val;
						}
						if (sum != 0)
						{
							for (int c = 0; c < input.C; c++)
							{
								cur_data[c*HW] /= sum;
							}
						}
					}
				}
			}
		}

		static bool _cmp_score(const ZQ_CNN_OrderScore& lsh, const ZQ_CNN_OrderScore& rsh)
		{
			return lsh.score < rsh.score;
		}

		static void _nms(std::vector<ZQ_CNN_BBox> &boundingBox, std::vector<ZQ_CNN_OrderScore> &bboxScore, const float overlap_threshold, const std::string& modelname = "Union", int overlap_num_thresh = 0)
		{
			if (boundingBox.empty())
			{
				return;
			}
			std::vector<int> heros;
			std::vector<int> overlap_num;
			//sort the score
			sort(bboxScore.begin(), bboxScore.end(), _cmp_score);

			int order = 0;
			float IOU = 0;
			float maxX = 0;
			float maxY = 0;
			float minX = 0;
			float minY = 0;
			while (bboxScore.size() > 0)
			{
				order = bboxScore.back().oriOrder;
				bboxScore.pop_back();
				if (order < 0)continue;
				int cur_overlap = 0;
				heros.push_back(order);
				boundingBox.at(order).exist = false;//delete it

				for (int num = 0; num < boundingBox.size(); num++)
				{
					if (boundingBox.at(num).exist)
					{
						//the iou
						maxX = (boundingBox.at(num).row1 > boundingBox.at(order).row1) ? boundingBox.at(num).row1 : boundingBox.at(order).row1;
						maxY = (boundingBox.at(num).col1 > boundingBox.at(order).col1) ? boundingBox.at(num).col1 : boundingBox.at(order).col1;
						minX = (boundingBox.at(num).row2 < boundingBox.at(order).row2) ? boundingBox.at(num).row2 : boundingBox.at(order).row2;
						minY = (boundingBox.at(num).col2 < boundingBox.at(order).col2) ? boundingBox.at(num).col2 : boundingBox.at(order).col2;
						//maxX1 and maxY1 reuse 
						maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
						maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
						//IOU reuse for the area of two bbox
						IOU = maxX * maxY;
						if (!modelname.compare("Union"))
							IOU = IOU / (boundingBox.at(num).area + boundingBox.at(order).area - IOU);
						else if (!modelname.compare("Min"))
						{
							IOU = IOU / ((boundingBox.at(num).area < boundingBox.at(order).area) ? boundingBox.at(num).area : boundingBox.at(order).area);
						}
						if (IOU > overlap_threshold)
						{
							cur_overlap++;
							boundingBox.at(num).exist = false;
							for (std::vector<ZQ_CNN_OrderScore>::iterator it = bboxScore.begin(); it != bboxScore.end(); it++)
							{
								if ((*it).oriOrder == num)
								{
									(*it).oriOrder = -1;
									break;
								}
							}
						}
					}
				}
				overlap_num.push_back(cur_overlap);
			}
			for (int i = 0; i < heros.size(); i++)
			{
				if(overlap_num[i] >= overlap_num_thresh)
					boundingBox.at(heros.at(i)).exist = true;
			}
		}

		static void _refine_and_square_bbox(std::vector<ZQ_CNN_BBox> &vecBbox, const int width, const int height)
		{
			float bbw = 0, bbh = 0, maxSide = 0;
			float h = 0, w = 0;
			float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
			for (std::vector<ZQ_CNN_BBox>::iterator it = vecBbox.begin(); it != vecBbox.end(); it++)
			{
				if ((*it).exist)
				{
					bbh = (*it).row2 - (*it).row1 + 1;
					bbw = (*it).col2 - (*it).col1 + 1;
					x1 = (*it).row1 + (*it).regreCoord[1] * bbh;
					y1 = (*it).col1 + (*it).regreCoord[0] * bbw;
					x2 = (*it).row2 + (*it).regreCoord[3] * bbh;
					y2 = (*it).col2 + (*it).regreCoord[2] * bbw;

					h = x2 - x1 + 1;
					w = y2 - y1 + 1;

					maxSide = (h>w) ? h : w;
					x1 = x1 + h*0.5 - maxSide*0.5;
					y1 = y1 + w*0.5 - maxSide*0.5;
					(*it).row2 = round(x1 + maxSide - 1);
					(*it).col2 = round(y1 + maxSide - 1);
					(*it).row1 = round(x1);
					(*it).col1 = round(y1);

					//boundary check
					if ((*it).row1<0)(*it).row1 = 0;
					if ((*it).col1<0)(*it).col1 = 0;
					if ((*it).row2>height)(*it).row2 = height - 1;
					if ((*it).col2>width)(*it).col2 = width - 1;

					it->area = (it->row2 - it->row1)*(it->col2 - it->col1);
				}
			}
		}

		template<class T>
		static bool _read_data(const std::string& filename, int num, const long dataNumber[], T * const pTeam[])
		{
			std::ifstream in(filename.data());
			std::string line;
			if (in)
			{
				for (int nn = 0; nn < num; nn++)
				{
					for (int i = 0; i < dataNumber[nn]; i++)
					{
						getline(in, line);
						try
						{
							line.erase(0, 1);
							int pos = line.find(']');
							line.erase(pos, 1);
							pTeam[nn][i] = atof(line.data());
						}
						catch (std::exception& e)
						{
							return false;
						}
					}
				}
				return true;
			}
			else
			{
				return false;
			}
		}

		template<class T>
		static bool _read_data_binary(const std::string& filename, int num, long dataNumber[], T *pTeam[])
		{
			FILE* in;
			if(0 != fopen_s(&in, filename.c_str(), "rb"))
			{
				return false;
			}
			for (int i = 0; i < num; i++)
			{
				int len = fread(pTeam[i], sizeof(T), dataNumber[i], in);
				if (len != dataNumber[i])
				{
					fclose(in);
					return false;
				}
			}
			fclose(in);
			return true;
		}

		template<class T>
		static bool _save_data_binary(const std::string& filename, int num, long dataNumber[], T *pTeam[])
		{
			FILE* out;
			if (0 != fopen_s(&out, filename.c_str(), "wb"))
			{
				return false;
			}
			for (int i = 0; i < num; i++)
			{
				int len = fwrite(pTeam[i], sizeof(T), dataNumber[i], out);
				if (len != dataNumber[i])
				{
					fclose(out);
					return false;
				}
			}
			fclose(out);
			return true;
		}
	};
	
}

#endif
