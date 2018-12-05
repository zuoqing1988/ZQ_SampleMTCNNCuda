#ifndef _ZQ_CNN_UTILS_OPENCV_H_
#define _ZQ_CNN_UTILS_OPENCV_H_
#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ZQ_CPU_Tensor4D_NCHW.h"

namespace ZQ
{
	class ZQ_CNN_Utils_OPENCV
	{
	public:
		template<class T>
		static bool Image2Matrix(const cv::Mat &input, ZQ_CPU_Tensor4D_NCHW<T> &output)
		{
			if (input.data == NULL)
			{
				return false;
			}

			if (input.type() == CV_8UC1)
			{
				int H = input.rows;
				int W = input.cols;
				int HW = H*W;
				output.ChangeSize(1, 3, H, W);

				T *p = output.pData;
				for (int rowI = 0; rowI < H; rowI++)
				{
					const unsigned char* row_data = input.ptr<unsigned char>(rowI);
					for (int colK = 0; colK < W; colK++)
					{
						*p = (row_data[colK] - 127.5f)*0.0078125f;
						*(p + HW) = *p;
						*(p + 2 * HW) = *p;
						p++;
					}
				}
			}
			else if (input.type() == CV_8UC3)
			{
				int H = input.rows;
				int W = input.cols;
				int HW = H*W;
				output.ChangeSize(1, 3, H, W);

				T *p = output.pData;
				for (int rowI = 0; rowI < H; rowI++)
				{
					const unsigned char* row_data = input.ptr<unsigned char>(rowI);
					for (int colK = 0; colK < W; colK++)
					{
						*p = (row_data[colK * 3 + 2] - 127.5f)*0.0078125f;
						*(p + HW) = (row_data[colK * 3 + 1] - 127.5f)*0.0078125f;
						*(p + 2 * HW) = (row_data[colK * 3 + 0] - 127.5f)*0.0078125f;
						p++;
					}
				}
			}
			else
				return false;


			return true;
		}


		template<class T>
		static bool Image2Matrix_rects(const cv::Mat &input, const std::vector<int>& rects, int dst_H, int dst_W, ZQ_CPU_Tensor4D_NCHW<T> &output)
		{
			if (input.data == NULL || rects.size()%4 != 0 || rects.size() == 0)
			{
				return false;
			}

			if (input.type() != CV_8UC1 && input.type() != CV_8UC3)
				return false;

			int W = input.cols;
			int H = input.rows;
			int num_rects = rects.size() / 4;
			for (int n = 0; n < num_rects; n++)
			{
				if (rects[n * 4 + 0] < 0 || rects[n * 4 + 1] < 0 || rects[n * 4 + 0] + rects[n * 4 + 2] > W || rects[n * 4 + 1] + rects[n * 4 + 3] > H)
					return false;
			}

			int C = 3;
			output.ChangeSize(num_rects, C, dst_H, dst_W);

			if (input.type() == CV_8UC1)
			{
				for (int n = 0; n < num_rects; n++)
				{
					cv::Mat roi = input(cv::Rect(rects[n * 4 + 0], rects[n * 4 + 1], rects[n * 4 + 2], rects[n * 4 + 3]));
					cv::Mat dst_roi;
					cv::resize(roi, dst_roi, cv::Size(dst_W, dst_H));
					T *p = output.pData+n*C*dst_H*dst_W;
					for (int rowI = 0; rowI < dst_H; rowI++)
					{
						const unsigned char* row_data = dst_roi.ptr<unsigned char>(rowI);
						for (int colK = 0; colK < dst_W; colK++)
						{
							*p = (row_data[colK] - 127.5)*0.0078125;
							*(p + dst_H*dst_W) = *p;
							*(p + 2 * dst_H*dst_W) = *p;
							p++;
						}
					}
				}
				
			}
			else if (input.type() == CV_8UC3)
			{
				for (int n = 0; n < num_rects; n++)
				{
					cv::Mat roi = input(cv::Rect(rects[n * 4 + 0], rects[n * 4 + 1], rects[n * 4 + 2], rects[n * 4 + 3]));
					cv::Mat dst_roi;
					cv::resize(roi, dst_roi, cv::Size(dst_W, dst_H));
					T *p = output.pData + n*C*dst_H*dst_W;
					for (int rowI = 0; rowI < dst_H; rowI++)
					{
						const unsigned char* row_data = dst_roi.ptr<unsigned char>(rowI);
						for (int colK = 0; colK < dst_W; colK++)
						{
							*p = (row_data[colK * 3 + 2] - 127.5)*0.0078125;
							*(p + dst_H*dst_W) = (row_data[colK * 3 + 1] - 127.5)*0.0078125;
							*(p + 2 * dst_H*dst_W) = (row_data[colK * 3 + 0] - 127.5)*0.0078125;
							p++;
						}
					}
				}
			}
			
			return true;
		}
	};
}
#endif
