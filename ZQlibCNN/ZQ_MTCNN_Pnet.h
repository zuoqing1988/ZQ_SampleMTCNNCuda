#ifndef _ZQ_MTCNN_PNET_H_
#define _ZQ_MTCNN_PNET_H_
#pragma once

#include "ZQ_CNN_Forward.h"
#ifdef ZQ_CNN_USE_CUDA
#include "ZQ_CNN_Forward_CUDA.h"
#endif
#include "ZQ_CNN_Utils_OPENCV.h"
#include <time.h>
#include <omp.h>

namespace ZQ
{
	class ZQ_MTCNN_Pnet
	{
		using BaseType = float;
	public:
		ZQ_MTCNN_Pnet() {}
		~ZQ_MTCNN_Pnet() {}

		bool Initialize(const std::string& filename)
		{
			conv1_filters.ChangeSize(0, 0, 1, 1, 10, 3, 3, 3);
			bias1.ChangeSize(1, 10, 1, 1);
			para1.ChangeSize(1, 10, 1, 1);
			conv2_filters.ChangeSize(0, 0, 1, 1, 16, 10, 3, 3);
			bias2.ChangeSize(1, 16, 1, 1);
			para2.ChangeSize(1, 16, 1, 1);
			conv3_filters.ChangeSize(0, 0, 1, 1, 32, 16, 3, 3);
			bias3.ChangeSize(1, 32, 1, 1);
			para3.ChangeSize(1, 32, 1, 1);
			conv4c1_filters.ChangeSize(0, 0, 1, 1, 2, 32, 1, 1);
			bias4c1.ChangeSize(1, 2, 1, 1);
			conv4c2_filters.ChangeSize(0, 0, 1, 1, 4, 32, 1, 1);
			bias4c2.ChangeSize(1, 4, 1, 1);
			long dataNumber[13] =
			{
				conv1_filters.filters.GetDataLength(), bias1.GetDataLength(), para1.GetDataLength(),
				conv2_filters.filters.GetDataLength(), bias2.GetDataLength(), para2.GetDataLength(),
				conv3_filters.filters.GetDataLength(), bias3.GetDataLength(), para3.GetDataLength(),
				conv4c1_filters.filters.GetDataLength(), bias4c1.GetDataLength(),
				conv4c2_filters.filters.GetDataLength(), bias4c2.GetDataLength()
			};
			BaseType *pointTeam[13] =
			{
				conv1_filters.filters.pData, bias1.pData, para1.pData,
				conv2_filters.filters.pData, bias2.pData, para2.pData,
				conv3_filters.filters.pData, bias3.pData, para3.pData,
				conv4c1_filters.filters.pData, bias4c1.pData,
				conv4c2_filters.filters.pData, bias4c2.pData
			};

			/*if (!ZQ_CNN_Forward::ReadData<BaseType>(filename, 13, dataNumber, pointTeam))
				return false;*/
			if (!ZQ_CNN_Forward::ReadDataBinary<BaseType>(filename, 13, dataNumber, pointTeam))
					return false;

#ifdef ZQ_CNN_USE_CUDA
			ZQ_CNN_Forward_CUDA::Upload(conv1_filters, gpu_conv1_filters);
			ZQ_CNN_Forward_CUDA::Upload(bias1, gpu_bias1);
			ZQ_CNN_Forward_CUDA::Upload(para1, gpu_para1);
			ZQ_CNN_Forward_CUDA::Upload(conv2_filters, gpu_conv2_filters);
			ZQ_CNN_Forward_CUDA::Upload(bias2, gpu_bias2);
			ZQ_CNN_Forward_CUDA::Upload(para2, gpu_para2);
			ZQ_CNN_Forward_CUDA::Upload(conv3_filters, gpu_conv3_filters);
			ZQ_CNN_Forward_CUDA::Upload(bias3, gpu_bias3);
			ZQ_CNN_Forward_CUDA::Upload(para3, gpu_para3);
			ZQ_CNN_Forward_CUDA::Upload(conv4c1_filters, gpu_conv4c1_filters);
			ZQ_CNN_Forward_CUDA::Upload(bias4c1, gpu_bias4c1);
			ZQ_CNN_Forward_CUDA::Upload(conv4c2_filters, gpu_conv4c2_filters);
			ZQ_CNN_Forward_CUDA::Upload(bias4c2, gpu_bias4c2);
#endif
			return true;
		}

		void Run(cv::Mat &image, float scale, float Pthreshold = 0.6, bool display = false)
		{
			boundingBox.clear();
			bboxScore.clear();
			this->Pthreshold = Pthreshold;
			double t1 = omp_get_wtime();
			ZQ_CNN_Utils_OPENCV::Image2Matrix(image, rgb);
			ZQ_CNN_Forward::Convolution(rgb, conv1_filters, conv1);
			ZQ_CNN_Forward::PReLU(conv1, bias1, para1);
			//Pooling layer
			ZQ_CNN_Forward::MaxPooling(conv1, maxPooling1, 2, 2, 2, 2);
			double t2 = omp_get_wtime();

			ZQ_CNN_Forward::Convolution(maxPooling1, conv2_filters, conv2);
			ZQ_CNN_Forward::PReLU(conv2, bias2, para2);

			double t3 = omp_get_wtime();

			//conv3 
			ZQ_CNN_Forward::Convolution(conv2, conv3_filters, conv3);
			ZQ_CNN_Forward::PReLU(conv3, bias3, para3);

			double t4 = omp_get_wtime();

			//conv4c1   score
			ZQ_CNN_Forward::Convolution(conv3, conv4c1_filters, score);
			score.AddBias(bias4c1);
			ZQ_CNN_Forward::SoftmaxChannel(score);

			double t5 = omp_get_wtime();

			//conv4c2   location
			ZQ_CNN_Forward::Convolution(conv3, conv4c2_filters, location);
			location.AddBias(bias4c2);

			double t6 = omp_get_wtime();
			//softmax layer
			_generateBbox(score, location, scale);

			double t7 = omp_get_wtime();

			if(display)
				printf("%.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", (t2 - t1), (t3 - t2), (t4 - t3), (t5 - t4), (t6 - t5), (t7 - t6));
		}

#ifdef ZQ_CNN_USE_CUDA
		void Run_CUDA(const ZQ_GPU_Tensor4D_NCHW& gpu_input, float scale, float Pthreshold = 0.6, bool display = false, bool useCUDNN = true)
		{
			double t1 = omp_get_wtime();
			boundingBox.clear();
			bboxScore.clear();
			this->Pthreshold = Pthreshold;
			
			float conv1_time = 0, addbias_prelu1_time = 0, pool1_time = 0;
			if(useCUDNN)
				ZQ_CNN_Forward_CUDA::ConvolutionCUDNN(gpu_input, gpu_conv1_filters, gpu_conv1, conv1_time);
			else
				ZQ_CNN_Forward_CUDA::Convolution(gpu_input, gpu_conv1_filters, gpu_conv1, conv1_time);
			ZQ_CNN_Forward_CUDA::PReLU(gpu_conv1, gpu_bias1, gpu_para1, addbias_prelu1_time);
			ZQ_CNN_Forward_CUDA::MaxPooling(gpu_conv1, gpu_maxPooling1, 2, 2, 2, 2, pool1_time);
			
			float conv2_time = 0, addbias_prelu2_time = 0;
			if(useCUDNN)
				ZQ_CNN_Forward_CUDA::ConvolutionCUDNN(gpu_maxPooling1, gpu_conv2_filters, gpu_conv2, conv2_time);
			else
				ZQ_CNN_Forward_CUDA::Convolution(gpu_maxPooling1, gpu_conv2_filters, gpu_conv2, conv2_time);
			ZQ_CNN_Forward_CUDA::PReLU(gpu_conv2, gpu_bias2, gpu_para2, addbias_prelu2_time);

			float conv3_time = 0, addbias_prelu3_time = 0;
			if(useCUDNN)
				ZQ_CNN_Forward_CUDA::ConvolutionCUDNN(gpu_conv2, gpu_conv3_filters, gpu_conv3, conv3_time);
			else
				ZQ_CNN_Forward_CUDA::Convolution(gpu_conv2, gpu_conv3_filters, gpu_conv3, conv3_time);
			ZQ_CNN_Forward_CUDA::PReLU(gpu_conv3, gpu_bias3, gpu_para3, addbias_prelu3_time);

			float conv4_1_time = 0, addbias4_1_time = 0, softmax4_1_time = 0;
			//conv4c1   score
			if(useCUDNN)
				ZQ_CNN_Forward_CUDA::ConvolutionCUDNN(gpu_conv3, gpu_conv4c1_filters, gpu_score, conv4_1_time);
			else
				ZQ_CNN_Forward_CUDA::Convolution(gpu_conv3, gpu_conv4c1_filters, gpu_score, conv4_1_time);
			ZQ_CNN_Forward_CUDA::AddBias(gpu_score, gpu_bias4c1, addbias4_1_time);
			ZQ_CNN_Forward_CUDA::SoftmaxChannel(gpu_score, softmax4_1_time);

			float conv4_2_time = 0, addbias4_2_time = 0;
			//conv4c2   location
			if(useCUDNN)
				ZQ_CNN_Forward_CUDA::ConvolutionCUDNN(gpu_conv3, gpu_conv4c2_filters, gpu_location, conv4_2_time);
			else
				ZQ_CNN_Forward_CUDA::Convolution(gpu_conv3, gpu_conv4c2_filters, gpu_location, conv4_2_time);
			ZQ_CNN_Forward_CUDA::AddBias(gpu_location, gpu_bias4c2,addbias4_2_time);
			
			double t2 = omp_get_wtime();

			ZQ_CNN_Forward_CUDA::Download(gpu_score, score);
			ZQ_CNN_Forward_CUDA::Download(gpu_location, location);
			_generateBbox(score, location, scale);

			double t3 = omp_get_wtime();

			/*printf("conv1          :%.3f ms\n", conv1_time);
			printf("addbias1+prelu1:%.3f ms\n", addbias_prelu1_time);
			printf("pool1          :%.3f ms\n", pool1_time);
			printf("conv2          :%.3f ms\n", conv2_time);
			printf("addbias2+prelu2:%.3f ms\n", addbias_prelu2_time);
			printf("conv3          :%.3f ms\n", conv3_time);
			printf("addbias3+prelu3:%.3f ms\n", addbias_prelu3_time);
			printf("conv4-1        :%.3f ms\n", conv4_1_time);
			printf("addbias4-1     :%.3f ms\n", addbias4_1_time);
			printf("softmax        :%.3f ms\n", softmax4_1_time);
			printf("conv4-2        :%.3f ms\n", conv4_2_time);
			printf("addbias4-2     :%.3f ms\n", addbias4_2_time);
			printf("download       :%.3f ms\n", 1000 * (t3 - t2));
			printf("total          :%.3f ms\n", 1000 * (t3 - t1));*/
		}
#endif

		void CopyData(const ZQ_MTCNN_Pnet& other)
		{
			Pthreshold = other.Pthreshold;
			boundingBox = other.boundingBox;
			bboxScore = other.bboxScore;
			rgb.CopyData(other.rgb);
			conv1.CopyData(other.conv1);
			maxPooling1.CopyData(other.maxPooling1);
			conv2.CopyData(other.conv2);
			score.CopyData(other.score);
			location.CopyData(other.location);
			conv1_filters.CopyData(other.conv1_filters);
			bias1.CopyData(other.bias1);
			para1.CopyData(other.para1);
			conv2_filters.CopyData(other.conv2_filters);
			bias2.CopyData(other.bias2);
			para2.CopyData(other.para2);
			conv3_filters.CopyData(other.conv3_filters);
			bias3.CopyData(other.bias3);
			para3.CopyData(other.para3);
			conv4c1_filters.CopyData(other.conv4c1_filters);
			bias4c1.CopyData(other.bias4c1);
			conv4c2_filters.CopyData(other.conv4c2_filters);
			bias4c2.CopyData(other.bias4c2);
#ifdef ZQ_CNN_USE_CUDA
			gpu_conv1.CopyData(other.gpu_conv1);
			gpu_maxPooling1.CopyData(other.gpu_maxPooling1);
			gpu_conv2.CopyData(other.gpu_conv2);
			gpu_score.CopyData(other.gpu_score);
			gpu_location.CopyData(other.gpu_location);
			gpu_conv1_filters.CopyData(other.gpu_conv1_filters);
			gpu_bias1.CopyData(other.gpu_bias1);
			gpu_para1.CopyData(other.gpu_para1);
			gpu_conv2_filters.CopyData(other.gpu_conv2_filters);
			gpu_bias2.CopyData(other.gpu_bias2);
			gpu_para2.CopyData(other.gpu_para2);
			gpu_conv3_filters.CopyData(other.gpu_conv3_filters);
			gpu_bias3.CopyData(other.gpu_bias3);
			gpu_para3.CopyData(other.gpu_para3);
			gpu_conv4c1_filters.CopyData(other.gpu_conv4c1_filters);
			gpu_bias4c1.CopyData(other.gpu_bias4c1);
			gpu_conv4c2_filters.CopyData(other.gpu_conv4c2_filters);
			gpu_bias4c2.CopyData(other.gpu_bias4c2);
#endif
		}

	public:
		BaseType Pthreshold;
		std::vector<ZQ_CNN_BBox> boundingBox;
		std::vector<ZQ_CNN_OrderScore> bboxScore;
	private:
		//the image for mxnet conv
		ZQ_CPU_Tensor4D_NCHW<BaseType> rgb;
		ZQ_CPU_Tensor4D_NCHW<BaseType> conv1;
		ZQ_CPU_Tensor4D_NCHW<BaseType> maxPooling1;
		ZQ_CPU_Tensor4D_NCHW<BaseType> conv2;
		ZQ_CPU_Tensor4D_NCHW<BaseType> conv3;
		ZQ_CPU_Tensor4D_NCHW<BaseType> score;
		ZQ_CPU_Tensor4D_NCHW<BaseType> location;

		//Weight
		ZQ_CPU_ConvolutionFilters_NCHW<BaseType> conv1_filters;
		ZQ_CPU_Tensor4D_NCHW<BaseType> bias1;
		ZQ_CPU_Tensor4D_NCHW<BaseType> para1;
		ZQ_CPU_ConvolutionFilters_NCHW<BaseType> conv2_filters;
		ZQ_CPU_Tensor4D_NCHW<BaseType> bias2;
		ZQ_CPU_Tensor4D_NCHW<BaseType> para2;
		ZQ_CPU_ConvolutionFilters_NCHW<BaseType> conv3_filters;
		ZQ_CPU_Tensor4D_NCHW<BaseType> bias3;
		ZQ_CPU_Tensor4D_NCHW<BaseType> para3;
		ZQ_CPU_ConvolutionFilters_NCHW<BaseType> conv4c1_filters;
		ZQ_CPU_Tensor4D_NCHW<BaseType> bias4c1;
		ZQ_CPU_ConvolutionFilters_NCHW<BaseType> conv4c2_filters;
		ZQ_CPU_Tensor4D_NCHW<BaseType> bias4c2;

#ifdef ZQ_CNN_USE_CUDA
	protected:
		ZQ_GPU_Tensor4D_NCHW gpu_rgb;
		ZQ_GPU_Tensor4D_NCHW gpu_conv1;
		ZQ_GPU_Tensor4D_NCHW gpu_maxPooling1;
		ZQ_GPU_Tensor4D_NCHW gpu_conv2;
		ZQ_GPU_Tensor4D_NCHW gpu_conv3;
		ZQ_GPU_Tensor4D_NCHW gpu_score;
		ZQ_GPU_Tensor4D_NCHW gpu_location;

		//Weight
		ZQ_GPU_ConvolutionFilters_NCHW gpu_conv1_filters;
		ZQ_GPU_Tensor4D_NCHW gpu_bias1;
		ZQ_GPU_Tensor4D_NCHW gpu_para1;
		ZQ_GPU_ConvolutionFilters_NCHW gpu_conv2_filters;
		ZQ_GPU_Tensor4D_NCHW gpu_bias2;
		ZQ_GPU_Tensor4D_NCHW gpu_para2;
		ZQ_GPU_ConvolutionFilters_NCHW gpu_conv3_filters;
		ZQ_GPU_Tensor4D_NCHW gpu_bias3;
		ZQ_GPU_Tensor4D_NCHW gpu_para3;
		ZQ_GPU_ConvolutionFilters_NCHW gpu_conv4c1_filters;
		ZQ_GPU_Tensor4D_NCHW gpu_bias4c1;
		ZQ_GPU_ConvolutionFilters_NCHW gpu_conv4c2_filters;
		ZQ_GPU_Tensor4D_NCHW gpu_bias4c2;
#endif
		

	private:
		void _generateBbox(ZQ_CPU_Tensor4D_NCHW<BaseType> &score, ZQ_CPU_Tensor4D_NCHW<BaseType> &location, BaseType scale)
		{
			//for pooling 
			int stride = 2;
			int cellsize = 12;
			int count = 0;
			//score p
			BaseType *p = score.pData + score.W*score.H;
			BaseType *plocal = location.pData;
			ZQ_CNN_BBox bbox;
			ZQ_CNN_OrderScore order;
			for (int row = 0; row < score.H; row++)
			{
				for (int col = 0; col < score.W; col++)
				{
					if (*p > Pthreshold)
					{
						bbox.score = *p;
						order.score = *p;
						order.oriOrder = count;
						bbox.row1 = round((stride*row + 1) / scale);
						bbox.col1 = round((stride*col + 1) / scale);
						bbox.row2 = round((stride*row + 1 + cellsize) / scale);
						bbox.col2 = round((stride*col + 1 + cellsize) / scale);
						bbox.exist = true;
						bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
						for (int channel = 0; channel < 4; channel++)
							bbox.regreCoord[channel] = *(plocal + channel*location.W*location.H);
						boundingBox.push_back(bbox);
						bboxScore.push_back(order);
						count++;
					}
					p++;
					plocal++;
				}
			}
		}
	};
}

#endif
