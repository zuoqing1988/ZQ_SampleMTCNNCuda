#ifndef _ZQ_MTCNN_RNET_H_
#define _ZQ_MTCNN_RNET_H_
#pragma once

#include "ZQ_CNN_Forward.h"
#ifdef ZQ_CNN_USE_CUDA
#include "ZQ_CNN_Forward_CUDA.h"
#endif
#include "ZQ_CNN_Utils_OPENCV.h"


namespace ZQ
{
	class ZQ_MTCNN_Rnet
	{
		using BaseType = float;
		friend class ZQ_MTCNN;
	public:
		ZQ_MTCNN_Rnet() {}
		~ZQ_MTCNN_Rnet() {}

		bool Initialize(const std::string& filename)
		{
			conv1_filters.ChangeSize(0, 0, 1, 1, 28, 3, 3, 3);
			bias1.ChangeSize(1, 28, 1, 1);
			para1.ChangeSize(1, 28, 1, 1);
			conv2_filters.ChangeSize(0, 0, 1, 1, 48, 28, 3, 3);
			bias2.ChangeSize(1, 48, 1, 1);
			para2.ChangeSize(1, 48, 1, 1);
			conv3_filters.ChangeSize(0, 0, 1, 1, 64, 48, 2, 2);
			bias3.ChangeSize(1, 64, 1, 1);
			para3.ChangeSize(1, 64, 1, 1);
			fc4_filters.ChangeSize(0, 0, 1, 1, 128, 64, 3, 3);
			bias4.ChangeSize(1, 128, 1, 1);
			para4.ChangeSize(1, 128, 1, 1);
			score_filters.ChangeSize(0, 0, 1, 1, 2, 128, 1, 1);
			score_bias.ChangeSize(1, 2, 1, 1);
			location_filters.ChangeSize(0, 0, 1, 1, 4, 128, 1, 1);
			location_bias.ChangeSize(1, 4, 1, 1);
			long dataNumber[16] =
			{
				conv1_filters.filters.GetDataLength(), bias1.GetDataLength(), para1.GetDataLength(),
				conv2_filters.filters.GetDataLength(), bias2.GetDataLength(), para2.GetDataLength(),
				conv3_filters.filters.GetDataLength(), bias3.GetDataLength(), para3.GetDataLength(),
				fc4_filters.filters.GetDataLength(), bias4.GetDataLength(), para4.GetDataLength(),
				score_filters.filters.GetDataLength(), score_bias.GetDataLength(), 
				location_filters.filters.GetDataLength(), location_bias.GetDataLength()
			};
			BaseType *pointTeam[16] =
			{
				conv1_filters.filters.pData, bias1.pData, para1.pData,
				conv2_filters.filters.pData, bias2.pData, para2.pData,
				conv3_filters.filters.pData, bias3.pData, para3.pData,
				fc4_filters.filters.pData, bias4.pData, para4.pData,
				score_filters.filters.pData, score_bias.pData, 
				location_filters.filters.pData, location_bias.pData
			};

			/*if (!ZQ_CNN_Forward::ReadData(filename, 16, dataNumber, pointTeam))
				return false;*/
			if (!ZQ_CNN_Forward::ReadDataBinary(filename, 16, dataNumber, pointTeam))
				return false;

			//Init the network
			rgb.ChangeSize(1, 3, 24, 24);

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
			ZQ_CNN_Forward_CUDA::Upload(fc4_filters, gpu_fc4_filters);
			ZQ_CNN_Forward_CUDA::Upload(bias4, gpu_bias4);
			ZQ_CNN_Forward_CUDA::Upload(para4, gpu_para4);
			ZQ_CNN_Forward_CUDA::Upload(score_filters, gpu_score_filters);
			ZQ_CNN_Forward_CUDA::Upload(score_bias, gpu_score_bias);
			ZQ_CNN_Forward_CUDA::Upload(location_filters, gpu_location_filters);
			ZQ_CNN_Forward_CUDA::Upload(location_bias, gpu_location_bias);
#endif

			return true;
		}

		void Run(cv::Mat &image, float Rthreshold = 0.7, bool display = false)
		{
			this->Rthreshold = Rthreshold;
			ZQ_CNN_Utils_OPENCV::Image2Matrix(image, rgb);
			ZQ_CNN_Forward::Convolution(rgb, conv1_filters, conv1_out);
			ZQ_CNN_Forward::PReLU(conv1_out, bias1, para1);
			ZQ_CNN_Forward::MaxPooling(conv1_out, pooling1_out, 3, 3, 2, 2);

			ZQ_CNN_Forward::Convolution(pooling1_out, conv2_filters, conv2_out);
			ZQ_CNN_Forward::PReLU(conv2_out, bias2, para2);
			ZQ_CNN_Forward::MaxPooling(conv2_out, pooling2_out, 3, 3, 2, 2);

			//conv3 
			ZQ_CNN_Forward::Convolution(pooling2_out, conv3_filters, conv3_out);
			ZQ_CNN_Forward::PReLU(conv3_out, bias3, para3);

			//flatten
			ZQ_CNN_Forward::FullConnect(conv3_out, fc4_filters, fc4_out);
			ZQ_CNN_Forward::PReLU(fc4_out, bias4, para4);

			//conv51   score
			ZQ_CNN_Forward::FullConnect(fc4_out, score_filters, score);
			score.AddBias(score_bias);
			ZQ_CNN_Forward::SoftmaxChannel(score);

			//conv5_2   location
			ZQ_CNN_Forward::FullConnect(fc4_out, location_filters, location);
			location.AddBias(location_bias);
		}

#ifdef ZQ_CNN_USE_CUDA
		void Run_Batch_CUDA(const ZQ_GPU_Tensor4D_NCHW& gpu_input, float Rthreshold = 0.7, bool display = false)
		{
			double t1 = omp_get_wtime();
			float conv1_time = 0, addbias_prelu1_time = 0, pool1_time = 0;
			ZQ_CNN_Forward_CUDA::ConvolutionCUDNN(gpu_input, gpu_conv1_filters, gpu_conv1_out, conv1_time);
			ZQ_CNN_Forward_CUDA::PReLU(gpu_conv1_out, gpu_bias1, gpu_para1, addbias_prelu1_time);
			ZQ_CNN_Forward_CUDA::MaxPooling(gpu_conv1_out, gpu_pooling1_out, 3, 3, 2, 2, pool1_time);

			float conv2_time = 0, addbias_prelu2_time = 0, pool2_time = 0;
			ZQ_CNN_Forward_CUDA::ConvolutionCUDNN(gpu_pooling1_out, gpu_conv2_filters, gpu_conv2_out, conv2_time);
			ZQ_CNN_Forward_CUDA::PReLU(gpu_conv2_out, gpu_bias2, gpu_para2, addbias_prelu2_time);
			ZQ_CNN_Forward_CUDA::MaxPooling(gpu_conv2_out, gpu_pooling2_out, 3, 3, 2, 2, pool2_time);

			float conv3_time = 0, addbias_prelu3_time = 0;
			//conv3 
			ZQ_CNN_Forward_CUDA::ConvolutionCUDNN(gpu_pooling2_out, gpu_conv3_filters, gpu_conv3_out,conv3_time);
			ZQ_CNN_Forward_CUDA::PReLU(gpu_conv3_out, gpu_bias3, gpu_para3, addbias_prelu3_time);

			float conv4_time = 0, addbias_prelu4_time = 0;
			//flatten
			ZQ_CNN_Forward_CUDA::ConvolutionCUDNN(gpu_conv3_out, gpu_fc4_filters, gpu_fc4_out, conv4_time);
			ZQ_CNN_Forward_CUDA::PReLU(gpu_fc4_out, gpu_bias4, gpu_para4, addbias_prelu4_time);

			float conv5_1_time = 0, addbias5_1_time = 0, softmax5_1_time = 0;
			//conv51   score
			ZQ_CNN_Forward_CUDA::ConvolutionCUDNN(gpu_fc4_out, gpu_score_filters, gpu_score, conv5_1_time);
			ZQ_CNN_Forward_CUDA::AddBias(gpu_score, gpu_score_bias, addbias5_1_time);
			ZQ_CNN_Forward_CUDA::SoftmaxChannel(gpu_score,softmax5_1_time);

			float conv5_2_time = 0, addbias5_2_time = 0, softmax5_2_time = 0;
			//conv5_2   location
			ZQ_CNN_Forward_CUDA::ConvolutionCUDNN(gpu_fc4_out, gpu_location_filters, gpu_location, conv5_2_time);
			ZQ_CNN_Forward_CUDA::AddBias(gpu_location, gpu_location_bias,addbias5_2_time);

			double t2 = omp_get_wtime();
			ZQ_CNN_Forward_CUDA::Download(gpu_score, score);
			ZQ_CNN_Forward_CUDA::Download(gpu_location, location);
			double t3 = omp_get_wtime();

			/*printf("conv1          :%.3f ms\n", conv1_time);
			printf("addbias1+prelu1:%.3f ms\n", addbias_prelu1_time);
			printf("pool1          :%.3f ms\n", pool1_time);
			printf("conv2          :%.3f ms\n", conv2_time);
			printf("addbias2+prelu2:%.3f ms\n", addbias_prelu2_time);
			printf("pool2          :%.3f ms\n", pool2_time);
			printf("conv3          :%.3f ms\n", conv3_time);
			printf("addbias3+prelu3:%.3f ms\n", addbias_prelu3_time);
			printf("conv4          :%.3f ms\n", conv4_time);
			printf("addbias4+prelu4:%.3f ms\n", addbias_prelu4_time);
			printf("conv5-1        :%.3f ms\n", conv5_1_time);
			printf("addbias5-1     :%.3f ms\n", addbias5_1_time);
			printf("softmax        :%.3f ms\n", softmax5_1_time);
			printf("conv5-2        :%.3f ms\n", conv5_2_time);
			printf("addbias5-2     :%.3f ms\n", addbias5_2_time);
			printf("download       :%.3f ms\n", 1000 * (t3 - t2));
			printf("total          :%.3f ms\n", 1000 * (t3 - t1));*/
		}
#endif

	public:
		float Rthreshold;
		ZQ_CPU_Tensor4D_NCHW<BaseType> score;
		ZQ_CPU_Tensor4D_NCHW<BaseType> location;

	private:
		ZQ_CPU_Tensor4D_NCHW<BaseType> rgb;
		ZQ_CPU_Tensor4D_NCHW<BaseType> conv1_out;
		ZQ_CPU_Tensor4D_NCHW<BaseType> pooling1_out;
		ZQ_CPU_Tensor4D_NCHW<BaseType> conv2_out;
		ZQ_CPU_Tensor4D_NCHW<BaseType> pooling2_out;
		ZQ_CPU_Tensor4D_NCHW<BaseType> conv3_out;
		ZQ_CPU_Tensor4D_NCHW<BaseType> fc4_out;

		ZQ_CPU_ConvolutionFilters_NCHW<BaseType> conv1_filters;
		ZQ_CPU_Tensor4D_NCHW<BaseType> bias1;
		ZQ_CPU_Tensor4D_NCHW<BaseType> para1;
		ZQ_CPU_ConvolutionFilters_NCHW<BaseType> conv2_filters;
		ZQ_CPU_Tensor4D_NCHW<BaseType> bias2;
		ZQ_CPU_Tensor4D_NCHW<BaseType> para2;
		ZQ_CPU_ConvolutionFilters_NCHW<BaseType> conv3_filters;
		ZQ_CPU_Tensor4D_NCHW<BaseType> bias3;
		ZQ_CPU_Tensor4D_NCHW<BaseType> para3;
		ZQ_CPU_ConvolutionFilters_NCHW<BaseType> fc4_filters;
		ZQ_CPU_Tensor4D_NCHW<BaseType> bias4;
		ZQ_CPU_Tensor4D_NCHW<BaseType> para4;
		ZQ_CPU_ConvolutionFilters_NCHW<BaseType> score_filters;
		ZQ_CPU_Tensor4D_NCHW<BaseType> score_bias;
		ZQ_CPU_ConvolutionFilters_NCHW<BaseType> location_filters;
		ZQ_CPU_Tensor4D_NCHW<BaseType> location_bias;

#ifdef ZQ_CNN_USE_CUDA
	protected:
		ZQ_GPU_Tensor4D_NCHW gpu_score;
		ZQ_GPU_Tensor4D_NCHW gpu_location;
		ZQ_GPU_Tensor4D_NCHW gpu_conv1_out;
		ZQ_GPU_Tensor4D_NCHW gpu_pooling1_out;
		ZQ_GPU_Tensor4D_NCHW gpu_conv2_out;
		ZQ_GPU_Tensor4D_NCHW gpu_pooling2_out;
		ZQ_GPU_Tensor4D_NCHW gpu_conv3_out;
		ZQ_GPU_Tensor4D_NCHW gpu_fc4_out;

		ZQ_GPU_ConvolutionFilters_NCHW gpu_conv1_filters;
		ZQ_GPU_Tensor4D_NCHW gpu_bias1;
		ZQ_GPU_Tensor4D_NCHW gpu_para1;
		ZQ_GPU_ConvolutionFilters_NCHW gpu_conv2_filters;
		ZQ_GPU_Tensor4D_NCHW gpu_bias2;
		ZQ_GPU_Tensor4D_NCHW gpu_para2;
		ZQ_GPU_ConvolutionFilters_NCHW gpu_conv3_filters;
		ZQ_GPU_Tensor4D_NCHW gpu_bias3;
		ZQ_GPU_Tensor4D_NCHW gpu_para3;
		ZQ_GPU_ConvolutionFilters_NCHW gpu_fc4_filters;
		ZQ_GPU_Tensor4D_NCHW gpu_bias4;
		ZQ_GPU_Tensor4D_NCHW gpu_para4;
		ZQ_GPU_ConvolutionFilters_NCHW gpu_score_filters;
		ZQ_GPU_Tensor4D_NCHW gpu_score_bias;
		ZQ_GPU_ConvolutionFilters_NCHW gpu_location_filters;
		ZQ_GPU_Tensor4D_NCHW gpu_location_bias;
#endif
	};
}

#endif
