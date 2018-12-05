#ifndef _ZQ_MTCNN_H_
#define _ZQ_MTCNN_H_
#pragma once

#include "ZQ_MTCNN_Pnet.h"
#include "ZQ_MTCNN_Rnet.h"
#include "ZQ_MTCNN_Onet.h"
#include <iostream>
#include <time.h>
#include <omp.h>

namespace ZQ
{
	class ZQ_MTCNN
	{
		enum CONST_VAL { MTCNN_MAX_THREADS = 16 };
	public:
		ZQ_MTCNN()
		{
			SetParas(640, 480);
		}

		~ZQ_MTCNN()
		{
			_clear();
		}

		bool Initialize(const std::string& pnet_filename = "Pnet.txt", const std::string& rnet_filename = "Rnet.txt", const std::string& onet_filename = "Onet.txt")
		{
			if (!simpleFace_PNet[0].Initialize(pnet_filename))
				return false;
			for (int i = 1; i < MTCNN_MAX_THREADS; i++)
				simpleFace_PNet[i].CopyData(simpleFace_PNet[0]);

			if (!refineNet.Initialize(rnet_filename))
				return false;
			if (!outNet.Initialize(onet_filename))
				return false;
			return true;
		}

		void SetParas(int width, int height, int min_size = 30, float thresh_p = 0.6, float thresh_r = 0.7, float thresh_o = 0.6,
			float thresh_nms_p = 0.7, float thresh_nms_r = 0.7, float thresh_nms_o = 0.7, bool use_CUDA = true, bool display = false, float factor = 0.709)
		{
			threshold[0] = thresh_p;
			threshold[1] = thresh_r;
			threshold[2] = thresh_o;
			nms_threshold[0] = thresh_nms_p;
			nms_threshold[1] = thresh_nms_r;
			nms_threshold[2] = thresh_nms_o;
			this->min_size = min_size;
			this->factor = factor;
			_init(height, width);
			this->width = width;
			this->height = height;
			useCUDA = use_CUDA;
			this->display = display;
		}

		bool FindFace(const cv::Mat &image, std::vector<ZQ_CNN_BBox>& outBBox)
		{
			double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
			outBBox.clear();
			if (image.rows != height || image.cols != width)
				return false;
			_clear();
#ifdef ZQ_CNN_USE_CUDA
			if (useCUDA)
			{
				ZQ_CPU_Tensor4D_NCHW<float> input;
				ZQ_GPU_Tensor4D_NCHW gpu_input;
				ZQ_CNN_Utils_OPENCV::Image2Matrix(image, input);
				ZQ_CNN_Forward_CUDA::Upload(input, gpu_input);
				t1 = omp_get_wtime();
				if (!_Pstage_CUDA(gpu_input))
				//if (!_Pstage_CUDA(image))
					return false;
				t2 = omp_get_wtime();
				if (!_Rstage_CUDA(gpu_input))
				//if (!_Rstage_CUDA(image))
					return false;

				t3 = omp_get_wtime();
				if (!_Ostage_CUDA(gpu_input))
				//if (!_Ostage_CUDA(image))
					return false;
			}
			else
			{
#endif
				t1 = omp_get_wtime();
				if (!_Pstage(image))
					return false;

				t2 = omp_get_wtime();
				if (!_Rstage(image))
					return false;

				t3 = omp_get_wtime();
				if (!_Ostage(image))
					return false;

#ifdef ZQ_CNN_USE_CUDA
			}
#endif
			t4 = omp_get_wtime();
			for (int i = 0; i < thirdBbox.size(); i++)
			{
				if (thirdBbox[i].exist)
				{
					outBBox.push_back(thirdBbox[i]);
				}
			}
			t5 = omp_get_wtime();
			if(display)
				printf("P: %.3f ms, R: %.3f ms, O: %.3f ms, out: %.3f ms\n",
					1000*(t2 - t1), 1000 * (t3 - t2), 1000 * (t4 - t3), 1000 * (t5 - t4));
			return true;
		}

		static void Draw(cv::Mat &image, const std::vector<ZQ_CNN_BBox>& thirdBbox)
		{
			std::vector<ZQ_CNN_BBox>::const_iterator it = thirdBbox.begin();
			for (; it != thirdBbox.end(); it++)
			{
				if ((*it).exist)
				{
					if (it->score > 0.9)
					{
						cv::rectangle(image, cv::Point((*it).col1, (*it).row1), cv::Point((*it).col2, (*it).row2), cv::Scalar(0, 0, 255), 2, 8, 0);
					}
					else
					{
						cv::rectangle(image, cv::Point((*it).col1, (*it).row1), cv::Point((*it).col2, (*it).row2), cv::Scalar(0, 255, 0), 2, 8, 0);
					}

					for (int num = 0; num < 5; num++)
						circle(image, cv::Point(*(it->ppoint + num)+0.5f, *(it->ppoint + num + 5)+0.5f), 3, cv::Scalar(0, 255, 255), -1);
				}
				else
				{
					printf("not exist!\n");
				}
			}
		}

	private:
		void _clear()
		{
			firstBbox.clear();
			firstOrderScore.clear();
			secondBbox.clear();
			secondBboxScore.clear();
			thirdBbox.clear();
			thirdBboxScore.clear();
		}

		void _init(int row, int col)
		{
			float minl = row < col ? row : col;
			int MIN_DET_SIZE = 12;
			float m = (float)MIN_DET_SIZE / min_size;
			minl *= m;
			//float factor = 0.709;
			//float factor = 0.85;
			//float factor = 0.5;
			int factor_count = 0;

			scales.clear();
			while (minl > MIN_DET_SIZE)
			{
				if (factor_count > 0)m = m*factor;
				scales.push_back(m);
				minl *= factor;
				factor_count++;
			}
			float minside = row < col ? row : col;
			int count = 0;
			std::vector<float>::iterator it = scales.begin();
			for (; it != scales.end(); it++)
			{
				if (*it > 1)
				{
					std::cout << "the minsize is too small" << std::endl;
				}
				if (*it < (MIN_DET_SIZE / minside))
				{
					scales.resize(count);
					break;
				}
				count++;
			}
		}

		bool _Pstage(const cv::Mat& image)
		{
			ZQ_CNN_OrderScore order;
			int count = 0;
			std::vector<cv::Mat> reImages(scales.size());
			double t00 = omp_get_wtime();

			int ori_width = image.cols;
			int ori_height = image.rows;
			cv::Mat ori_image = image;


			//cv::pyrDown(image, ori_image);

			//#pragma omp parallel for schedule(dynamic,MTCNN_MAX_THREADS)
			for (int i = 0; i < scales.size(); i++)
			{
				int changedH = (int)ceil(ori_height*scales[i]);
				int changedW = (int)ceil(ori_width*scales[i]);
				if (changedH < 2 || changedW < 2)
					continue;

				cv::resize(ori_image, reImages[i], cv::Size(changedW, changedH), 0, 0, cv::INTER_LINEAR);

			}
			double t01 = omp_get_wtime();
			//printf(" resize:%.3f\n", t01 - t00);

			std::vector<std::vector<ZQ_CNN_BBox>> bounding_boxes(scales.size());
			//#pragma omp parallel for schedule(static, MTCNN_MAX_THREADS)
			for (int i = 0; i < scales.size(); i++)
			{
				int changedH = (int)ceil(ori_height*scales[i]);
				int changedW = (int)ceil(ori_width*scales[i]);
				if (changedH < 2 || changedW < 2)
					continue;
				int id = i%MTCNN_MAX_THREADS;
				double t2 = omp_get_wtime();
				simpleFace_PNet[id].Run(reImages[i], scales[i], threshold[0],false);
				double t3 = omp_get_wtime();
				ZQ_CNN_Forward::NMS(simpleFace_PNet[id].boundingBox, simpleFace_PNet[id].bboxScore, nms_threshold[0]);

				//printf("i = %d, pnet: %.3f\n", i, (t3 - t2));

				bounding_boxes[i] = simpleFace_PNet[id].boundingBox;

				simpleFace_PNet[id].bboxScore.clear();
				simpleFace_PNet[id].boundingBox.clear();
			}


			for (int i = 0; i < scales.size(); i++)
			{
				std::vector<ZQ_CNN_BBox>::iterator it = bounding_boxes[i].begin();
				for (; it != bounding_boxes[i].end(); it++)
				{
					if ((*it).exist)
					{
						firstBbox.push_back(*it);
						order.score = (*it).score;
						order.oriOrder = count;
						firstOrderScore.push_back(order);
						count++;
					}
				}
			}
			//the first stage's nms
			if (count < 1) return false;
			ZQ_CNN_Forward::NMS(firstBbox, firstOrderScore, nms_threshold[0]);
			ZQ_CNN_Forward::RefineAndSquareBbox(firstBbox, image.cols, image.rows);

			return true;
		}

#ifdef ZQ_CNN_USE_CUDA
		bool _Pstage_CUDA(const cv::Mat& input)
		{
			ZQ_CNN_OrderScore order;
			int count = 0;
			std::vector<cv::Mat> reImages(scales.size());
			double t00 = omp_get_wtime();

			int ori_width = input.cols;
			int ori_height = input.rows;

			for (int i = 0; i < scales.size(); i++)
			{
				int changedH = (int)ceil(ori_height*scales[i]);
				int changedW = (int)ceil(ori_width*scales[i]);
				if (changedH < 2 || changedW < 2)
					continue;
				cv::resize(input, reImages[i], cv::Size(changedW, changedH));
			}
			double t01 = omp_get_wtime();
			//printf(" resize:%.3f\n", t01 - t00);

			std::vector<std::vector<ZQ_CNN_BBox>> bounding_boxes(scales.size());
			for (int i = 0; i < scales.size(); i++)
			{
				int changedH = (int)ceil(ori_height*scales[i]);
				int changedW = (int)ceil(ori_width*scales[i]);
				if (changedH < 2 || changedW < 2)
					continue;
				int id = i%MTCNN_MAX_THREADS;
				double t2 = omp_get_wtime();
				ZQ_CPU_Tensor4D_NCHW<float> tmp;
				ZQ_GPU_Tensor4D_NCHW tmp_gpu;
				ZQ_CNN_Utils_OPENCV::Image2Matrix(reImages[i], tmp);
				ZQ_CNN_Forward_CUDA::Upload(tmp, tmp_gpu);
				simpleFace_PNet[id].Run_CUDA(tmp_gpu, scales[i], threshold[0]);
				double t3 = omp_get_wtime();
				ZQ_CNN_Forward::NMS(simpleFace_PNet[id].boundingBox, simpleFace_PNet[id].bboxScore, nms_threshold[0], "Union", 4);

				//printf("i = %d, pnet: %.3f\n", i, (t3 - t2));

				bounding_boxes[i] = simpleFace_PNet[id].boundingBox;

				simpleFace_PNet[id].bboxScore.clear();
				simpleFace_PNet[id].boundingBox.clear();
			}


			for (int i = 0; i < scales.size(); i++)
			{
				std::vector<ZQ_CNN_BBox>::iterator it = bounding_boxes[i].begin();
				for (; it != bounding_boxes[i].end(); it++)
				{
					if ((*it).exist)
					{
						firstBbox.push_back(*it);
						order.score = (*it).score;
						order.oriOrder = count;
						firstOrderScore.push_back(order);
						count++;
					}
				}
			}
			//the first stage's nms
			if (count < 1) return false;
			printf("stage1 count = %d\n", firstBbox.size());
			ZQ_CNN_Forward::NMS(firstBbox, firstOrderScore, nms_threshold[0]);
			printf("stage1 count = %d\n", firstBbox.size());
			ZQ_CNN_Forward::RefineAndSquareBbox(firstBbox, ori_width, ori_height);

			return true;
		}

		bool _Pstage_CUDA(const ZQ_GPU_Tensor4D_NCHW& input)
		{
			ZQ_CNN_OrderScore order;
			int count = 0;
			std::vector<ZQ_GPU_Tensor4D_NCHW> reImages(scales.size());
			float resize_time = 0;
			
			int ori_width = input.GetW();
			int ori_height = input.GetH();
			
			for (int i = 0; i < scales.size(); i++)
			{
				int changedH = (int)ceil(ori_height*scales[i]);
				int changedW = (int)ceil(ori_width*scales[i]);
				if (changedH < 2 || changedW < 2)
					continue;

				float tmp_time = 0;
				ZQ_CNN_Forward_CUDA::Resize(input, changedH, changedW, reImages[i], tmp_time);
				resize_time += tmp_time;
			}
			printf(" resize:%.3f ms\n", resize_time);

			std::vector<std::vector<ZQ_CNN_BBox>> bounding_boxes(scales.size());
			for (int i = 0; i < scales.size(); i++)
			{
				int changedH = (int)ceil(ori_height*scales[i]);
				int changedW = (int)ceil(ori_width*scales[i]);
				if (changedH < 2 || changedW < 2)
					continue;
				bool useCUDNN = changedH*changedW >= 10000;
				int id = i%MTCNN_MAX_THREADS;
				double t2 = omp_get_wtime();
				simpleFace_PNet[id].Run_CUDA(reImages[i], scales[i], threshold[0], false, useCUDNN);
				double t3 = omp_get_wtime();
				ZQ_CNN_Forward::NMS(simpleFace_PNet[id].boundingBox, simpleFace_PNet[id].bboxScore, nms_threshold[0], "Union", 4);

				printf("i = %d, pnet: %.3f ms\n", i, 1000*(t3 - t2));

				bounding_boxes[i] = simpleFace_PNet[id].boundingBox;

				simpleFace_PNet[id].bboxScore.clear();
				simpleFace_PNet[id].boundingBox.clear();
			}


			for (int i = 0; i < scales.size(); i++)
			{
				std::vector<ZQ_CNN_BBox>::iterator it = bounding_boxes[i].begin();
				for (; it != bounding_boxes[i].end(); it++)
				{
					if ((*it).exist)
					{
						firstBbox.push_back(*it);
						order.score = (*it).score;
						order.oriOrder = count;
						firstOrderScore.push_back(order);
						count++;
					}
				}
			}
			//the first stage's nms
			if (count < 1) return false;
			printf("stage1 count = %d\n", firstBbox.size());
			ZQ_CNN_Forward::NMS(firstBbox, firstOrderScore, nms_threshold[0]);
			printf("stage1 count = %d\n", firstBbox.size());
			ZQ_CNN_Forward::RefineAndSquareBbox(firstBbox, ori_width, ori_height);

			return true;
		}
#endif



		bool _Rstage(const cv::Mat& image)
		{
			ZQ_CNN_OrderScore order;

			//second stage
			int count = 0;
			std::vector<ZQ_CNN_BBox>::iterator it = firstBbox.begin();
			double t1 = omp_get_wtime();
			double part1 = 0, part2 = 0;
			for (; it != firstBbox.end(); it++)
			{
				if ((*it).exist)
				{
					cv::Rect temp((*it).col1, (*it).row1, (*it).col2 - (*it).col1, (*it).row2 - (*it).row1);
					if (temp.width <= 2 || temp.height <= 2)
					{
						(*it).exist = false;
						continue;
					}
					cv::Mat secImage;
					double t11 = omp_get_wtime();
					cv::resize(image(temp), secImage, cv::Size(24, 24), 0, 0, cv::INTER_LINEAR);
					double t12 = omp_get_wtime();
					refineNet.Run(secImage, threshold[1]);
					double t13 = omp_get_wtime();
					if (*(refineNet.score.pData + 1) > refineNet.Rthreshold)
					{
						for (int j = 0; j < 4; j++)
							it->regreCoord[j] = refineNet.location.pData[j];
						it->area = (it->row2 - it->row1)*(it->col2 - it->col1);
						it->score = *(refineNet.score.pData + 1);
						secondBbox.push_back(*it);
						order.score = it->score;
						order.oriOrder = count++;
						secondBboxScore.push_back(order);
					}
					else
					{
						(*it).exist = false;
					}
					part1 += t12 - t11;
					part2 += t13 - t12;
				}
			}
			if (count < 1)
				return false;
			double t2 = omp_get_wtime();
			ZQ_CNN_Forward::NMS(secondBbox, secondBboxScore, nms_threshold[1]);
			double t3 = omp_get_wtime();
			ZQ_CNN_Forward::RefineAndSquareBbox(secondBbox, image.cols, image.rows);
			double t4 = omp_get_wtime();
			if(display)
				printf("rnet: %.3f ms(resize:%.3f ms, run: %.3f ms), nms: %.3f ms, refine: %.3f ms\n", 
					1000 * (t2 - t1), 1000 * part1, 1000 * part2, 1000 * (t3 - t2), 1000 * (t4 - t3));
			return true;
		}

#ifdef ZQ_CNN_USE_CUDA
		bool _Rstage_CUDA(const ZQ_GPU_Tensor4D_NCHW& input)
		{
			ZQ_CNN_OrderScore order;
			ZQ_GPU_Tensor4D_NCHW reImage;
			float resize_time = 0;
			//second stage
			int count = 0;
			double t1 = omp_get_wtime();
			double part1 = 0, part2 = 0;
			const int batch_size = 512;
			std::vector<int> rects;
			std::vector<int> ids;
			int cur_batch_size = 0;
			for(int i = 0;i < firstBbox.size();i++)
			{
				if (cur_batch_size == batch_size || (i == firstBbox.size() - 1 && cur_batch_size != 0))
				{
					double t11 = omp_get_wtime();
					float tmp_time = 0;
					ZQ_CNN_Forward_CUDA::ResizeRect(input, rects, 24, 24, reImage,tmp_time);
					resize_time += tmp_time;
					double t12 = omp_get_wtime();
					refineNet.Run_Batch_CUDA(reImage, threshold[1]);
					double t13 = omp_get_wtime();
					for (int n = 0; n < cur_batch_size; n++)
					{
						int cur_id = ids[n];
						if (refineNet.score.pData[n * 2 + 1] > refineNet.Rthreshold)
						{

							for (int j = 0; j < 4; j++)
								firstBbox[cur_id].regreCoord[j] = refineNet.location.pData[n * 4 + j];
							firstBbox[cur_id].area = (firstBbox[cur_id].row2 - firstBbox[cur_id].row1)*(firstBbox[cur_id].col2 - firstBbox[cur_id].col1);
							firstBbox[cur_id].score = refineNet.score.pData[2 * n + 1];
							secondBbox.push_back(firstBbox[cur_id]);
							order.score = firstBbox[cur_id].score;
							order.oriOrder = count++;
							secondBboxScore.push_back(order);
						}
						else
						{
							firstBbox[cur_id].exist = false;
						}
					}
					rects.clear();
					ids.clear();
					cur_batch_size = 0;

					part1 += resize_time/1000;
					part2 += t13 - t12;
				}

				if (firstBbox[i].exist)
				{
					cv::Rect temp(firstBbox[i].col1, firstBbox[i].row1, firstBbox[i].col2 - firstBbox[i].col1, firstBbox[i].row2 - firstBbox[i].row1);
					if (temp.width <= 2 || temp.height <= 2)
					{
						firstBbox[i].exist = false;
						continue;
					}
					rects.push_back(temp.x);
					rects.push_back(temp.y);
					rects.push_back(temp.width);
					rects.push_back(temp.height);
					ids.push_back(i);
					cur_batch_size++;

					
				}
			}
			if (count < 1)
				return false;
			double t2 = omp_get_wtime();
			printf("stage2 count = %d\n", secondBbox.size());
			ZQ_CNN_Forward::NMS(secondBbox, secondBboxScore, nms_threshold[1]);
			printf("stage2 count = %d\n", secondBbox.size());
			double t3 = omp_get_wtime();
			ZQ_CNN_Forward::RefineAndSquareBbox(secondBbox, input.GetW(), input.GetH());
			double t4 = omp_get_wtime();
			if(display)
				printf("rnet: %.3f ms (resize:%.3f ms, run: %.3f ms), nms: %.3f ms, refine: %.3f ms\n", 
					1000 * (t2 - t1), 1000 * part1, 1000 * part2, 1000 * (t3 - t2), 1000 * (t4 - t3));
			return true;
		}

		bool _Rstage_CUDA(const cv::Mat& input)
		{
			ZQ_CNN_OrderScore order;
			ZQ_CPU_Tensor4D_NCHW<float> reImage;
			ZQ_GPU_Tensor4D_NCHW gpu_reImage;
			//second stage
			int count = 0;
			double t1 = omp_get_wtime();
			double part1 = 0, part2 = 0;
			const int batch_size = 512;
			std::vector<int> rects;
			std::vector<int> ids;
			int cur_batch_size = 0;
			for (int i = 0; i < firstBbox.size(); i++)
			{
				if (cur_batch_size == batch_size || (i == firstBbox.size() - 1 && cur_batch_size != 0))
				{
					double t11 = omp_get_wtime();
					ZQ_CNN_Utils_OPENCV::Image2Matrix_rects(input, rects, 24, 24, reImage);
					ZQ_CNN_Forward_CUDA::Upload(reImage, gpu_reImage);
					double t12 = omp_get_wtime();
					refineNet.Run_Batch_CUDA(gpu_reImage, threshold[1]);
					double t13 = omp_get_wtime();
					for (int n = 0; n < cur_batch_size; n++)
					{
						int cur_id = ids[n];
						if (refineNet.score.pData[n * 2 + 1] > refineNet.Rthreshold)
						{

							for (int j = 0; j < 4; j++)
								firstBbox[cur_id].regreCoord[j] = refineNet.location.pData[n * 4 + j];
							firstBbox[cur_id].area = (firstBbox[cur_id].row2 - firstBbox[cur_id].row1)*(firstBbox[cur_id].col2 - firstBbox[cur_id].col1);
							firstBbox[cur_id].score = refineNet.score.pData[2 * n + 1];
							secondBbox.push_back(firstBbox[cur_id]);
							order.score = firstBbox[cur_id].score;
							order.oriOrder = count++;
							secondBboxScore.push_back(order);
						}
						else
						{
							firstBbox[cur_id].exist = false;
						}
					}
					rects.clear();
					ids.clear();
					cur_batch_size = 0;

					part1 += t12 - t11;
					part2 += t13 - t12;
				}

				if (firstBbox[i].exist)
				{
					cv::Rect temp(firstBbox[i].col1, firstBbox[i].row1, firstBbox[i].col2 - firstBbox[i].col1, firstBbox[i].row2 - firstBbox[i].row1);
					if (temp.width <= 2 || temp.height <= 2)
					{
						firstBbox[i].exist = false;
						continue;
					}
					rects.push_back(temp.x);
					rects.push_back(temp.y);
					rects.push_back(temp.width);
					rects.push_back(temp.height);
					ids.push_back(i);
					cur_batch_size++;


				}
			}
			if (count < 1)
				return false;
			double t2 = omp_get_wtime();
			printf("stage2 count = %d\n", secondBbox.size());
			ZQ_CNN_Forward::NMS(secondBbox, secondBboxScore, nms_threshold[1]);
			printf("stage2 count = %d\n", secondBbox.size());
			double t3 = omp_get_wtime();
			ZQ_CNN_Forward::RefineAndSquareBbox(secondBbox, input.cols, input.rows);
			double t4 = omp_get_wtime();
			if (display)
				printf("rnet: %.3f ms (resize:%.3f ms, run: %.3f ms), nms: %.3f ms, refine: %.3f ms\n", 
					1000 * (t2 - t1), 1000 * part1, 1000 * part2, 1000 * (t3 - t2), 1000 * (t4 - t3));
			return true;
		}

#endif

		bool _Ostage(const cv::Mat& image)
		{
			ZQ_CNN_OrderScore order;

			//third stage 
			int count = 0;
			std::vector<ZQ_CNN_BBox>::iterator it = secondBbox.begin();
			double t1 = omp_get_wtime();
			double part1 = 0, part2 = 0;
			for (; it != secondBbox.end(); it++)
			{
				if ((*it).exist)
				{
					cv::Rect temp((*it).col1, (*it).row1, (*it).col2 - (*it).col1, (*it).row2 - (*it).row1);
					if (temp.width <= 2 || temp.height <= 2)
					{
						(*it).exist = false;
						continue;
					}
					cv::Mat thirdImage;
					double t11 = omp_get_wtime();
					cv::resize(image(temp), thirdImage, cv::Size(48, 48), 0, 0, cv::INTER_LINEAR);
					double t12 = omp_get_wtime();
					outNet.Run(thirdImage);
					double t13 = omp_get_wtime();
					using BaseType = ZQ_MTCNN_Onet::BaseType;
					BaseType *pp = NULL;
					if (*(outNet.score.pData + 1) > threshold[2])
					{
						for (int j = 0; j < 4; j++)
							it->regreCoord[j] = outNet.location.pData[j];
						it->area = (it->row2 - it->row1)*(it->col2 - it->col1);
						it->score = *(outNet.score.pData + 1);
						pp = outNet.keyPoint.pData;
						for (int num = 0; num < 5; num++)
						{
							(it->ppoint)[num] = it->col1 + (it->col2 - it->col1)*(*(pp + num));
						}
						for (int num = 0; num < 5; num++)
						{
							(it->ppoint)[num + 5] = it->row1 + (it->row2 - it->row1)*(*(pp + num + 5));
						}
						thirdBbox.push_back(*it);
						order.score = it->score;
						order.oriOrder = count++;
						thirdBboxScore.push_back(order);
					}
					else
					{
						it->exist = false;
					}
					part1 += t12 - t11;
					part2 += t13 - t12;
				}
			}

			if (count < 1)
				return false;
			double t2 = omp_get_wtime();
			ZQ_CNN_Forward::RefineAndSquareBbox(thirdBbox, image.cols, image.rows);
			double t3 = omp_get_wtime();
			ZQ_CNN_Forward::NMS(thirdBbox, thirdBboxScore, nms_threshold[2], "Min");
			double t4 = omp_get_wtime();
			if(display)
				printf("onet: %.3f ms (resize: %.3f ms, onet:%.3f ms), refine: %.3f ms, nms: %.3f ms\n", 
					1000 * (t2 - t1), 1000 * part1, 1000 * part2, 1000 * (t3 - t2), 1000 * (t4 - t3));
			return true;
		}

#ifdef ZQ_CNN_USE_CUDA
		bool _Ostage_CUDA(const ZQ_GPU_Tensor4D_NCHW& input)
		{
			ZQ_CNN_OrderScore order;
			ZQ_GPU_Tensor4D_NCHW reImage;
			float resize_time = 0;
			//second stage
			int count = 0;
			double t1 = omp_get_wtime();
			double part1 = 0, part2 = 0;
			const int batch_size = 512;
			std::vector<int> rects;
			std::vector<int> ids;
			int cur_batch_size = 0;

			for(int i = 0;i < secondBbox.size();i++)
			{
				if (cur_batch_size == batch_size || (i == secondBbox.size() - 1 && cur_batch_size != 0))
				{
					double t11 = omp_get_wtime();
					float tmp_time = 0;
					ZQ_CNN_Forward_CUDA::ResizeRect(input, rects, 48, 48, reImage, tmp_time);
					resize_time += tmp_time;
					double t12 = omp_get_wtime();
					outNet.Run_Batch_CUDA(reImage);
					double t13 = omp_get_wtime();
					float *pp = NULL;
					for (int n = 0; n < cur_batch_size; n++)
					{
						int cur_id = ids[n];
						if (outNet.score.pData[n * 2 + 1] > threshold[2])
						{
							for (int j = 0; j < 4; j++)
								secondBbox[cur_id].regreCoord[j] = outNet.location.pData[n * 4 + j];
							secondBbox[cur_id].area = (secondBbox[cur_id].row2 - secondBbox[cur_id].row1)*(secondBbox[cur_id].col2 - secondBbox[cur_id].col1);
							secondBbox[cur_id].score = outNet.score.pData[n * 2 + 1];
							pp = outNet.keyPoint.pData;
							for (int num = 0; num < 5; num++)
							{
								secondBbox[cur_id].ppoint[num] = secondBbox[cur_id].col1 + (secondBbox[cur_id].col2 - secondBbox[cur_id].col1)*pp[n * 10 + num];
							}
							for (int num = 0; num < 5; num++)
							{
								secondBbox[cur_id].ppoint[num + 5] = secondBbox[cur_id].row1 + (secondBbox[cur_id].row2 - secondBbox[cur_id].row1)*pp[n * 10 + num + 5];
							}
							thirdBbox.push_back(secondBbox[cur_id]);
							order.score = secondBbox[cur_id].score;
							order.oriOrder = count++;
							thirdBboxScore.push_back(order);
						}
						else
						{
							secondBbox[cur_id].exist = false;
						}
					}
					rects.clear();
					ids.clear();
					cur_batch_size = 0;
					part1 += resize_time/1000;
					part2 += t13 - t12;
				}

				if (secondBbox[i].exist)
				{
					cv::Rect temp(secondBbox[i].col1, secondBbox[i].row1, secondBbox[i].col2 - secondBbox[i].col1, secondBbox[i].row2 - secondBbox[i].row1);
					if (temp.width <= 2 || temp.height <= 2)
					{
						secondBbox[i].exist = false;
						continue;
					}
					rects.push_back(temp.x);
					rects.push_back(temp.y);
					rects.push_back(temp.width);
					rects.push_back(temp.height);
					ids.push_back(i);
					cur_batch_size++;
				}
			}

			if (count < 1)
				return false;
			double t2 = omp_get_wtime();
			printf("stage3 count = %d\n", thirdBbox.size());
			ZQ_CNN_Forward::RefineAndSquareBbox(thirdBbox, input.GetW(), input.GetH());
			printf("stage3 count = %d\n", thirdBbox.size());
			double t3 = omp_get_wtime();
			ZQ_CNN_Forward::NMS(thirdBbox, thirdBboxScore, nms_threshold[2], "Min");
			double t4 = omp_get_wtime();
			if(display)
				printf("onet: %.3f ms (resize: %.3f ms, onet:%.3f ms), refine: %.3f ms, nms: %.3f ms\n", 
					1000 * (t2 - t1), 1000 * part1, 1000 * part2, 1000 * (t3 - t2), 1000 * (t4 - t3));
			return true;
		}

		bool _Ostage_CUDA(const cv::Mat& input)
		{
			ZQ_CNN_OrderScore order;
			ZQ_CPU_Tensor4D_NCHW<float> reImage; 
			ZQ_GPU_Tensor4D_NCHW gpu_reImage;
			//second stage
			int count = 0;
			double t1 = omp_get_wtime();
			double part1 = 0, part2 = 0;
			const int batch_size = 512;
			std::vector<int> rects;
			std::vector<int> ids;
			int cur_batch_size = 0;

			for (int i = 0; i < secondBbox.size(); i++)
			{
				if (cur_batch_size == batch_size || (i == secondBbox.size() - 1 && cur_batch_size != 0))
				{
					double t11 = omp_get_wtime();
					ZQ_CNN_Utils_OPENCV::Image2Matrix_rects(input, rects, 48, 48, reImage);
					ZQ_CNN_Forward_CUDA::Upload(reImage, gpu_reImage);
					double t12 = omp_get_wtime();
					outNet.Run_Batch_CUDA(gpu_reImage);
					double t13 = omp_get_wtime();
					float *pp = NULL;
					for (int n = 0; n < cur_batch_size; n++)
					{
						int cur_id = ids[n];
						if (outNet.score.pData[n * 2 + 1] > threshold[2])
						{
							for (int j = 0; j < 4; j++)
								secondBbox[cur_id].regreCoord[j] = outNet.location.pData[n * 4 + j];
							secondBbox[cur_id].area = (secondBbox[cur_id].row2 - secondBbox[cur_id].row1)*(secondBbox[cur_id].col2 - secondBbox[cur_id].col1);
							secondBbox[cur_id].score = outNet.score.pData[n * 2 + 1];
							pp = outNet.keyPoint.pData;
							for (int num = 0; num < 5; num++)
							{
								secondBbox[cur_id].ppoint[num] = secondBbox[cur_id].col1 + (secondBbox[cur_id].col2 - secondBbox[cur_id].col1)*pp[n * 10 + num];
							}
							for (int num = 0; num < 5; num++)
							{
								secondBbox[cur_id].ppoint[num + 5] = secondBbox[cur_id].row1 + (secondBbox[cur_id].row2 - secondBbox[cur_id].row1)*pp[n * 10 + num + 5];
							}
							thirdBbox.push_back(secondBbox[cur_id]);
							order.score = secondBbox[cur_id].score;
							order.oriOrder = count++;
							thirdBboxScore.push_back(order);
						}
						else
						{
							secondBbox[cur_id].exist = false;
						}
					}
					rects.clear();
					ids.clear();
					cur_batch_size = 0;
					part1 += t12 - t11;
					part2 += t13 - t12;
				}

				if (secondBbox[i].exist)
				{
					cv::Rect temp(secondBbox[i].col1, secondBbox[i].row1, secondBbox[i].col2 - secondBbox[i].col1, secondBbox[i].row2 - secondBbox[i].row1);
					if (temp.width <= 2 || temp.height <= 2)
					{
						secondBbox[i].exist = false;
						continue;
					}
					rects.push_back(temp.x);
					rects.push_back(temp.y);
					rects.push_back(temp.width);
					rects.push_back(temp.height);
					ids.push_back(i);
					cur_batch_size++;
				}
			}

			if (count < 1)
				return false;
			double t2 = omp_get_wtime();
			printf("stage3 count = %d\n", thirdBbox.size());
			ZQ_CNN_Forward::RefineAndSquareBbox(thirdBbox, input.cols, input.rows);
			printf("stage3 count = %d\n", thirdBbox.size());
			double t3 = omp_get_wtime();
			ZQ_CNN_Forward::NMS(thirdBbox, thirdBboxScore, nms_threshold[2], "Min");
			double t4 = omp_get_wtime();
			if (display)
				printf("onet: %.3f ms (resize: %.3f ms, onet:%.3f ms), refine: %.3f ms, nms: %.3f ms\n", 
					1000 * (t2 - t1), 1000 * part1, 1000 * part2, 1000 * (t3 - t2), 1000 * (t4 - t3));
			return true;
		}
#endif

	private:
		float factor;
		bool useCUDA;
		bool display;
		int width, height;
		cv::Mat reImage;
		float nms_threshold[3];
		float threshold[3];
		int min_size;
		std::vector<float> scales;
		ZQ_MTCNN_Pnet simpleFace_PNet[MTCNN_MAX_THREADS];
		std::vector<ZQ_CNN_BBox> firstBbox;
		std::vector<ZQ_CNN_OrderScore> firstOrderScore;
		ZQ_MTCNN_Rnet refineNet;
		std::vector<ZQ_CNN_BBox> secondBbox;
		std::vector<ZQ_CNN_OrderScore> secondBboxScore;
		ZQ_MTCNN_Onet outNet;
		std::vector<ZQ_CNN_BBox> thirdBbox;
		std::vector<ZQ_CNN_OrderScore> thirdBboxScore;
	};
}

#endif
