#define ZQ_CNN_USE_CUDA
#include "ZQ_MTCNN.h"
#include <omp.h>
#include "opencv2\opencv.hpp"

#ifdef ZQ_CNN_USE_CUDA
#pragma comment(lib,"ZQlibCNN.lib")
#endif
using namespace cv;
using namespace std;
using namespace ZQ;


int main()
{
	ZQ_CNN_Forward_CUDA::InitCUDNN();

	Mat image0 = cv::imread("data\\test2.jpg", 1);
	if (image0.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}

	std::vector<ZQ_CNN_BBox> thirdBbox;
	ZQ_MTCNN find;
	if (!find.Initialize("model\\det1.nchwbin", "model\\det2.nchwbin", "model\\det3.nchwbin"))
	{
		cout << "failed to init MTCNN\n";
		return EXIT_FAILURE;
	}

	find.SetParas(image0.cols, image0.rows, 20, 0.6, 0.7, 0.7, 0.5, 0.5, 0.5, true, true);

	double t1 = omp_get_wtime();
	int nIters = 1;
	for (int i = 0; i < nIters; i++)
	{
		if (!find.FindFace(image0, thirdBbox))
		{
			cout << "failed to find face!\n";
			return EXIT_FAILURE;
		}
		
	}
	double t2 = omp_get_wtime();
	printf("[%d] iters cost %.3f secs, 1 iter costs %.3f ms\n", nIters, t2 - t1, 1000 * (t2 - t1) / nIters);
	find.Draw(image0, thirdBbox);


	namedWindow("result0");
	imshow("result0", image0);
	imwrite("result0.jpg", image0);

	waitKey(0);
	return EXIT_SUCCESS;
}

int main2()
{


	Mat image;
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cout << "fail to open!\n";
		return EXIT_FAILURE;
	}
	cap >> image;
	if (!image.data) {
		cout << "��ȡ��Ƶʧ��\n";
		return EXIT_FAILURE;
	}
	ZQ_MTCNN find;
	if (!find.Initialize("model\\Pnet.txt", "model\\Rnet.txt", "model\\Onet.txt"))
	{
		cout << "failed to init MTCNN\n";
		return EXIT_FAILURE;
	}
	find.SetParas(image.cols, image.rows, 60, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, true);
	double start;
	int stop = 1200;
	while (stop--) {
		start = omp_get_wtime();
		cap >> image;
		std::vector<ZQ_CNN_BBox> thirdBbox;
		find.FindFace(image, thirdBbox);
		start = omp_get_wtime() - start;
		cout << "find " << thirdBbox.size() << " time is  " << start << endl;
		find.Draw(image, thirdBbox);
		imshow("result", image);
		if (waitKey(30) >= 0) break;


	}
	image.release();
	return EXIT_SUCCESS;
}