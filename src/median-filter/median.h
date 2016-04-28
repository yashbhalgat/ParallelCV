#include <ctime>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

void medianGPU(Mat src, Mat& dst);

void medianGPU_opti(Mat src, Mat& dst);

void medianCPU(Mat& src, Mat& dst);
