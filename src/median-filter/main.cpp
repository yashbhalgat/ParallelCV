#include "median.h"

int main(int argc, char** argv)
{
	namedWindow("src", WINDOW_AUTOSIZE);
	namedWindow("src", WINDOW_AUTOSIZE);
	string directory = "../input/";
	string filename = "tiger_pepper.jpg";
	filename = directory+filename;
	Mat src, dst;
	src = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	src.convertTo(src, CV_8UC1);
	clock_t start, stop;
	start = clock();
#if 1
	medianGPU_opti(src, dst);
	imwrite("tiger_median.jpg", dst);
	imwrite("tiger_pepper_input.jpg", src);
#else
	medianCPU(src, dst);
	//medianBlur(src, dst, 3);
	imwrite("lena_median_cpu.jpg", dst);
#endif
	stop = clock();
	cout<<"All cost time is "<<static_cast<double>(stop - start)*1000/CLOCKS_PER_SEC<<" ms"<<endl;
	dst = dst.rowRange(1, dst.rows - 1).colRange(1, dst.cols - 1);

	imshow("src", src);
	imshow("dst", dst);
	waitKey(0);
	return 0;
}

