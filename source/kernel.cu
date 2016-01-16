#include <iostream>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <cmath>
#include "../header/IPNCLImage.h"
#include "../header/GaussianFilter.cuh"
#include "../header/IPNCL.h"
#include "../header/Negative.cuh"
#include "../header/GrayScale.cuh"
#include "../header/GammaCorrection.cuh"
#include "../header/Brightness.cuh"
#include "../header/Thresholding.cuh"
#include "../header/Contrast.cuh"
#include "../header/PassFilter.cuh"
#include "../header/MedianFilter.cuh"
#include <time.h>

typedef unsigned char uchar;

using namespace cv;

int main(int argc, char **argv) {

	time_t start, koniec;
	double roznica;


	IPNCLImage obj2("median_after.jpg");


	/*IPNCL::filter<PassFilter>(&obj2, new PassFilter());
	obj2.showImage(2000);*/

	/*IPNCL::filter<Negative>(&obj2, new Negative());
	obj2.showImage(2000);

	IPNCL::filter<GrayScale>(&obj2, new GrayScale());
		obj2.showImage(2000);

	IPNCL::filter<GammaCorrection>(&obj2, new GammaCorrection(0.5));
		obj2.showImage(2000);

	IPNCL::filter<Brightness>(&obj2, new Brightness(-50));
		obj2.showImage(2000);

	IPNCL::filter<Thresholding>(&obj2, new Thresholding(50));
		obj2.showImage(2000);

	IPNCL::filter<Contrast>(&obj2, new Contrast(2));
		obj2.showImage(2000);
*/
	int mask[3][3] = {-1,-1,-1, 1,-2,1, 1,1,1};
	int **mask2 = new int* [3];
	for(int i=0; i<3; ++i) {
		mask2[i] = new int[3];
	}

	mask2[0][0] = -1;
	mask2[0][1] = -1;
	mask2[0][2] = -1;
	mask2[1][0] = 1;
	mask2[1][1] = -2;
	mask2[1][2] = 1;
	mask2[2][0] = 1;
	mask2[2][1] = 1;
	mask2[2][2] = 1;


	IPNCL::filter<PassFilter>(&obj2, new PassFilter(mask));
		obj2.showImage(10000);


	//IPNCL::filter<MedianFilter>(&obj2, new MedianFilter(3));
	/*IPNCL::filter<Thresholding>(&obj2, new Thresholding(50));
			obj2.showImage(2000);*/
	/*IPNCL::filter<GaussianFilter>(&obj2, new GaussianFilter());
			obj2.showImage(2000);*/
	//obj2.showImage(1000);
	IPNCL::filter<GammaCorrection>(&obj2, new GammaCorrection(0.1));


	obj2.showImage(10000);
	obj2.saveImage("median01.jpg");


	return 0;
}
