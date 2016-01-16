/*
 * Thresholding.cuh
 *
 *      Author: meetshah1995
 */

#ifndef THRESHOLDING_CUH_
#define THRESHOLDING_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include "../header/IPNCLImage.h"
#include <cmath>

class Thresholding {
public:
	__host__ __device__ Thresholding(int threshold) {
		this->threshold = threshold;
	}
	__host__ __device__ ~Thresholding() {}
	__host__ __device__ void operator()(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *resultRed, unsigned char *resultGreen, unsigned char *resultBlue, int width, int height) {
		int col = blockIdx.x*blockDim.x + threadIdx.x;
		int row = blockIdx.y*blockDim.y + threadIdx.y;
		if(col > width-1 || row > height-1)
			return;
		if(row < 0 || col < 0 || row > height-1 || col > width-1) {
			return;
		}

		int newValue = (red[(row)*width + (col)]+green[(row)*width + (col)]+blue[(row)*width + (col)])/3;

		resultRed[row*width+col] = newValue;
		resultGreen[row*width+col] = newValue;
		resultBlue[row*width+col] = newValue;

		if(resultRed[row*width+col] < threshold) {
			resultRed[row*width+col] = 0;
		} else {
			resultRed[row*width+col] = 255;
		}

		if(resultBlue[row*width+col] < threshold) {
			resultBlue[row*width+col] = 0;
		} else {
			resultBlue[row*width+col] = 255;
		}

		if(resultGreen[row*width+col] < threshold) {
			resultGreen[row*width+col] = 0;
		} else {
			resultGreen[row*width+col] = 255;
		}

	}
private:
	int threshold;
};


#endif /* THRESHOLDING_CUH_ */
