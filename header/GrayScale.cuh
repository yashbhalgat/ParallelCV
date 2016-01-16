/*
 * GrayScale.cuh
 *
 *      Author: meetshah1995
 */

#ifndef GRAYSCALE_CUH_
#define GRAYSCALE_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

class GrayScale {
public:
	__host__ __device__ GrayScale() {}
	__host__ __device__ ~GrayScale() {}
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
	}
};


#endif /* GRAYSCALE_CUH_ */
