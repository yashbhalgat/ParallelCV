/*
 * Negative.cuh
 *
 *      Author: meetshah1995
 */

#ifndef NEGATIVE_CUH_
#define NEGATIVE_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

class Negative {
public:
	__host__ __device__ Negative() {}
	__host__ __device__ ~Negative() {}
	__host__ __device__ void operator()(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *resultRed, unsigned char *resultGreen, unsigned char *resultBlue, int width, int height) {
		int col = blockIdx.x*blockDim.x + threadIdx.x;
		int row = blockIdx.y*blockDim.y + threadIdx.y;
		if(col > width-1 || row > height-1)
			return;
		if(row < 0 || col < 0 || row > height-1 || col > width-1) {
			return;
		}

		resultRed[row*width+col] = 255 - red[(row)*width + (col)];
		resultGreen[row*width+col] = 255 - green[(row)*width + (col)];
		resultBlue[row*width+col] = 255 - blue[(row)*width + (col)];
	}
};


#endif /* NEGATIVE_CUH_ */
