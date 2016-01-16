/*
 * GaussianFilter.cuh
 *
 *      Author: meetshah1995
 */

#ifndef GaussianFilter_CUH_
#define GaussianFilter_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

class GaussianFilter {
public:
	__host__ __device__ GaussianFilter() {}
	__host__ __device__ ~GaussianFilter() {}
	__host__ __device__ void operator()(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *resultRed, unsigned char *resultGreen, unsigned char *resultBlue, int width, int height) {
		int col = blockIdx.x*blockDim.x + threadIdx.x;
		int row = blockIdx.y*blockDim.y + threadIdx.y;
		if(col > width-1 || row > height-1)
			return;
		if(row < 1 || col < 1 || row > height-2 || col > width-2) {
			resultRed[row*width+col] = 0;
			resultGreen[row*width+col] = 0;
			resultBlue[row*width+col] = 0;
			return;
		}

		int mask[3][3] = {1,2,1, 2,3,2, 1,2,1};
		int sumRed = 0;
		int sumGreen = 0;
		int sumBlue = 0;

		for(int j=-1; j<=1; ++j) {
			for(int i=-1; i<=1; ++i) {
				int colorRed = red[(row+j)*width + (col+i)];
				int colorGreen = green[(row+j)*width + (col+i)];
				int colorBlue = blue[(row+j)*width + (col+i)];
				sumRed += colorRed*mask[i+1][j+1];
				sumGreen += colorGreen*mask[i+1][j+1];
				sumBlue += colorBlue*mask[i+1][j+1];
			}
		}
		resultRed[row*width+col] = sumRed/15;
		resultGreen[row*width+col] = sumGreen/15;
		resultBlue[row*width+col] = sumBlue/15;
	}
};


#endif /* GaussianFilter_CUH_ */
