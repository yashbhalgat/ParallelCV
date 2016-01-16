/*
 * Contrast.cuh
 *
 *      Author: meetshah1995
 */

#ifndef CONTRAST_CUH_
#define CONTRAST_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include "../header/IPNCLImage.h"
#include <cmath>

class Contrast {
public:
	__host__ __device__ Contrast(double contrast) {
		for(int i=0; i<256; ++i) {
			int tmp = contrast*(i-127) + 127;
			if(tmp < 0) {
				LUT[i] = 0;
			} else if(tmp > 255) {
				LUT[i] = 255;
			} else {
				LUT[i] = tmp;
			}
		}
	}
	__host__ __device__ ~Contrast() {}
	__host__ __device__ void operator()(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *resultRed, unsigned char *resultGreen, unsigned char *resultBlue, int width, int height) {
		int col = blockIdx.x*blockDim.x + threadIdx.x;
		int row = blockIdx.y*blockDim.y + threadIdx.y;
		if(col > width-1 || row > height-1)
			return;
		if(row < 0 || col < 0 || row > height-1 || col > width-1) {
			return;
		}
		resultRed[row*width+col] = LUT[red[row*width+col]];
		resultGreen[row*width+col] = LUT[green[row*width+col]];
		resultBlue[row*width+col] = LUT[blue[row*width+col]];
	}

private:
	int LUT[256];
};

#endif /* CONTRAST_CUH_ */
