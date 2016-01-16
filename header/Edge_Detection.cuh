/*
 * Brightness.cuh
 *
 *      Author: meetshah1995
 */

#ifndef BRIGHTNESS_CUH_
#define BRIGHTNESS_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include "../header/IPNCLImage.h"
#include <cmath>

class Brightness {
public:
	__host__ __device__ Brightness(int b) {
		for(int i=0; i<256; ++i) {
			if(i+b < 0) {
				LUT[i] = 0;
			} else if(i+b > 255) {
				LUT[i] = 255;
			} else {
				LUT[i] = i+b;
			}
		}
	}
	__host__ __device__ ~Brightness() {}
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

#endif /* BRIGHTNESS_CUH_ */
