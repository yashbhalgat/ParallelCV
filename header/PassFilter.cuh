/*
 * HighPassFilterMN.cuh
 *
 *      Author: meetshah1995
 */

#ifndef HIGHPASSFILTER_CUH_
#define HIGHPASSFILTER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

class PassFilter {
public:
	__host__ __device__ PassFilter(int mask[3][3]) {
		for(int i=0; i<3; ++i) {
			for(int j=0; j<3; ++j) {
				this->mask[i][j] = mask[i][j];
			}
		}
		this->size = 3;
	}

	__host__ __device__ PassFilter(int **mask, int size) {
		for(int i=0; i<size; ++i) {
			for(int j=0; j<size; ++j) {
				this->mask[i][j] = mask[i][j];
			}
		}
		this->size = size;
	}
 	__host__ __device__ ~PassFilter() {}
	__host__ __device__ void operator()(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *resultRed, unsigned char *resultGreen, unsigned char *resultBlue, int width, int height) {

		if(size > 7 || !(size%2)) {
			return;
		}

		int newSize = size/2;
		int i, j;

		int col = blockIdx.x*blockDim.x + threadIdx.x;
		int row = blockIdx.y*blockDim.y + threadIdx.y;
		if(col > width-1 || row > height-1)
			return;
		if(row < newSize || col < newSize || row > (height-newSize)-1 || col > (width-newSize)-1) {
			resultRed[row*width+col] = 0;
			resultGreen[row*width+col] = 0;
			resultBlue[row*width+col] = 0;
			return;
		}
		int sumRed = 0;
		int sumGreen = 0;
		int sumBlue = 0;

		int divisor = 0;

		for(j=-newSize; j<=newSize; ++j) {
			for(i=-newSize; i<=newSize; ++i) {
				int colorRed = red[(row+j)*width + (col+i)];
				int colorGreen = green[(row+j)*width + (col+i)];
				int colorBlue = blue[(row+j)*width + (col+i)];
				sumRed += colorRed*mask[i+1][j+1];
				sumGreen += colorGreen*mask[i+1][j+1];
				sumBlue += colorBlue*mask[i+1][j+1];
				divisor += mask[i+1][j+1];
			}
		}

		if(sumRed >= 0 && sumRed <= 255) {
			if(divisor != 0) {
				resultRed[row*width+col] = sumRed/divisor;
			} else {
				resultRed[row*width+col] = sumRed;
			}
		} else if(sumRed < 0) {
			resultRed[row*width+col] = 0;
		} else {
			resultRed[row*width+col] = 255;
		}

		if(sumGreen >= 0 && sumGreen <= 255) {
			if(divisor != 0) {
				resultGreen[row*width+col] = sumGreen/divisor;
			} else {
				resultGreen[row*width+col] = sumGreen;
			}
		}else if(sumGreen < 0) {
			resultGreen[row*width+col] = 0;
		} else {
			resultGreen[row*width+col] = 255;
		}

		if(sumBlue >= 0 && sumBlue <= 255) {
			if(divisor != 0) {
				resultBlue[row*width+col] = sumBlue/divisor;
			} else {
				resultBlue[row*width+col] = sumBlue;
			}
		}else if(sumBlue < 0) {
			resultBlue[row*width+col] = 0;
		} else {
			resultBlue[row*width+col] = 255;
		}
	}

private:
	/*
	 * Hardcoded ;/
	 *
	 */
	int mask[10][10];
	int size;
};

#endif /* HIGHPASSFILTER_CUH_ */
