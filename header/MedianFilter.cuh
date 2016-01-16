/*
 * MedianFilter.cuh
 *
 *      Author: meetshah1995
 */

#ifndef MEDIANFILTER_CUH_
#define MEDIANFILTER_CUH_

class MedianFilter {
public:
	__host__ __device__ MedianFilter(int value) { this->size = value; newMatrixSize = (value*value); }
	__host__ __device__ ~MedianFilter() {}
	__host__ __device__ void operator()(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *resultRed, unsigned char *resultGreen, unsigned char *resultBlue, int width, int height) {

		if(size > 7) {
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

		int redTable[100];
		int blueTable[100];
		int greenTable[100];

		int counter = 0;

		for(j=-newSize; j<=newSize; ++j) {
			for(i=-newSize; i<=newSize; ++i) {
				redTable[counter] = red[(row+j)*width + (col+i)];
				greenTable[counter] = green[(row+j)*width + (col+i)];
				blueTable[counter] = blue[(row+j)*width + (col+i)];
				++counter;
			}
		}

		int rmin, gmin, bmin;
		for(j=0; j<newMatrixSize-1; j++) {
			rmin = j;
			gmin = j;
			bmin = j;
			for(i=j+1; i<newMatrixSize; i++) {
				if(redTable[i] < redTable[rmin]) rmin = i;
				if(blueTable[i] < blueTable[bmin]) bmin = i;
				if(greenTable[i] < greenTable[gmin]) gmin = i;
			}

			int tmp = redTable[rmin];
			redTable[rmin] = redTable[j];
			redTable[j] = tmp;

			tmp = blueTable[bmin];
			blueTable[bmin] = blueTable[j];
			blueTable[j] = tmp;

			tmp = greenTable[gmin];
			greenTable[gmin] = greenTable[j];
			greenTable[j] = tmp;
		}

		resultRed[row*width+col] = redTable[newMatrixSize/2];
		resultGreen[row*width+col] = greenTable[newMatrixSize/2];
		resultBlue[row*width+col] = blueTable[newMatrixSize/2];

	}

private:
	int size;
	int newMatrixSize;
};

#endif /* MEDIANFILTER_CUH_ */
