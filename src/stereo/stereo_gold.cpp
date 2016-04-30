/*
 * 5kk73 GPU assignment
 */

/*
 * The host code for dense stereo vision
 */

#include "stereo.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

extern "C"
void computeGold(unsigned char* h_inLeft, unsigned char* h_inRight, unsigned char * gold_outLeft, int imageHeight, int imageWidth, int grayMax);

void
computeGold(unsigned char* h_inLeft, unsigned char* h_inRight, unsigned char * gold_outLeft, int imageHeight, int imageWidth, int grayMax)
{

  // common variables
  int i,j;

  /****************************************************
   *
   * Pad the images with margin pixels to handle 
   * the SSD window operation at the boarder.
   *
   ****************************************************/

  printf("CPU: padding images\n");

  // set board width to half SAD window
  int marginWidth = WIN_SIZE / 2;
  int marginHeight = WIN_SIZE / 2;
  // set height of padded image
  int paddedHeight = imageHeight + marginWidth * 2;
  // set width of padded image
  int paddedWidth = imageWidth + marginWidth * 2;
  // memory size (Byte) of the padded image
  int mem_size_paddedRight = sizeof(unsigned char) * paddedHeight * paddedWidth;
  int mem_size_paddedLeft = mem_size_paddedRight;
  // malloc for padded left image
  unsigned char* paddedLeft = (unsigned char*) malloc(mem_size_paddedLeft);
  // malloc for padded right image
  unsigned char* paddedRight = (unsigned char*) malloc(mem_size_paddedRight);

  // initial the padded image and shifted image to '0'
  memset(paddedLeft, 0, mem_size_paddedLeft);
  memset(paddedRight, 0, mem_size_paddedRight);

  // copy the image to the padded image
  for(i = 0; i < imageHeight; i++){
    for(j = 0; j < imageWidth; j++){
      paddedLeft[(marginHeight+i)*paddedWidth + marginWidth + j] =	\
	h_inLeft[i*imageWidth + j];
      paddedRight[(marginHeight+i)*paddedWidth + marginWidth + j] =	\
	h_inRight[i*imageWidth + j];
    }
  }

  /****************************************************
   *
   * Generate Disparity Map
   *
   ****************************************************/

  printf("CPU: generating disparity map\n");

  // malloc for shifted right image
  unsigned char* shiftedRight = (unsigned char*) malloc(mem_size_paddedRight);
  // malloc for SSD image, use integer (4 Byte) to avoid over flow
  int mem_size_ssd = sizeof(unsigned int) * paddedHeight * paddedWidth;
  unsigned int* ssd = (unsigned int*) malloc(mem_size_ssd);
  // malloc for window over SSD
  int mem_size_winssd = sizeof(unsigned int) * imageHeight * imageWidth;
  unsigned int* winssd = (unsigned int*) malloc(mem_size_winssd);
  // minimun of window over SSD
  int mem_size_minssd = sizeof(unsigned int) * imageHeight * imageWidth;
  unsigned int* minssd = (unsigned int*) malloc(mem_size_minssd);
  // disparity map
  int mem_size_disparity = sizeof(unsigned char) * imageHeight * imageWidth;
  unsigned char* disparity = (unsigned char*) malloc(mem_size_disparity);
  // the shift value of the right image
  int shift;

  memset(shiftedRight, 0, mem_size_paddedRight);
  // initialize SSD to xFFFFFFFF
  memset(ssd, 255, mem_size_ssd);
  // initialize window over SSD to xFFFFFFFF
  memset(winssd, 255, mem_size_winssd);
  // initialize min window SSD to xFFFFFFFF
  memset(minssd, 255, mem_size_minssd);
  // initialize the disparity map to 0
  memset(disparity, 0, mem_size_disparity);

  for (shift = 0; shift < MAX_SHIFT; shift++){

    /**************************************************
     * For each shift value, perform:
     * 1. Shift the right image
     * 2. Wondow SSD/SAD between left and shifted right image
     * 3. Find the shift value that leads to minimun SSD 
     *************************************************/

    // make a shifted right image
    for(i = 0; i < paddedHeight; i++){
      for(j = 0; j < paddedWidth - shift - marginWidth; j++){
	shiftedRight[i*paddedWidth + shift + j] =	\
	  paddedRight[i*paddedWidth + j];
      }
    }

    // SSD/SAD
    for(i = 0; i < paddedHeight; i++){
      for(j = 0; j < paddedWidth; j++){
	int index = i * paddedWidth+j;
	int diff = paddedLeft[index] - shiftedRight[index];
	// SSD
	#if CORRELATION == 1
	ssd[index] = (unsigned int)(diff * diff);
	// SAD
	#elif CORRELATION == 2
	ssd[index] = (unsigned int)abs(diff);
	// Unknown
	#else
	printf("Unknown correlation method. See stereo.h for detail.\n");
	exit(1);
	#endif
      }
    }

    // accumulate SSD over a window
    for(i = 0; i < imageHeight; i++){
      for(j = 0; j < imageWidth; j++){
	// indexwinssd: index to the center of the window over ssd
	int indexwinssd = i * imageWidth+j;
	// accu is used to accumulate the sum over the window
	unsigned int accu = 0;
	// loop over the window
	for(int ii = -1*marginHeight; ii < marginHeight + 1; ii++){
	  for(int jj = -1*marginWidth; jj < marginWidth + 1; jj++){
	    // indexssd: index to the ssd position for accumulating the window
	    int indexssd = (i + marginHeight + ii)*paddedWidth + j + marginWidth + jj;
	    accu = accu + ssd[indexssd];
	  }//loop over window width
	}//loop over window height
	winssd[indexwinssd] = accu;
      }// loop over imageWidth
    }// loop over imageHeight

    // update the minimun of window over SSD and corresponding shift value
    for(i = 0; i < imageHeight; i++){
      for(j = 0; j < imageWidth; j++){
	int index = i * imageWidth + j;
	if(winssd[index] < minssd[index]){
	  minssd[index] = winssd[index];
	  disparity[index] = (unsigned char)shift;
	}
      }
    }
      
  }// loop over the shift value

  /****************************************************
   *
   * Histogram Equalization
   *
   ****************************************************/

  printf("CPU: performing histogram equalization\n");

  // histogram
  unsigned int hist[HIST_BIN];
  // cumulative distribution function
  int cdf[HIST_BIN];
  // trsnafer function for histogram equalization
  unsigned int trans[HIST_BIN];
  // minimun non-zero value of cumulative distribution function
  int mincdf = 0;

  // initialize histogram to 0
  memset(hist, 0, sizeof(unsigned int) * HIST_BIN);
  // initialize cumulative distribution function to 0
  memset(cdf, 0, sizeof(int) * HIST_BIN);
  
  for(i = 0; i < imageHeight*imageWidth; i++){
      hist[disparity[i]]++;
  }

  cdf[0] = hist[0];
  for(i = 1; i < HIST_BIN; i++){
    cdf[i] = cdf[i-1] + hist[i];
  }

  i = 0;
  while(mincdf == 0){
    mincdf = cdf[i];
    i++;
  }

  // denonminator of transfer function
  int d = imageWidth * imageHeight - mincdf;
  for(i = 0; i < HIST_BIN; i++){
    if(cdf[i] - mincdf < 0){
      trans[i] = 0;
    } else {
      trans[i] = (unsigned int)( (float)( (cdf[i] - mincdf) * grayMax ) / (float)(d) );
    }
  }

  for(i = 0; i < imageHeight*imageWidth; i++){
    gold_outLeft[i] = (unsigned char)trans[disparity[i]];
  }

  // clean up memory
  free(paddedLeft);
  free(paddedRight);
  free(shiftedRight);
  free(ssd);
  free(winssd);
  free(minssd);
  free(disparity);

}
