#include "utils.h"
#include <stdio.h>

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  int y = threadIdx.y+ blockIdx.y*32;
  int x = threadIdx.x+ blockIdx.x*32;
  if (y < numCols && x < numRows) {
  	int index = numRows*y +x;
  uchar4 color = rgbaImage[index];
  unsigned char grey = (unsigned char)(0.299f*color.x+ 0.587f*color.y + 0.114f*color.z);
  greyImage[index] = grey;
  }
}

void rgb2gray(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  int   blockWidth = 32;
  
  const dim3 blockSize(blockWidth, blockWidth, 1);
  int   blocksX = numRows/blockWidth+1;
  int   blocksY = numCols/blockWidth+1; //TODO
  const dim3 gridSize( blocksX, blocksY, 1);  //TODO
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}