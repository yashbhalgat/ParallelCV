#include <stdio.h>
#include "utils.h"

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{    
    const int2 p = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y);
    const int m = p.y * numCols + p.x; 
    if(p.x >= numCols || p.y >= numRows)
         return;
    float color = 0.0f;
    
    for(int f_y = 0; f_y < filterWidth; f_y++) {
        for(int f_x = 0; f_x < filterWidth; f_x++) {
   
            int c_x = p.x + f_x - filterWidth/2;
            int c_y = p.y + f_y - filterWidth/2;
            c_x = min(max(c_x, 0), numCols - 1);
            c_y = min(max(c_y, 0), numRows - 1);
            float filter_value = filter[f_y*filterWidth + f_x];
            color += filter_value*static_cast<float>(inputChannel[c_y*numCols + c_x]);
        }
    }
    outputChannel[m] = color;
}

__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  const int2 p = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);
  const int m = p.y * numCols + p.x;    
  if(p.x >= numCols || p.y >= numRows)
      return;
  redChannel[m]   = inputImageRGBA[m].x;
  greenChannel[m] = inputImageRGBA[m].y;
  blueChannel[m]  = inputImageRGBA[m].z;
}

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);
  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  checkCudaErrors(cudaMalloc(&d_filter, sizeof( float) * filterWidth * filterWidth));
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
}

void gaussian_filter(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  const dim3 blockSize(32, 32);
  const dim3 gridSize(numCols/blockSize.x + 1, numRows/blockSize.y + 1);
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
                                            numRows,
                                            numCols,
                                            d_red,
                                            d_green,
                                            d_blue);
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());
  gaussian_blur<<<gridSize, blockSize>>>(
      d_red,
      d_redBlurred,
      numRows,
      numCols,
      d_filter,
      filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  gaussian_blur<<<gridSize, blockSize>>>(
      d_blue,
      d_blueBlurred,
      numRows,
      numCols,
      d_filter,
      filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());  
  gaussian_blur<<<gridSize, blockSize>>>(
      d_green,
      d_greenBlurred,
      numRows,
      numCols,
      d_filter,
      filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
}