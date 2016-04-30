/*
 * Template for 5kk73 GPU assignment
 */

/**********************************************
 * Dense Stereo Vison
 * Host code
 *
 ********************************************/

// Utilities and system includes
#include <shrUtils.h>
#include "cutil_inline.h"
#include <time.h>

// includes, headers
#include "stereo.h"

// includes, kernels
#include <stereo_kernel.cu>

/******************************************************************
 *
 * Function list
 *
 *****************************************************************/

// CPU code for stereo vision 
extern "C"
void computeGold(unsigned char* h_inLeft, unsigned char* h_inRight, unsigned char * gold_outLeft, int imageHeight, int imageWidth, int grayMax);

// read in PGM image
extern "C"
int readPGM(unsigned char **image, char *filename,		\
	    int *imageWidth, int *imageHeight, int *grayMax);

// write out PGM image
extern "C"
int writePGM(unsigned char *image, char *filename,		\
	     int imageWidth, int imageHeight, int grayMax);

// sample name
static char *sSDKsample = "stereo";

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
   
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv){
  printf("[ %s ]\n", sSDKsample);
  
  shrSetLogFileName ("stereo.txt");
  shrLog("%s Starting...\n\n", argv[0]);
  
  // everything is here
  runTest(argc, argv);
  
  shrEXIT(argc, (const char**)argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv){
  
  /******************************************************************
   *
   * Query of the device
   *
   *****************************************************************/
  
  if(shrCheckCmdLineFlag(argc, (const char**)argv, "device")){
    cutilDeviceInit(argc, argv);
  }
  else{
    cudaSetDevice(cutGetMaxGflopsDeviceId());
  }

  int devID;
  cudaDeviceProp props;
  
  // get number of SMs on this GPU
  cutilSafeCall(cudaGetDevice(&devID));
  cutilSafeCall(cudaGetDeviceProperties(&props, devID));
  
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

  /******************************************************************
   *
   * Starting the image IO function
   *
   *****************************************************************/

  // pointer to the left image 
  unsigned char *h_inLeft;
  // pointer to the right image
  unsigned char *h_inRight;
  // image width (original width without padding)
  int imageWidth;
  // image height (original height without padding)
  int imageHeight;
  // max gray value of the PGM, usually is 255.
  int grayMax;
  // path and file name of the left image
  char inLeftName[100]="./dataset/inLeft.pgm";
  // path and file name of the right image
  char inRightName[100]="./dataset/inRight.pgm";
  // path and file name of the disparity map for the left image (CPU)
  char outLeftGoldName[100]="./dataset/outLeftGold.pgm";
  // path and file name of the disparity map for the left image (GPU)
  char outLeftGPUName[100]="./dataset/outLeftGPU.pgm";

  // read left image and allocate memory space for it
  readPGM(&h_inLeft, inLeftName,			\
	  &imageWidth, &imageHeight, &grayMax);
  // read right image and allocate memory space for it
  readPGM(&h_inRight, inRightName,			\
	  &imageWidth, &imageHeight, &grayMax);
    
  /******************************************************************
   *
   * Allocate device, host memory, and perform memory copy
   *
   *****************************************************************/

  // device memory allocation for left and right images
  int size_inLeft = imageHeight * imageWidth;
  int size_inRight = imageHeight * imageWidth;
  int mem_size_inLeft = sizeof(unsigned char) * size_inLeft;
  int mem_size_inRight = sizeof(unsigned char) * size_inRight;
  // pointer to the left image on the device (GPU) memory
  unsigned char* d_inLeft;
  cutilSafeCall(cudaMalloc((void**) &d_inLeft, mem_size_inLeft));
  // pointer to the right image on the device (GPU) memory
  unsigned char* d_inRight;
  cutilSafeCall(cudaMalloc((void**) &d_inRight, mem_size_inRight));

  // copy left and right image from host memory to device
  cutilSafeCall(cudaMemcpy(d_inLeft, h_inLeft, mem_size_inLeft,
			   cudaMemcpyHostToDevice) );
  cutilSafeCall(cudaMemcpy(d_inRight, h_inRight, mem_size_inRight,
			   cudaMemcpyHostToDevice) );

  // allocate device memory for disparity 
  unsigned int size_outLeft = imageHeight * imageWidth;
  unsigned int mem_size_outLeft = sizeof(unsigned char) * size_outLeft;
  // pointer to the left disparity map on the device (GPU) memory
  unsigned char* d_outLeft;
  cutilSafeCall(cudaMalloc((void**) &d_outLeft, mem_size_outLeft));

  // pointer to the left disparity map on the host (CPU) memory
  unsigned char* h_outLeft = (unsigned char*) malloc(mem_size_outLeft);

  /******************************************************************
   *
   * Run stereo vision on GPU (empty template)
   *
   *****************************************************************/


  // setup execution parameters
  // thread block size
  dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  // grid size
  dim3 grid(imageWidth / (BLOCK_SIZE_X * 4) + 1, imageHeight / BLOCK_SIZE_Y + 1);

  // kernel warmup
  stereo<<< grid, threads >>>(d_inLeft, d_inRight, d_outLeft, imageWidth);
  cudaThreadSynchronize();
    
  // create and start timer, CUDA toolkit provides a timer more precise than ANSI C
  shrLog("GPU running empty Kernels...\n\n");
  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));

  // execute the kernel on GPU
  // It is an empty kernel, you need to fill in your own stuff.
  // Iterate n times and then average the running time.
  int nIter = 10;
  for (int j = 0; j < nIter; j++) {
    stereo<<< grid, threads >>>(d_inLeft, d_inRight, d_outLeft, imageWidth);
  }
    
  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");

  cudaThreadSynchronize();

  // stop and destroy timer
  cutilCheckError(cutStopTimer(timer));
  double dSeconds = cutGetTimerValue(timer)/((double)nIter * 1000.0);

  //Log througput, etc
  shrLogEx(LOGBOTH | MASTER, 0, "GPU empty kernel, Time = %.5f s\n\n", dSeconds);
  cutilCheckError(cutDeleteTimer(timer));

  // copy result from device to host
  cutilSafeCall(cudaMemcpy(h_outLeft, d_outLeft, mem_size_outLeft,
			   cudaMemcpyDeviceToHost) );

  // write GPU disparity map to the output file.
  writePGM(h_outLeft, outLeftGPUName,		\
	   imageWidth, imageHeight, grayMax);

  /******************************************************************
   *
   * Run stereo vision on CPU (gold reference)
   *
   *****************************************************************/
    
  // allocate host memory for the gold result
  unsigned char* gold_outLeft = (unsigned char*) malloc(mem_size_outLeft);

  // ANSI C timer, not precise, but enough for CPU timing
  clock_t start, end;
  double elapsed;  

  //start timer
  start = clock();

  // compute the gold disparity on the host (CPU)
  computeGold(h_inLeft, h_inRight, gold_outLeft, imageHeight, imageWidth, grayMax);

  // end timer
  end = clock();
  elapsed = (double)(end - start) / CLOCKS_PER_SEC;
  printf("CPU stereo vision time: %f s\n", elapsed);

  // write gold disparity map to the output file.
  writePGM(gold_outLeft, outLeftGoldName,		\
	   imageWidth, imageHeight, grayMax);


  // clean up memory
  free(h_inLeft);
  free(h_inRight);
  free(h_outLeft);
  free(gold_outLeft);
  cutilSafeCall(cudaFree(d_inLeft));
  cutilSafeCall(cudaFree(d_inRight));
  cutilSafeCall(cudaFree(d_outLeft));

  cudaThreadExit();
}

