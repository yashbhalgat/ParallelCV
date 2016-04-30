#include "Image_Decoder/decoder.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#define _NJ_INCLUDE_HEADER_ONLY

// Code written for image matrices without using openCV
// to avoid overhead
// If ".jpg" file is given as an input, the decoder library
// converts it to ".pgm" and gives it to the program

// Functions for Opening JPEG Image Decoding
extern int njIsColor(void);
extern unsigned char* njGetImage(void);
extern void njDone(void);
extern int njGetWidth(void);
extern nj_result_t njDecode(const void* jpeg, const int size);
extern void njInit(void);

// paramters for bilateral filter
#define SIGMA_DOMAIN 18.0f
#define SIGMA_RANGE 300.0f
#define MAX_THREADS_DIM 16
#define MAX_THREADS 512
#define MAX_BLOCKS 65536	//2^16 blocks
#define BNW "image_bnw.pgm"
#define BIL "image_bil.pgm"
#define SOB "image_sob.pgm"

//Texture memory for input image
texture<float, 2, cudaReadModeElementType> tex_img;

//GPU Error Checking macro
#define GPUerrchk(ans) { GPUassert((ans), __FILE__, __LINE__); }

//Error Checking Manual(Abort)
inline void GPUassert(cudaError_t code, const char *file, int line, bool Abort=true)
{
	if (code != 0) 
	{
        	fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),file,line);
        	if (Abort) exit(code);
    	}
}

//Bilateral Filter kernel size >= 9

// Kernel for Bilateral filter
__global__ void bilFil(float *out,int *rows_,int *cols_)
{
	const int KSIZE = 6;
    int rows = *rows_;
    int cols = *cols_;
    int row = (blockIdx.x*blockDim.x + threadIdx.x);
    int col = (blockIdx.y*blockDim.y + threadIdx.y);
    if (row>(rows-KSIZE) || col>(cols-KSIZE) || row<KSIZE || col<KSIZE)
    	return;
	float org = tex2D(tex_img,col,row),sum=0.0,sumk =0.0;
    for (int k=-KSIZE;k<=KSIZE;k++){
		for (int ij = -KSIZE;ij<= KSIZE;ij++){
        	float cur = tex2D(tex_img,col+k,row+ij);
            float d   = (abs(KSIZE) + abs(KSIZE));
            float f   = sqrt((cur-org)*(cur-org));
            float cf  = expf(-(d*d)/(SIGMA_DOMAIN*SIGMA_DOMAIN*2));
            float sf  = expf(-(f*f)/(SIGMA_RANGE*SIGMA_RANGE*2));
            sum  += cf*sf*cur;
            sumk += cf*sf;
         }
     }
	sum = sum/sumk;
	out[row*cols+col] = sum;
}



//Kernel for Sobel filter
__global__ void kernelImage(float *out,int *rows_,int *cols_)
{
    const int KSIZE = 3;
   	int rows,cols,i,j;
	float sumh=0,sumv=0;
	float pix[KSIZE][KSIZE];
	float sobv[KSIZE][KSIZE] = {-1,-2,-1,0,0,0,1,2,1};
	float sobh[KSIZE][KSIZE] = {-1,0,1,-2,0,2,-1,0,1};
	rows = *rows_;
	cols = *cols_;
    int row = (blockIdx.x*blockDim.x + threadIdx.x);
	int col = (blockIdx.y*blockDim.y + threadIdx.y);
	//unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
	if (row>(rows-KSIZE) || col>(cols-KSIZE) || row<KSIZE || col<KSIZE) //|| row>(cols-KSIZE)) //|| row >rows)
		return;
	for (i=-1;i<=1;i++){
		for (j=-1;j<=1;j++){
			pix[i+1][j+1] = tex2D(tex_img,col+i,row+j);
		}
	}
	
	for (i=0;i<KSIZE;i++){
        for (j=0;j<KSIZE;j++){
            sumv += sobv[i][j]*pix[i][j];
			sumh += sobh[i][j]*pix[i][j];
		}
	}

	sumh = abs(sumh) + abs(sumv);
	sumh = sumh>255?255:(sumh<0?0:sumh);
	out[row*cols+col] = sumh; // in[row*rows+col];
}




//Main host function to call kernel
int main(int argc, char* argv[]) 
{
	unsigned int size,i,j;
	int *d_row,*d_col;
    unsigned char *buf;
    FILE *f;
    unsigned int image_height;
    unsigned int image_width;
	float *bnw,*d_out;//, *d_bnw;
	
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        
	GPUerrchk(cudaSetDevice(0));					//Set device 0 or use first device
    GPUerrchk(cudaDeviceSynchronize());
    GPUerrchk(cudaThreadSynchronize());
	
	if (argc < 2){
   		printf("Usage: %s <input.jpg> [<output1.pgm> <output2.pgm> <output3.pgm>]\n", argv[0]);
    	return 2;
    }

	//Open File for reading JPEG format
	f = fopen(argv[1], "rb");
	if (!f) 
	{
    	printf("Error opening the input file.\n");
    	return 1;
	}

	fseek(f, 0, SEEK_END);
	size = (int) ftell(f);
	buf = (unsigned char *)malloc(size);
	fseek(f, 0, SEEK_SET);
	size = (int) fread(buf, 1, size, f);
	fclose(f);
	njInit();

	// Decode the image
	//Image is decoded using njDecode
	if (njDecode(buf, size)) 
	{
		printf("Error decoding the input file.\n");
		return 1;
	}
	image_height = njGetHeight();
    image_width = njGetWidth();
	
	printf("\nImage Height = %d,Image Width = %d\n",image_height,image_width);

	//Memory Allocation for variables to Host and Device 
	bnw = (float*)malloc(image_height*image_width*sizeof(float));
	unsigned char *ubnw = (unsigned char*)malloc(image_height*image_width*sizeof(unsigned char));
	GPUerrchk(cudaMalloc(&d_out,image_height*image_width*sizeof(float)));
	GPUerrchk(cudaMalloc(&d_row,sizeof(int)));
	GPUerrchk(cudaMalloc(&d_col,sizeof(int)));

	//Get image data
	unsigned char *data = njGetImage();

	//Gray scale image conversion
	for (i=0,j=0;i<image_height*image_width*3;j++)                
	{
        float red    = data[i++];
        float green  = data[i++];
        float blue   = data[i++];
		float sum = 0.21*red + 0.72*green + 0.07*blue;
		bnw[j] = sum;
		ubnw[j] = (unsigned char)sum;
    }

    //Writing to image file - Grayscale output
    f = fopen((argc > 2) ? argv[2] : BNW, "wb");
    if (!f)
    {
        printf("Error opening the output file.\n");
        return 1;
    }
    fprintf(f, "P%d\n%d %d\n255\n",5, image_width, image_height);
    fwrite(ubnw, 1, image_height*image_width, f);
    fclose(f);

	GPUerrchk(cudaMemcpy(d_row,&image_height,sizeof(int),cudaMemcpyHostToDevice));
	GPUerrchk(cudaMemcpy(d_col,&image_width,sizeof(int),cudaMemcpyHostToDevice));
	cudaArray* cuArray;
    cudaMallocArray(&cuArray, &desc, image_width, image_height);
	cudaMemcpyToArray(cuArray, 0, 0, bnw, image_width*image_height*sizeof(float), cudaMemcpyHostToDevice);
	
	// Using texture memory
	tex_img.addressMode[0] = cudaAddressModeWrap;
	tex_img.addressMode[1] = cudaAddressModeWrap;
	tex_img.filterMode = cudaFilterModePoint;
	tex_img.normalized = false;
	if (cudaBindTextureToArray(tex_img, cuArray, desc) != cudaSuccess) {
		printf("failed to bind texture: %s\n", cudaGetErrorString(cudaGetLastError()));
		free(bnw);
		free(ubnw);
		cudaFree(d_col);cudaFree(d_row);//cudaFree(d_bnw);
		cudaFree(d_out);
		return -2;
	}

	//Initializing Variables for thread size and block size
    dim3 th_per_blk,blk_per_grid;

    if ((image_height)/MAX_THREADS_DIM<MAX_BLOCKS && (image_width)/MAX_THREADS_DIM<MAX_BLOCKS)                       //Check maximum (ROW and COL) blocks count per Grid
	{
		th_per_blk = dim3(MAX_THREADS_DIM,MAX_THREADS_DIM,1);
		blk_per_grid = dim3((image_height+MAX_THREADS_DIM-1)/MAX_THREADS_DIM,(image_width+MAX_THREADS_DIM-1)/MAX_THREADS_DIM,1);
		printf("\nTotal Threads = %d, Blocks = %d \n",MAX_THREADS_DIM*MAX_THREADS_DIM*((image_height-1)/MAX_THREADS_DIM+1)*((image_width-1)/MAX_THREADS_DIM+1),(image_height*image_width+MAX_THREADS_DIM-1)/MAX_THREADS_DIM);
	}
	else
	{
		printf("Cannot have more threads");
		exit(1);
	}



	//Bilateral kernel execution
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	bilFil<<<blk_per_grid,th_per_blk>>>(d_out,d_row,d_col);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("elapsed time for bilateral is %f ms", elapsed_time*20);

	GPUerrchk(cudaMemcpy(bnw,d_out,image_height*image_width*sizeof(float),cudaMemcpyDeviceToHost));

	// Converting pixel values to unsigned char
	for (i=0;i<image_height*image_width;i++){
            ubnw[i] = (unsigned char)bnw[i];
    }

	//Writing to image file - Bilateral Filter output
    f = fopen((argc > 3) ? argv[3] : BIL, "wb");
    if (!f)
    {
            printf("Error opening the output file.\n");
            return 1;
    }
    fprintf(f, "P%d\n%d %d\n255\n",5, image_width, image_height);
    fwrite(ubnw, 1, image_height*image_width, f);
    fclose(f);


    // Executing kernel for Sobel filter
    cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, 0);

	kernelImage<<<blk_per_grid,th_per_blk>>>(d_out,d_row,d_col);
	
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	float elapsed_time1;
	cudaEventElapsedTime(&elapsed_time1, start1, stop1);
	// cout<<"elapsed time for sobel is "<<elapsed_time1<<" ms"<<endl;
	printf("elapsed time for sobel is %f ms", elapsed_time1*20);

	GPUerrchk(cudaPeekAtLastError());
	cudaThreadSynchronize();
	GPUerrchk(cudaMemcpy(bnw,d_out,image_height*image_width*sizeof(float),cudaMemcpyDeviceToHost));
	for (i = 0;i<image_height*image_width;i++)
        {
		ubnw[i] = (unsigned char)bnw[i];
	}

	//Writing to image file - Sobel Filter output
	f = fopen((argc > 4) ? argv[4] : SOB, "wb");
    if (!f) {
        printf("Error opening the output file.\n");
        return 1;
    }
    fprintf(f, "P%d\n%d %d\n255\n",5, image_width, image_height);
    fwrite(ubnw, 1, image_height*image_width, f);
	printf("Exexuted\n");
    fclose(f);


    // Free memory used
	cudaUnbindTexture (tex_img);
	njDone();
	cudaFree(d_out);
	cudaFree(d_row);
	free(buf);
	cudaFree(d_col);
	free(ubnw);
	free(bnw);
	cudaFreeArray(cuArray);
    return 0;
}
