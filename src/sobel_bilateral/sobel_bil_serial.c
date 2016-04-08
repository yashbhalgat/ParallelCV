// Code written for image matrices without using openCV
// to avoid overhead
// If ".jpg" file is given as an input, the decoder library
// converts it to ".pgm" and gives it to the program

#include "Image_Decoder/decoder.c"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#define _NJ_INCLUDE_HEADER_ONLY

extern int njIsColor(void);
extern unsigned char* njGetImage(void);
extern void njDone(void);
extern int njGetWidth(void);
extern nj_result_t njDecode(const void* jpeg, const int size);
extern void njInit(void);


int main(int argc, char* argv[]) {
	unsigned int size,i,j;
    unsigned char *buf;
    FILE *f;
    unsigned int H, W;
    unsigned char *r,*g,*b; 
    float *bnw;

    // Input error handling
	if (argc < 2) {
    	printf("Usage: %s <input.jpg> [<output.pgm>]\n", argv[0]);
    	return 2;
	}
	f = fopen(argv[1], "rb");
	if (!f) {
    	printf("Error opening the input file.\n");
    	return 1;
	}
	
    // File operations
    // Set indicator to end of file
    fseek(f, 0, SEEK_END);
	size = (int) ftell(f);
	buf = (unsigned char*)malloc(size);
    // Set indicator to beginning of file
	fseek(f, 0, SEEK_SET);
	size = (int) fread(buf, 1, size, f);
	fclose(f);

	njInit();
	if (njDecode(buf, size)) {
		printf("Error decoding the input file.\n");
		return 1;
	}

	H = njGetHeight();
    W = njGetWidth();

	bnw = (float*)malloc(H*W*sizeof(float));
	unsigned char *ubnw = (unsigned char*)malloc(H*W*sizeof(unsigned char));

	f = fopen((argc > 2) ? argv[2] : ("output_bnw.pgm"), "wb");
	if (!f) {
    	printf("Error opening the output file.\n");
    	return 1;
	}
    
    unsigned char *data = njGetImage();
	for (i=0,j=0;i<H*W*3;j++){
    	unsigned int temp=0;float r,g,b;
        r    = data[i++];
        g  = data[i++];
        b   = data[i++];
        bnw[j] = (float)(0.21*r + 0.72*g + 0.07*b);
		ubnw[j] = (int)(0.21*r + 0.72*g + 0.07*b);
    }

	fprintf(f, "P%d\n%d %d\n255\n", 5, njGetWidth(), njGetHeight());
	fwrite(ubnw, 1, H*W, f);
	fclose(f);

	#define SIGD 18.0f
	#define SIGR 300.0f

 	for (i=3;i<H-3;i++){
        for (j=3;j<W-3;j++){
        	float org = ubnw[i*W+j],sum=0.0,sumk =0.0;
            for (int k=i-3;k<=i+3;k++){
                for (int ij = j-3;ij<=j+3;ij++){
        			float cur = (float)ubnw[k*W+ij];
        			float d   = ((float)((i-k)*(i-k) + (j-ij)*(j-ij)));
        			float f   = sqrt((cur-org)*(cur-org));
        			float cf  = expf(-(d*d)/(SIGD*SIGD*2));
        			float sf  = expf(-(f*f)/(SIGR*SIGR*2));
        			sum  += cf*sf*cur;
        			sumk += cf*sf;
		        }
            }
		sum = sum/sumk;
		bnw[i*W+j] = sum;
		}
	}
	printf("\ni = %d, j = %d\nIm_h = %d, Im_w = %d",i,j,H,W);
	f = fopen((argc > 2) ? argv[2] : ("output_bil.pgm"), "wb");
    if (!f) {
        printf("Error opening the output file.\n");
        return 1;
    }

    fprintf(f, "P%d\n%d %d\n255\n", 5, njGetWidth(), njGetHeight());
    fwrite(ubnw, 1, H*W, f);
    fclose(f);

    // Filters for Sobel operation
	float win[] = {-1.0,0,1.0,-2.0,0.0,2.0,-1.0,0.0,1.0};
    int m=0;
	float win_[] = {-1.0,-2.0,-1.0,0,0,0,1.0,2.0,1.0};
    m=0;


    // Windowing convolution
	memset((void*)ubnw,0,H*W*sizeof(unsigned char));
	for(int i=1;i<H-1;i++){
        for(j=1;j<W-1;j++){
			float sumv=0,sumh=0,sum=0;
			int k,ij,h=W;
            for (m=0,k=(i-1);k<=(i+1);k++){
                for (ij = j-1;ij<=j+1;ij++){
                    sumv += win_[m]*bnw[k*h+ij];
					sumh += win[m++]*bnw[k*h+ij];		
                }
            }
			sum = abs(sumv)+abs(sumh);
			ubnw[i*h+j] = (unsigned char)(abs(sum)>255?255:(abs(sum)<0?0:abs(sum)));
        }
    }
	
    printf("\th = %d, w = %d\n",H,W);
	
    // Write to file
    f = fopen((argc > 2) ? argv[2] : ("output_sob.pgm"),"wb");
    if (!f) {
            printf("Error opening the output file.\n");
            return 1;
    }
    fprintf(f, "P%d\n%d %d\n255\n", 5, njGetWidth(), njGetHeight());
    fwrite(ubnw, 1, H*W, f);
    fclose(f);

	njDone();
	// freeing memory
    free(buf);
	free(ubnw);
	free(bnw);
	return 0;
}



