NVCC=nvcc
OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS=-O3 -Wall -Wextra -m64

sobel_bil: sobel_bil_serial sobel_bil.o Makefile
	$(NVCC) -o sobel_bil sobel_bil.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

sobel_bil_serial: sobel_bil_serial.c
	g++ sobel_bil_serial.c -o sobel_bil_serial $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH)

sobel_bil.o: sobel_bil.cu
	nvcc -c sobel_bil.cu $(NVCC_OPTS)
	
clean:
	rm -f *.o *.png sobel_bil
