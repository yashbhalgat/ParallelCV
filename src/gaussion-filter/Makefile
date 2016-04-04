NVCC=nvcc
OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include

NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS=-O3 -Wall -Wextra -m64

student: main.o kernel.o Makefile
	$(NVCC) -o gaussian_filter main.o kernel.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: main.cpp timer.h utils.h gaussianFilter.cpp
	g++ -c main.cpp $(GCC_OPTS) -I $(OPENCV_INCLUDEPATH) -I $(CUDA_INCLUDEPATH)

kernel.o: kernel.cu gaussianFilter.cpp utils.h
	nvcc -c kernel.cu $(NVCC_OPTS)

clean:
	rm -f *.o *.png hw
